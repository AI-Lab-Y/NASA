import speck as sp
import numpy as np
import heapq
import time

from pickle import dump
from keras.models import Model, load_model, model_from_json
from os import urandom, path, mkdir


MASK_VAL = 2 ** sp.WORD_SIZE() - 1
word_size = sp.WORD_SIZE()


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + word_size * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]

    return new_x


# make a homogeneous set from a random plaintext pair
# diff: difference of the plaintext pair
# neutral_bits is used to form the homogeneous set
def make_homogeneous_set(diff=(0x211, 0xa04), neutral_bits=[10,11,12]):
    p0l = np.frombuffer(urandom(2), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2), dtype=np.uint16)
    for i in neutral_bits:
        if isinstance(i, int):
            i = [i]
        d0 = 0; d1 = 0
        for j in i:
            d = 1 << j
            d0 |= d >> 16
            d1 |= d & 0xffff
        p0l = np.concatenate([p0l, p0l ^ d0])
        p0r = np.concatenate([p0r, p0r ^ d1])
    p1l = p0l ^ diff[0]; p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r


# generate a new set of key guess candidates for bayesian key search
# k_old: key guess candidates set from the previous iteration
# T: corresponding statistics T of k_old
# key_length: key search space is (2**key_length)
# u_d: mean value of statics T
# sigma_d: the reciprocal of standard deviation of statics T
# return: array of new key guess candidates
def gen_keys_candidate(k_old, T, key_length, u_d, sigma_d):
    n_cand = len(k_old)
    # traverse key search space: [0,2**key_length)
    k = np.array(range(2**key_length))
    k = np.repeat(k, n_cand)
    T_tmp = np.tile(T, 2**key_length)
    k_tmp = np.tile(k_old, 2**key_length)
    k_tmp = k_tmp ^ k
    # d: calculate the hamming distance between key guesses and previous key candidates
    d = np.zeros(len(k_tmp), dtype=np.uint32)
    for i in range(key_length):
        d = d + (k_tmp & 1)
        k_tmp = k_tmp >> 1
    # scores: calculate score for every key guess
    scores = (T_tmp - u_d[d]) * sigma_d[d]
    scores = np.reshape(scores, (-1, n_cand))
    scores = np.linalg.norm(scores, ord=2, axis=1)
    k = np.argsort(scores)
    # pick n_cand key guesses with the n_cand smallest score to form the new set of key guess candidates
    return k[0:n_cand]


# generate u_d and sigma_d used by gen_keys_candidate for bayesian key search
def get_u_d_s_d(n, p0, p2_d1, p3):
    u_d = n * (p0 * p2_d1 + (1 - p0) * p3)
    s_d = np.sqrt(n * p0 * p2_d1 * (1 - p2_d1) + n * (1 - p0) * p3 * (1 - p3))
    s_d = 1 / s_d
    return u_d, s_d


# perform bayesian key search algorithm
# c is a tuple of ciphertest pairs array (c0l, c0r, c1l, c1r)
# u_d and s_d are related to net and will be used in gen_keys_candidate
# net, bits, c3 and c will be used to compute the statistic T
# bits can be full_bits(teacher distinguisher) or selected_bits(student distinguisher)
# n_cand and n_iter are bayesian parameters in bayesian key search algorithm
# th is used to distinguish surviving key from all key candidates
# return (survive_key, cnt). survive_key is the surviving key list and cnt is the corresponding statics T of surviving keys
def bayesian_search(c, net, u_d, s_d, c3=0.55, th=1636227, n_cand=32, n_iter=3, key_guess_length=8):
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]
    n = len(c0l)

    # meybe this can speed up the inference of neural distinguisher when n is small
    if n * n_cand < 10**7:
        speed_flag = True
    else:
        speed_flag = False

    if speed_flag:
        c0l = np.tile(c0l, n_cand)
        c0r = np.tile(c0r, n_cand)
        c1l = np.tile(c1l, n_cand)
        c1r = np.tile(c1r, n_cand)
    key_cand = np.random.choice(2**key_guess_length, n_cand, False)
    T_cand = np.zeros(n_cand, dtype=np.uint32)
    cnt = np.zeros(2**key_guess_length, dtype=np.uint32)

    # begin bayesian search
    for i in range(n_iter):
        if speed_flag:
            key_guess = np.repeat(key_cand, n)
            t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
            t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
            x = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            z = net.predict(x, batch_size=10000)
            z = np.squeeze(z)
            z = np.reshape(z, (n_cand, n))
            T_cand = np.sum(z > c3, axis=1)
            for j in range(n_cand):
                cnt[key_cand[j]] = T_cand[j]
        else:
            for j in range(n_cand):
                key_guess = key_cand[j]
                t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
                t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
                x = sp.convert_to_binary([t0l, t0r, t1l, t1r])
                z = net.predict(x, batch_size=2**16)
                z = np.squeeze(z)
                T_cand[j] = np.sum(z > c3)
                cnt[key_cand[j]] = T_cand[j]
        if i == n_iter - 1:
            break
        # generate key guess candidates for next iteration
        key_cand = gen_keys_candidate(key_cand, T_cand, key_guess_length, u_d, s_d)
    survive_key = [i for i in range(2**key_guess_length) if cnt[i] > th]
    return survive_key, cnt


def attack_with_ND8T(c, net, u_d, s_d, th=13064, c3=0.5):
    n_cand = 256
    n_iter = 5
    key_guess_length = 16
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]

    survive_key = bayesian_search((c0l, c0r, c1l, c1r), net, u_d, s_d, c3, th, n_cand, n_iter,
                                  key_guess_length=key_guess_length)[0]
    return survive_key


# use ND7s to perform valid homogeneous set identification as well as sk11[0~7] recovery attack
# M, th_M and ts is used for identifying valid homogeneous set
# n and th_n is used for recovery attack
# n_cand, search_iter, u_d and s_d is used for bayesian key search
# return (survive_key_list, is_valid_homogeneous_set):
#   survive_key_list is the surviving keys in stage1 attack
#   is_valid_homogeneous_set implies whether the homogeneous passed valid homogeneous set test
def attack_with_ND7s(c, th_M, ts, n, th_n, c3, net, u_d, s_d, c_bits=[10,11,12], survive_key=None):
    n_cand, search_iter = 32, 3
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]
    key_guess_length = 8

    for kg_12 in survive_key:
        d0l, d0r = sp.dec_one_round((c0l, c0r), kg_12)
        d1l, d1r = sp.dec_one_round((c1l, c1r), kg_12)
        cnt_M = np.zeros(2 ** key_guess_length, dtype=np.uint32)
        cnt_n = np.zeros(2 ** key_guess_length, dtype=np.uint32)
        key_cand = np.random.choice(2 ** key_guess_length, n_cand, False)
        T_cand = np.zeros(n_cand, dtype=np.uint32)

        # begin bayesian search
        for i in range(search_iter):
            for j in range(n_cand):
                key_guess = key_cand[j]
                t0l, t0r = sp.dec_one_round((d0l, d0r), key_guess)
                t1l, t1r = sp.dec_one_round((d1l, d1r), key_guess)
                raw_x = sp.convert_to_binary([t0l, t0r, t1l, t1r])
                x = extract_sensitive_bits(raw_x, c_bits)
                z = net.predict(x, batch_size=10000)
                z = np.squeeze(z)
                T_cand[j] = np.sum(z > c3)
                cnt_M[key_guess] = T_cand[j]
                cnt_n[key_guess] = np.sum(z[:n] > c3)
            key_cand = gen_keys_candidate(key_cand, T_cand, key_guess_length, u_d, s_d)
        valid_index = np.sum(cnt_M > th_M)
        if valid_index >= ts:
            # the homogeneous set passes test and is a valid plaintext structure
            survive_key = [i for i in range(2 ** key_guess_length) if cnt_n[i] > th_n]
            return survive_key, True
    # the homogeneous set does not pass the test and we don't need to return surviving keys
    return [], False


def output_surviving_key_difference(sur_key, sk12, sk11):
    if len(sur_key) == 0:
        return
    print('Difference between surviving (kg_12, kg_11[7~0]) and (sk_12, sk_11[7~0]) are ')
    for i in range(len(sur_key)):
        print((hex(sur_key[i][0] ^ sk12), hex((sur_key[i][1] ^ sk11) & 0xff)))


def key_recovery_on_12_round(t, diff=(0x211, 0xa04), neutral_bits=[10,11,12]):
    # load 8-round ND
    json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
    json_model = json_file.read()
    net8 = model_from_json(json_model)
    net8.load_weights('./saved_model/teacher/0x0040-0x0/net8_small.h5')
    net8.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net7s = load_model('./saved_model/student/0x0040-0x0/hard_label/student_7_distinguisher.h5')

    # settings of key recovery
    nr = 12
    p0 = 2 ** (-6)
    n = (353518, 22586)
    th_n = (175328, 6690)
    # settings of valid plaintext structure identification
    M = 37938
    th_M = 11103
    ts = 8
    # parameters related to NDs, 17 / 9 normal distributions
    c_bits = [14, 13, 12, 11, 10, 9, 8, 7]
    p3 = (0.4913596, 0.2862612)
    ND8t_p2_d1 = np.array([0.5183955, 0.5056154, 0.4992611, 0.4957098, 0.4939058, 0.4927488, 0.4924816, 0.4918235, 0.4916707,
                          0.4914085, 0.4913381, 0.4913277, 0.4911237, 0.4913378, 0.4914284, 0.49097, 0.4914429])
    ND7s_p2_d1 = np.load('./p2_estimation_res/student/0x0040-0x0/{0}/{0}_p2_d1.npy'.format(7))
    u_d_8, s_d_8 = get_u_d_s_d(n[0], p0, ND8t_p2_d1, p3[0])
    u_d_7, s_d_7 = get_u_d_s_d(n[1], p0, ND7s_p2_d1, p3[1])

    # some statistics of the attack
    true_success, true_fail, false_success, false_fail = 0, 0, 0, 0
    survive_num = np.zeros(t, dtype=np.uint32)
    consumption_num = np.zeros(t, dtype=np.uint32)
    is_valid = np.zeros(t, dtype=np.uint8)
    is_success = np.zeros(t, dtype=np.uint8)
    attack_time = np.zeros(t)

    for attack_index in range(t):
        start = time.time()
        key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(key, nr)
        sk_11, sk_12 = ks[nr - 2][0], ks[nr - 1][0]
        valid_homogeneous_set = False
        homogeneous_set_consumption = 0
        # identify valid homogeneous set and recover keys
        while not valid_homogeneous_set:
            if homogeneous_set_consumption >= 18:
                print('attack failed because no valid homogeneous set can be found')
                break
            homogeneous_set_consumption += 1
            p0l, p0r, p1l, p1r = make_homogeneous_set(diff=diff, neutral_bits=neutral_bits)
            p0l, p0r = sp.dec_one_round((p0l[0:n[0]], p0r[0:n[0]]), 0)
            p1l, p1r = sp.dec_one_round((p1l[0:n[0]], p1r[0:n[0]]), 0)
            c0l, c0r = sp.encrypt((p0l, p0r), ks)
            c1l, c1r = sp.encrypt((p1l, p1r), ks)
            survive_key_1 = attack_with_ND8T((c0l, c0r, c1l, c1r), net=net8, u_d=u_d_8, s_d=s_d_8, th=th_n[0], c3=0.5)
            c0l, c0r, c1l, c1r = c0l[:M], c0r[:M], c1l[:M], c1r[:M]
            survive_key_2, valid_homogeneous_set = attack_with_ND7s((c0l, c0r, c1l, c1r), th_M=th_M, ts=ts, n=n[1],
                                                                    th_n=th_n[1], c3=0.55, net=net7s, u_d=u_d_7,
                                                                    s_d=s_d_7, c_bits=c_bits, survive_key=survive_key_1)
            print(homogeneous_set_consumption, ' plaintext structures generated.')
            if valid_homogeneous_set:
                # find a valid homogeneous set
                t0l, t0r = sp.encrypt((p0l, p0r), ks[:2])
                t1l, t1r = sp.encrypt((p1l, p1r), ks[:2])
                good_num = ((t0l ^ t1l) == 0x2800) * ((t0r ^ t1r) == 0x10)
                good_num = np.sum(good_num)
                if good_num == n[0]:
                    print('valid homogeneous set')
                    is_valid[attack_index] = 1
                else:
                    print('invalid homogeneous set')
                    is_valid[attack_index] = 0

        end = time.time()
        output_surviving_key_difference(sur_key=survive_key_2, sk12=sk_12, sk11=sk_11)
        attack_time[attack_index] = end - start
        print('time cost: {}s'.format(attack_time[attack_index]))

        if (sk_12, sk_11 & 0xff) in survive_key_2:
            print('true key survived.')
            is_success[attack_index] = 1
            if is_valid[attack_index] == 1:
                true_success += 1
            else:
                false_success += 1
        else:
            print('true key didn\'t survived')
            is_success[attack_index] = 0
            if is_valid[attack_index] == 1:
                true_fail += 1
            else:
                false_fail += 1
    average_surviving_num = np.mean(survive_num)
    print('attack success time: {}'.format(np.sum(is_success)))
    print('get valid homogeneous set time: {}'.format(np.sum(is_valid)))
    print('success time with valid homogeneous sets: {}'.format(true_success))
    print('failure time with valid homogeneous sets: {}'.format(true_fail))
    print('success time with invalid homogeneous sets: {}'.format(false_success))
    print('failure time with invalid homogeneous sets: {}'.format(false_fail))
    print('average surviving key in stage1: {}'.format(average_surviving_num))
    print('average attack time: {}s'.format(np.mean(attack_time)))
    print('average homogeneous set consumption: {}'.format(np.mean(consumption_num)))

    folder = './key_recovery_record/bayesian/3_8_6_{}_{}_{}_{}'.format(n[0], n[1], n[2], n[3])
    if not path.exists(folder):
        mkdir(folder)
    folder += '/'
    np.save(folder + 'is_valid.npy', is_valid)
    np.save(folder + 'is_success.npy', is_success)
    np.save(folder + 'survive_num.npy', survive_num)
    np.save(folder + 'consumption_num.npy', consumption_num)
    np.save(folder + 'attack_time.npy', attack_time)


def key_recovery_on_12_round_v2(t, diff=(0x211, 0xa04), neutral_bits=[10,11,12]):
    # load 8-round ND
    json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
    json_model = json_file.read()
    net8 = model_from_json(json_model)
    net8.load_weights('./saved_model/teacher/0x0040-0x0/net8_small.h5')
    net8.compile(optimizer='adam', loss='mse', metrics=['acc'])

    # settings of key recovery
    nr = 12
    p0 = 2 ** (-6)
    n = 353518
    th_n = 175328
    # parameters related to NDs, 17 / 9 normal distributions
    p3 = 0.4913596
    ND8t_p2_d1 = np.array([0.5183955, 0.5056154, 0.4992611, 0.4957098, 0.4939058, 0.4927488, 0.4924816, 0.4918235, 0.4916707,
                          0.4914085, 0.4913381, 0.4913277, 0.4911237, 0.4913378, 0.4914284, 0.49097, 0.4914429])
    u_d_8, s_d_8 = get_u_d_s_d(n, p0, ND8t_p2_d1, p3)

    acc = 0
    for attack_index in range(t):
        key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(key, nr)
        sk_12 = ks[nr - 1][0]
        for j in range(18):
            print(j, ' plaintext structures generated.')
            p0l, p0r, p1l, p1r = make_homogeneous_set(diff=diff, neutral_bits=neutral_bits)
            p0l, p0r = sp.dec_one_round((p0l[0:n], p0r[0:n]), 0)
            p1l, p1r = sp.dec_one_round((p1l[0:n], p1r[0:n]), 0)
            c0l, c0r = sp.encrypt((p0l, p0r), ks)
            c1l, c1r = sp.encrypt((p1l, p1r), ks)
            survive_key = attack_with_ND8T((c0l, c0r, c1l, c1r), net=net8, u_d=u_d_8, s_d=s_d_8, th=th_n, c3=0.5)
            if sk_12 in survive_key:
                acc = acc + 1
                print('true sk_12 survive')
                print('the number of surviving keys is ', len(survive_key))
                break


# Attack on 12 round Speck32/64
neutral_bits = [11,14,15,20,21,22,0,1,3,4,5,6,23,24,26,27,28,29,30]
key_recovery_on_12_round(t=10, diff=(0x211, 0xa04), neutral_bits=neutral_bits)
# key_recovery_on_12_round_v2(t=100, diff=(0x211, 0xa04), neutral_bits=neutral_bits)

