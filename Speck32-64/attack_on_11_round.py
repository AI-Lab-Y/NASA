import numpy as np
import speck as sp
from keras.models import load_model
from os import urandom, path, mkdir
import time

word_size = sp.WORD_SIZE()
full_bits = [15,14,13,12,11,10,9,8, 7,6,5,4,3,2,1,0]
selected_bits = [14, 13, 12, 11, 10, 9, 8, 7]
# neutral_bits_1 = [0,11,14,15,20,21,22,26,1,3,4,5,23,24,27,28]   # the neutrality of first 8 bits is 1
neutral_bits_1 = [0,1,3,4,5,11,14,15,20,21,22,23,24,26,27,28]
neutral_bits_2 = [20,21,22,[9,16],[2,11,25],14,15,[6,29],23,30,7]
neutral_bits_3 = [20,21,22,[9,16],[2,11,25],14,15,[6,29],23]


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


def extract_sensitive_bits(raw_x, bits=selected_bits):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    id1 = [v + word_size * i for i in range(4) for v in id0]
    new_x = raw_x[:, id1]
    return new_x


def make_target_diff_samples(n=5697905, nr=11, diff=(0x211,0xa04)):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
    key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(key, nr)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)
    return c0l, c0r, c1l, c1r, ks[nr - 1][0]


# perform bayesian key search algorithm
# c is a tuple of ciphertest pairs array (c0l, c0r, c1l, c1r)
# u_d and s_d are related to net and will be used in gen_keys_candidate
# net, bits, c3 and c will be used to compute the statistic T
# bits can be full_bits(teacher distinguisher) or selected_bits(student distinguisher)
# n_cand and n_iter are bayesian parameters in bayesian key search algorithm
# when hi_guess is True and key_guess_length is 8, the key search space is sk[8~15] and the corresponding sk[0~7] is fixed to be key_low
# when hi_guess is False, the key search space is sk[0~key_guess_length-1]
# th is used to distinguish surviving key from all key candidates
# return (survive_key, cnt). survive_key is the surviving key list and cnt is the corresponding statics T of surviving keys
def bayesian_search(c, net, u_d, s_d, c3=0.55, th=1636227, n_cand=32, n_iter=3, bits=selected_bits, hi_guess=False, key_low=None, key_guess_length=8):
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
            if hi_guess:
                key_guess = (key_guess << 8) | (key_low & 0xff)
            t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
            t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
            raw_x = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            x = extract_sensitive_bits(raw_x, bits)
            z = net.predict(x, batch_size=10000)
            z = np.squeeze(z)
            z = np.reshape(z, (n_cand, n))
            T_cand = np.sum(z > c3, axis=1)
            for j in range(n_cand):
                cnt[key_cand[j]] = T_cand[j]
        else:
            for j in range(n_cand):
                if hi_guess:
                    key_guess = (key_cand[j] << 8) | (key_low & 0xff)
                else:
                    key_guess = key_cand[j]
                t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
                t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
                raw_x = sp.convert_to_binary([t0l, t0r, t1l, t1r])
                x = extract_sensitive_bits(raw_x, bits)
                z = net.predict(x, batch_size=10000)
                z = np.squeeze(z)
                T_cand[j] = np.sum(z > c3)
                cnt[key_cand[j]] = T_cand[j]
        if i == n_iter - 1:
            break
        # generate key guess candidates for next iteration
        key_cand = gen_keys_candidate(key_cand, T_cand, key_guess_length, u_d, s_d)
    survive_key = [i for i in range(2**key_guess_length) if cnt[i] > th]
    return survive_key, cnt


# make a plaintext structure from a random plaintext pair
# diff: difference of the plaintext pair
# neutral_bits is used to form the plaintext structure
def make_homogeneous_set(diff=(0x211, 0xa04), neutral_bits=neutral_bits_1):
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


# use ND7s to perform valid plaintext structure identification as well as sk11[0~7] recovery attack
# M, th_M and ts is used for identifying valid plaintext structure
# n and th_n is used for recovery attack
# n_cand, search_iter, u_d and s_d is used for bayesian key search
# return (survive_key_list, is_valid_plaintext_structure):
#   survive_key_list is the surviving keys in stage1 attack
#   is_valid_plaintxt_structure implies whether the structure passed valid plaintext structure test
def attack_with_ND7s(c, M, th_M, ts, n, th_n, c3, n_cand, search_iter, net, u_d, s_d, bits=selected_bits):
    c0l, c0r, c1l, c1r = c[0], c[1], c[2], c[3]
    key_guess_length = 8
    cnt_M = np.zeros(2**key_guess_length, dtype=np.uint32)
    cnt_n = np.zeros(2**key_guess_length, dtype=np.uint32)
    key_cand = np.random.choice(2**key_guess_length, n_cand, False)
    T_cand = np.zeros(n_cand, dtype=np.uint32)

    # begin bayesian search
    for i in range(search_iter):
        for j in range(n_cand):
            key_guess = key_cand[j]
            t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
            t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
            raw_x = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            x = extract_sensitive_bits(raw_x, bits)
            z = net.predict(x, batch_size=10000)
            z = np.squeeze(z)
            T_cand[j] = np.sum(z > c3)
            cnt_M[key_guess] = T_cand[j]
            cnt_n[key_guess] = np.sum(z[:n] > c3)
        key_cand = gen_keys_candidate(key_cand, T_cand, key_guess_length, u_d, s_d)
    valid_index = np.sum(cnt_M > th_M)
    if valid_index >= ts:
        # the plaintext structure passes valid test
        survive_key = [i for i in range(2**key_guess_length) if cnt_n[i] > th_n]
        return survive_key, True
    else:
        # the plaintext structure does not pass valid test and we don't need to return surviving keys
        return [], False


# use ND7s to perform sk11[0~7] recovery attack and don't perform valid plaintext structure identification
# n and th_n is used in recovery attack
# n_cand, search_iter, u_d and s_d is used for bayesian key search
# return surviving sk11[0~7] list
def attack_with_ND7s_without_homogeneous_set(c, n, th_n, c3, n_cand, search_iter, net, u_d, s_d, bits=selected_bits):
    key_guess_length = 8
    survive_key = bayesian_search(c, net, u_d, s_d, c3, th_n, n_cand, search_iter, bits, False, None, key_guess_length)[0]
    return survive_key


# use ND7t to perform sk11[8~15] recovery attack
# n and th_n is used in recovery attack
# n_cand, search_iter, u_d and s_d is used for bayesian key search
# survive_key is the surviving sk11[0~7] list
# return surviving sk11 list
def attack_with_ND7t(c, n, th_n, c3, n_cand, search_iter, net, u_d, s_d, survive_key):
    key_guess_length = 8
    survive_key_2 = []
    for key_low in survive_key:
        tmp = bayesian_search(c, net, u_d, s_d, c3, th_n, n_cand, search_iter, full_bits, True, key_low, key_guess_length)[0]
        for key_hi in tmp:
            survive_key_2.append((key_hi << 8) | key_low)
    return survive_key_2


# use ND6s to perform sk10[0~7] recovery attack
# n and th_n is used in recovery attack
# n_cand, search_iter, u_d and s_d is used for bayesian key search
# survive_key is the surviving sk11 list
# return surviving (sk11, sk10[0~7]) list
def attack_with_ND6s(c, n, th_n, c3, n_cand, search_iter, net, u_d, s_d, survive_key, bits=selected_bits):
    survive_key_2 = []
    key_guess_length = 8
    for sk_11 in survive_key:
        t0l, t0r = sp.dec_one_round((c[0], c[1]), sk_11)
        t1l, t1r = sp.dec_one_round((c[2], c[3]), sk_11)
        tmp = bayesian_search((t0l, t0r, t1l, t1r), net, u_d, s_d, c3, th_n, n_cand, search_iter, bits, False, None, key_guess_length)[0]
        for key_low in tmp:
            survive_key_2.append((sk_11, key_low))

        # a speed-up trick based on the analysis of p_2
        # if len(tmp) > 4:
        #     break
    return survive_key_2


# use ND6t to perform sk10[8~15] recovery attack
# n and th_n is used in recovery attack
# n_cand, search_iter, u_d and s_d is used for bayesian key search
# survive_key is the surviving (sk11, sk10[0~7]) list
# return surviving (sk11, sk10) list
def attack_with_ND6t(c, n, th_n, c3, n_cand, search_iter, net, u_d, s_d, survive_key):
    survive_key_2 = []
    key_guess_length = 8
    for sk_11, key_low in survive_key:
        t0l, t0r = sp.dec_one_round((c[0], c[1]), sk_11)
        t1l, t1r = sp.dec_one_round((c[2], c[3]), sk_11)
        tmp = bayesian_search((t0l, t0r, t1l, t1r), net, u_d, s_d, c3, th_n, n_cand, search_iter, full_bits, True, key_low, key_guess_length)[0]
        for key_hi in tmp:
            survive_key_2.append((sk_11, (key_hi << 8) | key_low))
    return survive_key_2


def output_surviving_key_difference(sur_key, sk11, sk10):
    if len(sur_key) == 0:
        return
    print('Difference between surviving (kg_11, kg_10) and (sk_11, sk_10) are ')
    for i in range(len(sur_key)):
        print((hex(sur_key[i][0] ^ sk11), hex(sur_key[i][1] ^ sk10)))


# attack: sk11[0~7] -> sk11 -> (sk11, sk[0~7]) -> (sk11, sk10)
# p0 = 2**(-2)
# don't use plaintext structures / do not use neutral bits
def key_recovery_attack_2_rounds_with_bayesian_search(t=10, nr=11, n=(5697905, 1282921, 1274059, 168923), th_n=(1636227, 280229, 335246, 20702), c3=0.55, diff=(0x211, 0xa04),
                                                    n_cand=32, search_iter=(3, 4, 3, 3), net_path=None, p0=2**(-6), p2_d1=None, p3=(0.2865, 0.2162, 0.2604, 0.1162), 
                                                    bits=selected_bits):
    net = []
    for i in range(4):
        net.append(load_model(net_path[i]))
    u_d = [0] * 4; s_d = [0] * 4
    for i in range(4):
        u_d[i], s_d[i] = get_u_d_s_d(n[i], p0, p2_d1[i], p3[i])

    #
    survive_num = np.zeros((t, 4), dtype=np.uint32)
    is_success = np.zeros(t, dtype=np.uint8)
    attack_time = np.zeros(t)
    
    for attack_index in range(t):
        start = time.time()
        print('attack: {}'.format(attack_index))
        key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(key, nr)
        sk_10, sk_11 = ks[nr-2][0], ks[nr-1][0]
        p0l = np.frombuffer(urandom(2 * n[0]), dtype=np.uint16)
        p0r = np.frombuffer(urandom(2 * n[0]), dtype=np.uint16)
        p1l = p0l ^ diff[0]; p1r = p0r ^ diff[1]
        p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
        p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)
        stage = 0
        # recover sk11[0~7]
        survive_key_1 = attack_with_ND7s_without_homogeneous_set((c0l, c0r, c1l, c1r), n[stage], th_n[stage], c3, n_cand, search_iter[stage], net[stage], u_d[stage], s_d[stage], bits)
        print('survive_key_1 number: {}'.format(len(survive_key_1)))
        #
        survive_num[attack_index][0] = len(survive_key_1)
        stage += 1
        c0l = c0l[:n[stage]]; c0r = c0r[:n[stage]]; c1l = c1l[:n[stage]]; c1r = c1r[:n[stage]];
        # recover sk11
        survive_key_2 = attack_with_ND7t((c0l, c0r, c1l, c1r), n[stage], th_n[stage], c3, n_cand, search_iter[stage], net[stage], u_d[stage], s_d[stage], survive_key_1)
        print('survive_key_2 number: {}'.format(len(survive_key_2)))
        #
        survive_num[attack_index][1] = len(survive_key_2)
        stage += 1
        c0l = c0l[:n[stage]]; c0r = c0r[:n[stage]]; c1l = c1l[:n[stage]]; c1r = c1r[:n[stage]];
        # recover (sk11, sk10[0~7])
        survive_key_3 = attack_with_ND6s((c0l, c0r, c1l, c1r), n[stage], th_n[stage], c3, n_cand, search_iter[stage], net[stage], u_d[stage], s_d[stage], survive_key_2, bits)
        print('survive_key_3 number: {}'.format(len(survive_key_3)))
        #
        survive_num[attack_index][2] = len(survive_key_3)
        stage += 1
        c0l = c0l[:n[stage]]; c0r = c0r[:n[stage]]; c1l = c1l[:n[stage]]; c1r = c1r[:n[stage]];
        # recover (sk11, sk10)
        survive_key_4 = attack_with_ND6t((c0l, c0r, c1l, c1r), n[stage], th_n[stage], c3, n_cand, search_iter[stage], net[stage], u_d[stage], s_d[stage], survive_key_3)
        print('survive_key_4 number: {}'.format(len(survive_key_4)))
        survive_num[attack_index][3] = len(survive_key_4)
        end = time.time()
        output_surviving_key_difference(sur_key=survive_key_4, sk11=sk_11, sk10=sk_10)
        attack_time[attack_index] = end - start
        print('time cost: {}s'.format(attack_time[attack_index]))
        if (sk_11, sk_10) in survive_key_4:
            print('true key survived.')
            is_success[attack_index] = 1
        else:
            print('true key didn\'t survived')
            is_success[attack_index] = 0

    average_surviving_num = np.mean(survive_num, axis=0)
    print('attack success time: {}'.format(np.sum(is_success)))
    print('average surviving key in stage1: {}'.format(average_surviving_num[0]))
    print('average surviving key in stage2: {}'.format(average_surviving_num[1]))
    print('average surviving key in stage3: {}'.format(average_surviving_num[2]))
    print('average surviving key in stage4: {}'.format(average_surviving_num[3]))
    print('average attack time: {}s'.format(np.mean(attack_time)))

    folder = './key_recovery_record/bayesian/2_7_6_{}_{}_{}_{}'.format(n[0], n[1], n[2], n[3])
    if not path.exists(folder):
        mkdir(folder)
    folder += '/'
    np.save(folder + 'is_success.npy', is_success)
    np.save(folder + 'survive_num.npy', survive_num)
    np.save(folder + 'attack_time.npy', attack_time)


# attack: sk11[0~7] -> sk11 -> (sk11, sk10[0~7]) -> (sk11, sk10)
# use plaintext structures
def key_recovery_attack_2_rounds_with_bayesian_search_using_homogeneous_sets(t=10, nr=11, M=38565, th_M=11285, ts=8, n=(22616, 5305, 5257, 829), th_n=(6703, 1333, 1598, 180), c3=0.55, diff=(0x211, 0xa04),
                                                                            n_cand=32, search_iter=(3, 4, 3, 3), net_path=None, p0=2**(-2), p2_d1=None, p3=(0.2865, 0.2162, 0.2604, 0.1162), 
                                                                            bits=selected_bits, neutral_bits=neutral_bits_1):
    net = []
    for i in range(4):
        net.append(load_model(net_path[i]))
    u_d = [0] * 4; s_d = [0] * 4
    u_d[0], s_d[0] = get_u_d_s_d(M, p0, p2_d1[0], p3[0])
    for i in range(1, 4):
        u_d[i], s_d[i] = get_u_d_s_d(n[i], p0, p2_d1[i], p3[i])

    true_success, true_fail, false_success, false_fail = 0, 0, 0, 0
    survive_num = np.zeros((t, 4), dtype=np.uint32)
    consumption_num = np.zeros(t, dtype=np.uint32)
    is_valid = np.zeros(t, dtype=np.uint8)
    is_success = np.zeros(t, dtype=np.uint8)
    attack_time = np.zeros(t)
    for attack_index in range(t):
        start = time.time()
        print('attack: {}'.format(attack_index))
        key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(key, nr)
        sk_10, sk_11 = ks[nr-2][0], ks[nr-1][0]
        valid_homogeneous_set = False
        homogeneous_set_consumption = 0
        # identify valid plaintext structure and recover sk11[0~7]
        while not valid_homogeneous_set:
            if homogeneous_set_consumption >= 24:
                print('attack failed because no valid plaintext structures can be found')
                break
            homogeneous_set_consumption += 1
            p0l, p0r, p1l, p1r = make_homogeneous_set(diff=diff, neutral_bits=neutral_bits)
            p0l, p0r = sp.dec_one_round((p0l[0:M], p0r[0:M]), 0)
            p1l, p1r = sp.dec_one_round((p1l[0:M], p1r[0:M]), 0)
            c0l, c0r = sp.encrypt((p0l, p0r), ks)
            c1l, c1r = sp.encrypt((p1l, p1r), ks)
            survive_key_1, valid_homogeneous_set = attack_with_ND7s((c0l, c0r, c1l, c1r), M, th_M, ts, n[0], th_n[0], c3, n_cand, search_iter[0], net[0], u_d[0], s_d[0], bits)
        
        # find a valid plaintext structure
        t0l, t0r = sp.encrypt((p0l, p0r), ks[:2])
        t1l, t1r = sp.encrypt((p1l, p1r), ks[:2])
        good_num = ((t0l ^ t1l) == 0x2800) * ((t0r ^ t1r) == 0x10)
        good_num = np.sum(good_num)
        if good_num == M:
            print('valid plaintext structure')
            is_valid[attack_index] = 1
        else:
            print('invalid plaintext structure')
            is_valid[attack_index] = 0

        print('plaintext structures consumption: {}'.format(homogeneous_set_consumption))
        consumption_num[attack_index] = homogeneous_set_consumption
        print('survive_key_1 number: {}'.format(len(survive_key_1)))
        survive_num[attack_index][0] = len(survive_key_1)
        stage = 1
        c0l = c0l[:n[stage]]; c0r = c0r[:n[stage]]; c1l = c1l[:n[stage]]; c1r = c1r[:n[stage]];
        # recover sk11
        survive_key_2 = attack_with_ND7t((c0l, c0r, c1l, c1r), n[stage], th_n[stage], c3, n_cand, search_iter[stage], net[stage], u_d[stage], s_d[stage], survive_key_1)
        print('survive_key_2 number: {}'.format(len(survive_key_2)))
        survive_num[attack_index][1] = len(survive_key_2)
        stage += 1
        c0l = c0l[:n[stage]]; c0r = c0r[:n[stage]]; c1l = c1l[:n[stage]]; c1r = c1r[:n[stage]];
        # recover (sk11, sk10[0~7])
        survive_key_3 = attack_with_ND6s((c0l, c0r, c1l, c1r), n[stage], th_n[stage], c3, n_cand, search_iter[stage], net[stage], u_d[stage], s_d[stage], survive_key_2, bits)
        print('survive_key_3 number: {}'.format(len(survive_key_3)))
        survive_num[attack_index][2] = len(survive_key_3)
        stage += 1
        c0l = c0l[:n[stage]]; c0r = c0r[:n[stage]]; c1l = c1l[:n[stage]]; c1r = c1r[:n[stage]];
        # recover (sk11, sk10)
        survive_key_4 = attack_with_ND6t((c0l, c0r, c1l, c1r), n[stage], th_n[stage], c3, n_cand, search_iter[stage], net[stage], u_d[stage], s_d[stage], survive_key_3)
        print('survive_key_4 number: {}'.format(len(survive_key_4)))
        survive_num[attack_index][3] = len(survive_key_4)
        end = time.time()
        output_surviving_key_difference(sur_key=survive_key_4, sk11=sk_11, sk10=sk_10)
        attack_time[attack_index] = end - start
        print('time cost: {}s'.format(attack_time[attack_index]))
        if (sk_11, sk_10) in survive_key_4:
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
    average_surviving_num = np.mean(survive_num, axis=0)
    print('attack success time: {}'.format(np.sum(is_success)))
    print('get valid plaintext structure time: {}'.format(np.sum(is_valid)))
    print('success time with valid plaintext structures: {}'.format(true_success))
    print('failure time with valid plaintext structures: {}'.format(true_fail))
    print('success time with invalid plaintext structures: {}'.format(false_success))
    print('failure time with invalid plaintext structures: {}'.format(false_fail))
    print('average surviving key in stage1: {}'.format(average_surviving_num[0]))
    print('average surviving key in stage2: {}'.format(average_surviving_num[1]))
    print('average surviving key in stage3: {}'.format(average_surviving_num[2]))
    print('average surviving key in stage4: {}'.format(average_surviving_num[3]))
    print('average attack time: {}s'.format(np.mean(attack_time)))
    print('average plaintext structure consumption: {}'.format(np.mean(consumption_num)))

    folder = './key_recovery_record/bayesian/3_7_6_{}_{}_{}_{}'.format(n[0], n[1], n[2], n[3])
    if not path.exists(folder):
        mkdir(folder)
    folder += '/'
    np.save(folder + 'is_valid.npy', is_valid)
    np.save(folder + 'is_success.npy', is_success)
    np.save(folder + 'survive_num.npy', survive_num)
    np.save(folder + 'consumption_num.npy', consumption_num)
    np.save(folder + 'attack_time.npy', attack_time)
 

if __name__ == '__main__':
    # attack on 11-round Speck32/64 using neutral bits
    dis_nr = 7
    nr = 11
    net1 = './saved_model/student/0x0040-0x0/hard_label/student_{}_distinguisher.h5'.format(dis_nr)
    net2 = './saved_model/teacher/0x0040-0x0/{}_distinguisher.h5'.format(dis_nr)
    net3 = './saved_model/student/0x0040-0x0/hard_label/student_{}_distinguisher.h5'.format(dis_nr - 1)
    net4 = './saved_model/teacher/0x0040-0x0/{}_distinguisher.h5'.format(dis_nr - 1)
    M = 37938
    th_M = 11103
    ts = 8
    n = (22586, 5271, 5228, 829)
    th_n = (6690, 1325, 1589, 180)
    c3 = 0.55
    diff = (0x211, 0xa04)
    n_cand = 32
    search_iter = (3, 4, 3, 3)
    p0 = 2**(-2)
    p2_d1_1 = np.load('./p2_estimation_res/student/0x0040-0x0/{0}/{0}_p2_d1.npy'.format(dis_nr))
    p2_d1_2 = np.load('./p2_estimation_res/teacher/0x0040-0x0/{0}/{0}_0.55_p2_d1.npy'.format(dis_nr))
    p2_d1_3 = np.load('./p2_estimation_res/student/0x0040-0x0/{0}/{0}_p2_d1.npy'.format(dis_nr - 1))
    p2_d1_4 = np.load('./p2_estimation_res/teacher/0x0040-0x0/{0}/{0}_0.55_p2_d1.npy'.format(dis_nr - 1))
    p3 = (0.2862612, 0.2162363, 0.2603019, 0.1161114)
    key_recovery_attack_2_rounds_with_bayesian_search_using_homogeneous_sets(t=500, nr=nr, M=M, th_M=th_M, ts=8, n=n, th_n=th_n, c3=c3, diff=diff, n_cand=n_cand, search_iter=search_iter,
                                                                            net_path=(net1, net2, net3, net4), p0=p0, p2_d1=(p2_d1_1, p2_d1_2, p2_d1_3, p2_d1_4), p3=p3,
                                                                            neutral_bits=neutral_bits_1)

    # verify the negative influence of neutral bits quickly
    # The Bayesian search strategy is used, which has a minor negative influence on the success rate.
    # If you wanna reproduce the result presented in our paper, do not use Bayesian search strategy,
    # and run verify_negative_influence_of_NBs.
    # dis_nr = 7
    # nr = 10
    # net1 = './saved_model/student/0x0040-0x0/hard_label/student_{}_distinguisher.h5'.format(dis_nr)
    # net2 = './saved_model/teacher/0x0040-0x0/{}_distinguisher.h5'.format(dis_nr)
    # net3 = './saved_model/student/0x0040-0x0/hard_label/student_{}_distinguisher.h5'.format(dis_nr - 1)
    # net4 = './saved_model/teacher/0x0040-0x0/{}_distinguisher.h5'.format(dis_nr - 1)
    # n = (22586, 5271, 5228, 829)
    # th_n = (6690, 1325, 1589, 180)
    # c3 = 0.55
    # diff = (0x2800, 0x10)
    # n_cand = 32
    # search_iter = (3, 4, 3, 3)
    # p0 = 2**(-2)
    # p2_d1_1 = np.load('./p2_estimation_res/student/0x0040-0x0/{0}/{0}_p2_d1.npy'.format(dis_nr))
    # p2_d1_2 = np.load('./p2_estimation_res/teacher/0x0040-0x0/{0}/{0}_0.55_p2_d1.npy'.format(dis_nr))
    # p2_d1_3 = np.load('./p2_estimation_res/student/0x0040-0x0/{0}/{0}_p2_d1.npy'.format(dis_nr - 1))
    # p2_d1_4 = np.load('./p2_estimation_res/teacher/0x0040-0x0/{0}/{0}_0.55_p2_d1.npy'.format(dis_nr - 1))
    # p3 = (0.2862612, 0.2162363, 0.2603019, 0.1161114)
    # key_recovery_attack_2_rounds_with_bayesian_search(t=500, nr=nr, n=n, th_n=th_n, c3=c3, diff=diff, n_cand=n_cand, search_iter=search_iter,
    #                                                 net_path=(net1, net2, net3, net4), p0=p0, p2_d1=(p2_d1_1, p2_d1_2, p2_d1_3, p2_d1_4), p3=p3)
