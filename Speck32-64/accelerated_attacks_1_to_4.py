import speck as sp
import numpy as np
import heapq

from pickle import dump
from keras.models import Model, load_model, model_from_json
from os import urandom


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


# type = 1, return the complete sk, or return sk & 0xff
def make_target_diff_samples(n=2**12, nr=10, diff=(0x2800, 0x10)):
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


def key_recovery_with_reduced_keySpace(t=100, n1=0, th1=0, n2=0, th2=0, nr=10, c3=0.55, diff=(0x2800, 0x10), bits=[14], net_t='./', net_s='./'):
    student = load_model(net_s)
    teacher = load_model(net_t)
    num = 0

    bits_len = len(bits)
    sub_space_1 = 2**(bits_len)
    sub_space_2 = 2**(word_size - bits_len)
    print('sub_space_1 is ', sub_space_1, ' sub_space_2 is ', sub_space_2)
    cnt = np.zeros((t, 2**word_size), dtype=np.uint32)
    tk = np.zeros(t, dtype=np.uint16)
    for i in range(t):
        print('i is ', i)
        c0l, c0r, c1l, c1r, true_key = make_target_diff_samples(n=n1, nr=nr, diff=diff)
        print('true key is ', hex(true_key))
        tk[i] = true_key

        c0l_2, c0r_2 = c0l[0:n2], c0r[0:n2]
        c1l_2, c1r_2 = c1l[0:n2], c1r[0:n2]
        # print('c0l_2 shape is ', np.shape(c0l_2))

        for sk_1 in range(sub_space_1):
            # print('cur sk_1 is ', sk_1)
            t0l, t0r = sp.dec_one_round((c0l, c0r), sk_1)
            t1l, t1r = sp.dec_one_round((c1l, c1r), sk_1)
            raw_X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            X = extract_sensitive_bits(raw_X, bits=bits)
            Z = student.predict(X, batch_size=n1)
            Z = np.squeeze(Z)

            tp_1 = np.sum(Z > c3)
            if tp_1 < th1:
                continue

            # print('tp_1 is ', tp_1)
            cnt[i][np.uint16(sk_1)] = tp_1          # record the surviving number of the first stage.
            for sk_2 in range(sub_space_2):
                sk = (sk_2 << bits_len) | sk_1
                t0l_2, t0r_2 = sp.dec_one_round((c0l_2, c0r_2), sk)
                t1l_2, t1r_2 = sp.dec_one_round((c1l_2, c1r_2), sk)

                X_2 = sp.convert_to_binary([t0l_2, t0r_2, t1l_2, t1r_2])
                Z_2 = teacher.predict(X_2, batch_size=n2)
                Z_2 = np.squeeze(Z_2)

                tp_2 = np.sum(Z_2 > c3)
                if tp_2 > th2:
                    # print('the difference between surviving kg and right key is ', hex(np.uint16(sk) ^ true_key))
                    # print('the cnt is ', tp_2)
                    cnt[i][np.uint16(sk)] = tp_2

        key_survive_num = np.sum(cnt[i, :] > th2)
        print('the number of surviving keys is ', key_survive_num)
        print('the cnt of the true key is ', cnt[i][true_key])

        num = num + key_survive_num
    print('the average number of surviving keys is ', num / t)

    return cnt, tk


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


def key_recovery_with_ND8T_Bayesian_search(t=100, n=25680, th=13064, nr=10, c3=0.5, diff=(0x2800, 0x10)):
    json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
    json_model = json_file.read()
    cur_net = model_from_json(json_model)
    cur_net.load_weights('./saved_model/teacher/0x0040-0x0/net8_small.h5')
    cur_net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    p2_d1 = np.array([0.5183955, 0.5056154, 0.4992611, 0.4957098, 0.4939058, 0.4927488, 0.4924816, 0.4918235, 0.4916707,
                      0.4914085, 0.4913381, 0.4913277, 0.4911237, 0.4913378, 0.4914284, 0.49097,   0.4914429])
    p3 = 0.4913596
    p0 = 0.5183955
    n_cand = 256
    n_iter = 5
    key_guess_length = 16
    u_d, s_d = get_u_d_s_d(n, p0, p2_d1, p3)

    acc = 0
    average_num = 0
    for i in range(t):
        print('attack: {}'.format(i))
        c0l, c0r, c1l, c1r, true_key = make_target_diff_samples(n=n, nr=nr, diff=diff)
        print('true key is ', hex(true_key))

        survive_key, cnt = bayesian_search((c0l, c0r, c1l, c1r), cur_net, u_d, s_d, c3, th, n_cand, n_iter,
                                           key_guess_length=key_guess_length)
        if true_key in survive_key:
            acc = acc + 1
            print('true key survive')
            print('the cnt of the true key is ', cnt[true_key])
        print('the number of surviving keys is ', len(survive_key))
        average_num = average_num + len(survive_key)

    acc = acc / t
    print('total acc is ', acc)
    print('the average number of surviving keys is ', average_num / t)


# attack 1 with reduced key space, 3 + 5
# net_path_s = './saved_model/student/0x0040-0x0/hard_label/student_5_distinguisher.h5'
# net_path_t = './saved_model/teacher/0x0040-0x0/5_distinguisher.h5'
# selected_bits = [14 -i for i in range(8)]      # 14 ~ 7
# cnt, tk = key_recovery_with_reduced_keySpace(t=1000, n1=97382, th1=13196, n2=15905, th2=758, nr=9, c3=0.55,
#                                           diff=(0x211, 0xa04), bits=selected_bits, net_t=net_path_t, net_s=net_path_s)


# attack 2 with reduced key space, 2 + 6
# net_path_s = './saved_model/student/0x0040-0x0/hard_label/student_6_distinguisher.h5'
# net_path_t = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'
# selected_bits = [14 - i for i in range(8)]      # 14 ~ 7
# cnt, tk = key_recovery_with_reduced_keySpace(t=100, n1=1995, th1=593, n2=475, th2=101, nr=9, c3=0.55,
#                                           diff=(0x2800, 0x10), bits=selected_bits, net_t=net_path_t, net_s=net_path_s)


# attack 3 with reduced key space, 2 + 7
# net_path_s = './saved_model/student/0x0040-0x0/hard_label/student_7_distinguisher.h5'
# net_path_t = './saved_model/teacher/0x0040-0x0/7_distinguisher.h5'
# selected_bits = [14 - i for i in range(8)]      # 14 ~ 7
# cnt, tk = key_recovery_with_reduced_keySpace(t=100, n1=22586, th1=6690, n2=5272, th2=1325, nr=10, c3=0.55,
#                                           diff=(0x2800, 0x10), bits=selected_bits, net_t=net_path_t, net_s=net_path_s)


# Bayesian key search has a minor negative influence of the success rate
# attack 4 with Bayesian key search, 1 + 8
key_recovery_with_ND8T_Bayesian_search(t=100, n=25680, th=13064, nr=10, c3=0.5, diff=(0x0040, 0x0))


