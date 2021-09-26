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


# type = 1, return the complete sk for teacher distinguishers, or return sk & 0xff for student distinguishers
def make_target_diff_samples(n=2**12, nr=10, diff=(0x2800, 0x10), type=1):
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

    if type == 1:
        return c0l, c0r, c1l, c1r, ks[nr - 1][0]
    else:
        return c0l, c0r, c1l, c1r, ks[nr-1][0] & np.uint16(0xff)


def naive_key_recovery_attack(t=100, n=2**12, th=11070, nr=10, c3=0.55, type=1, net='./', diff=(0x2800, 0x10), bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    cur_net = load_model(net)

    key_len = len(bits)
    acc = 0
    cnt = np.zeros((t, 2**(key_len)), dtype=np.uint32)
    tk = np.zeros(t, dtype=np.uint16)
    for i in range(t):
        print('i is ', i)
        c0l, c0r, c1l, c1r, true_key = make_target_diff_samples(n=n, nr=nr, diff=diff, type=type)
        print('true key is ', hex(true_key))
        tk[i] = true_key

        bc = 0
        bk = 0      # -1不行
        for sk in range(2**(key_len)):
            key_guess = sk
            t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
            t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
            raw_X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            X = extract_sensitive_bits(raw_X, bits=bits)
            Z = cur_net.predict(X, batch_size=10000)
            Z = np.squeeze(Z)
            cnt[i][sk] = np.sum(Z > c3)

            # if key_guess == true_key:
                # print('true key cnt is ', np.sum(Z > c3))

            # if np.sum(Z > c3) >= bc:
                # bc = np.sum(Z > c3)
                # bk = key_guess
                # print('difference between cur key and true key is ', hex(true_key ^ np.uint16(key_guess)))
                # print('new best sk is ', bk, 'cur bc is ', bc)

        bk = np.argmax(cnt[i])
        print('difference between best key guess and true key is ', hex(bk ^ true_key))
        print('the number of surviving keys is ', np.sum(cnt[i, :] > th))
        print('the cnt of the true key is ', cnt[i][true_key])
        if bk == true_key:
            acc = acc + 1
    acc = acc / t
    print('total acc is ', acc)

    return acc, cnt, tk


def attack_with_8_round_distinguisher(t=100, n=2**12, th=11070, nr=10, c3=0.55, diff=(0x2800, 0x10)):
    json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
    json_model = json_file.read()
    cur_net = model_from_json(json_model)
    cur_net.load_weights('./saved_model/teacher/0x0040-0x0/net8_small.h5')
    cur_net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    acc = 0
    cnt = np.zeros((t, 2 ** 16), dtype=np.uint32)
    tk = np.zeros(t, dtype=np.uint64)
    for i in range(t):
        print('i is ', i)
        c0l, c0r, c1l, c1r, true_key = make_target_diff_samples(n=n, nr=nr, diff=diff, type=1)
        print('true key is ', hex(true_key))
        tk[i] = true_key

        for sk in range(2 ** 16):
            key_guess = sk
            t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
            t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
            X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            Z = cur_net.predict(X, batch_size=n)
            Z = np.squeeze(Z)
            cnt[i][sk] = np.sum(Z > c3)

            # if key_guess == true_key:
                # print('true key cnt is ', np.sum(Z > c3))

            # if np.sum(Z > c3) >= bc:
                # bc = np.sum(Z > c3)
                # bk = key_guess
                # print('difference between cur key and true key is ', hex(true_key ^ np.uint64(key_guess)))
                # print('new best sk is ', bk, 'cur bc is ', bc)

        bk = np.argmax(cnt[i])
        if bk == true_key:
            acc = acc + 1
        print('bk is ')
        print('difference between best key guess and true key is ', hex(bk ^ true_key))
        print('the number of surviving keys is ', np.sum(cnt[i, :] > th))
    acc = acc / t
    print('total acc is ', acc)

    return acc, cnt, tk


# attack 1, attack 9 round Speck32/64
# selected_bits = [15 - i for i in range(16)]
# net_path = './saved_model/teacher/0x0040-0x0/5_distinguisher.h5'
# acc, cnt, tk = naive_key_recovery_attack(t=100, n=15905, th=758, nr=9, c3=0.55,
#                                           type=1,net=net_path, diff=(0x211, 0xa04), bits=selected_bits)
# np.save('./key_recovery_record/3_5_0.55_15905_758_cnt_record.npy', cnt)
# np.save('./key_recovery_record/3_5_0.55_15905_758_true_keys.npy', tk)

# attack 2
selected_bits = [15 - i for i in range(16)]
net_path = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'
acc, cnt, tk = naive_key_recovery_attack(t=100, n=475, th=101, nr=9, c3=0.55,
                                          type=1, net=net_path, diff=(0x2800, 0x10), bits=selected_bits)
np.save('./key_recovery_record/2_6_0.55_475_101_cnt_record.npy', cnt)
np.save('./key_recovery_record/2_6_0.55_475_101_true_keys.npy', tk)

# attack 3
# selected_bits = [15 - i for i in range(16)]
# net_path = './saved_model/teacher/0x0040-0x0/7_distinguisher.h5'
# acc, cnt, tk = naive_key_recovery_attack(t=100, n=5272, th=1325, nr=10, c3=0.55,
#                                           type=1,net=net_path, diff=(0x2800, 0x10), bits=selected_bits)
# np.save('./key_recovery_record/2_7_0.55_5272_1325_cnt_record.npy', cnt)
# np.save('./key_recovery_record/2_7_0.55_5272_1325_true_keys.npy', tk)

# attack 4
# acc, cnt, tk = attack_with_8_round_distinguisher(t=100, n=25680, th=13064, nr=10, c3=0.5, diff=(0x0040, 0x0))
# np.save('./key_recovery_record/1_8_0.5_25680_13064_cnt_record.npy', cnt)
# np.save('./key_recovery_record/1_8_0.5_25680_13064_true_keys.npy', tk)

