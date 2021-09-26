import speck as sp
import numpy as np

from pickle import dump
from keras.models import Model, load_model
from os import urandom
import random
import gc


gc.disable()

nr = 7
net_path = './saved_model/student/0x80-0x0/soft_label/21_8_student_7_distinguisher.h5'
# net_path = './saved_model/teacher/0x80-0x0/7_distinguisher.h5'
word_size = sp.WORD_SIZE()
selected_bits_1 = [21,20,19,18,17,16,15,14,13,12,11,10,9,8]
selected_bits_2 = [word_size - 1 - v for v in range(word_size)]
mask_val = (1 << word_size) - 1


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + word_size * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]

    return new_x


# diff = ()
def make_target_diff_samples(n=64*10, nr=8, diff_type=1, diff=(0x80, 0x0)):
    p0l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    p0r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    if diff_type == 1:
        p1l = p0l ^ diff[0]
        p1r = p0r ^ diff[1]
    else:
        p1l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        p1r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    # n different master keys for n plaintext pairs
    key = np.frombuffer(urandom(16*n), dtype=np.uint64).reshape(2, -1)
    ks = sp.expand_key(key, nr)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r


def show_distinguisher_acc(n=10**7, nr=7, net_path=net_path, diff=(0x80, 0x0), bits=[14, 13, 12, 11, 10, 9, 8]):
    net = load_model(net_path)
    c0l, c0r, c1l, c1r = make_target_diff_samples(n=n, nr=nr, diff_type=1, diff=diff)
    raw_x = sp.convert_to_binary([c0l, c0r, c1l, c1r])
    x = extract_sensitive_bits(raw_x, bits=bits)
    # x = raw_x
    y = net.predict(x, batch_size=10000)
    y = np.squeeze(y)
    tp = np.sum(y > 0.5) / n
    fn = 1 - tp

    c0l, c0r, c1l, c1r = make_target_diff_samples(n=n, nr=nr, diff_type=0)
    raw_x = sp.convert_to_binary([c0l, c0r, c1l, c1r])
    x = extract_sensitive_bits(raw_x, bits=bits)
    y = net.predict(x, batch_size=10000)
    y = np.squeeze(y)
    tn = np.sum(y <= 0.5) / n
    fp = 1 - tn

    print('acc of cur distinguisher is ')
    print('tp_to_tp: ', tp, ' tp_to_fn: ', fn, ' tn_to_tn: ', tn, ' tn_to_fp: ', fp)


def cal_p1_p3(n=10**7, nr=7, c3=0.5, net_path=net_path, diff=(0x80, 0), bits=[14, 13, 12, 11, 10, 9, 8]):
    net = load_model(net_path)

    keys = np.frombuffer(urandom(16*n), dtype=np.uint64).reshape(2, -1)
    ks = sp.expand_key(keys, nr+1)
    p0l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    p0r = np.frombuffer(urandom(8 * n), dtype=np.uint64)

    for i in range(1, 4):
        if i == 1:
            p1l = p0l ^ diff[0]
            p1r = p0r ^ diff[1]
        elif i == 2:
            continue
        else:
            p1l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
            p1r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)

        if i == 1:
            dk = ks[nr]
        else:
            dk = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        d0l, d0r = sp.dec_one_round((c0l, c0r), dk)
        d1l, d1r = sp.dec_one_round((c1l, c1r), dk)
        raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
        x = extract_sensitive_bits(raw_x, bits=bits)
        Z = net.predict(x, batch_size=10000)

        acc = np.sum(Z > c3) / n
        if i == 1:
            p1 = acc
        elif i == 3:
            p3 = acc

    print("p1: ", p1, ' p3: ', p3)
    return p1, p3


# need to be tested
def gen_fk(arr):
    fk = 0
    for v in arr:
        fk = fk + (1 << v)

    return fk


def cal_p2_d1_for_speck(n=10**7, nr=7, c3=0.55, net_path=net_path, diff=(0x80, 0), bits=selected_bits_2):
    net = load_model(net_path)
    d1 = len(bits)
    p2_d1 = np.zeros(d1+1)

    sample_range = [i for i in range(d1)]
    # d1 = 1, ... , bit_num - 1
    for i in range(d1+1):
        print('cur i is ', i)
        keys = np.frombuffer(urandom(16 * n), dtype=np.uint64).reshape(2, -1)
        ks = sp.expand_key(keys, nr + 1)  # ks[nr-1] = 17123;

        pt0a = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        pt1a = np.frombuffer(urandom(8 * n), dtype=np.uint64)
        pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1]
        ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
        ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)

        fks = np.array([gen_fk(random.sample(sample_range, i)) for j in range(n)], dtype=np.uint64)
        fks = fks ^ ks[nr]
        c0a, c1a = sp.dec_one_round((ct0a, ct1a), fks)
        c0b, c1b = sp.dec_one_round((ct0b, ct1b), fks)
        raw_X = sp.convert_to_binary([c0a, c1a, c0b, c1b])
        X = extract_sensitive_bits(raw_X, bits=bits)

        Z = net.predict(X, batch_size=10000)
        Z = np.squeeze(Z)
        p2_d1[i] = np.sum(Z > c3) / n  # save the probability
        print('cur p2_d1 is ', p2_d1[i])

    np.save('./p2_estimation_res/student/0x80-0x0/' + str(nr) + '/' + str(nr) + '_p2_d1.npy', p2_d1)
    # print(p2_d1)


def cal_p2_d1_d2_for_speck(n=10**7, nr=5, c3=0.55, net_path=net_path, diff=(0x80, 0), bits_1=selected_bits_1, bits_2=selected_bits_2):
    net = load_model(net_path)
    # d1 is the last_to_second subkey length, d2 is the last round subkey length
    d1 = len(bits_1)
    d2 = len(bits_2)
    p2_d1_d2 = np.zeros((d1+1, d2+1))

    sample_range_1 = [i for i in range(d1+1)]
    sample_range_2 = [i for i in range(d2+1)]
    for i in range(d1+1):
        for j in range(d2+1):
            print('cur i, j are ', i, ' ', j)
            keys = np.frombuffer(urandom(16 * n), dtype=np.uint64).reshape(2, -1)
            ks = sp.expand_key(keys, nr + 2)  # ks[nr-1] = 17123;

            pt0a = np.frombuffer(urandom(8 * n), dtype=np.uint64)
            pt1a = np.frombuffer(urandom(8 * n), dtype=np.uint64)
            pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1]
            ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
            ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)

            fks_1 = np.array([gen_fk(random.sample(sample_range_1, i)) for k in range(n)], dtype=np.uint32)
            fks_2 = np.array([gen_fk(random.sample(sample_range_2, j)) for k in range(n)], dtype=np.uint32)
            # print('fks shape is ', np.shape(fks))
            # print('ks[nr] shape is ', np.shape(ks[nr]))
            fks_1 = fks_1 ^ ks[nr]
            fks_2 = fks_2 ^ ks[nr+1]
            # decrypt one round
            c0a, c1a = sp.dec_one_round((ct0a, ct1a), fks_2)
            c0b, c1b = sp.dec_one_round((ct0b, ct1b), fks_2)
            # decrypt one round
            c0a, c1a = sp.dec_one_round((c0a, c1a), fks_1)
            c0b, c1b = sp.dec_one_round((c0b, c1b), fks_1)
            raw_X = sp.convert_to_binary([c0a, c1a, c0b, c1b])
            X = extract_sensitive_bits(raw_X, bits=bits_1)

            Z = net.predict(X, batch_size=10000)
            Z = np.squeeze(Z)
            p2_d1_d2[i][j] = np.sum(Z > c3) / n  # save the probability
            print('cur p2_d1_d2 is ', p2_d1_d2[i][j])

    np.save('./p2_estimation_res/student/0x80-0x0/' + str(nr) + '/' + str(c3) + '_' + str(nr) + '_p2_d1_d2.npy', p2_d1_d2)
    # print(p2_d1)


#show_distinguisher_acc(n=10**7, nr=nr, net_path=net_path, diff=(0x80, 0x0), bits=selected_bits_1)
p1, p3 = cal_p1_p3(n=10**6, nr=nr, c3=0.5, net_path=net_path, diff=(0x80, 0x0), bits=selected_bits_1)
cal_p2_d1_for_speck(n=10**6, nr=nr, c3=0.5, net_path=net_path, diff=(0x80, 0x0), bits=selected_bits_1)
# cal_p2_d1_d2_for_speck(n=10**7, nr=nr, c3=0.5, net_path=net_path, diff=(0x80, 0x0), bits_1=selected_bits_1, bits_2=selected_bits_2)
# cal_p2_d1_d2_for_speck(n=10**7, nr=nr, c3=0.55, net_path=net_path, diff=(0x80, 0x0), bits_1=selected_bits_1, bits_2=selected_bits_2)

gc.enable()