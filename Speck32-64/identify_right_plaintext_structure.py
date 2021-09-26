import speck as sp
import numpy as np

from keras.models import load_model
from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2

word_size = sp.WORD_SIZE()
MASK_VAL = 2 ** word_size - 1


#make a plaintext structure with neutral bits
def make_structure(pt0, pt1, diff=(0x211, 0xa04), neutral_bits=[20, 21, 22, 14, 15]):
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for i in neutral_bits:
        d = 1 << i
        d0 = d >> word_size
        d1 = d & MASK_VAL
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    p0, p1, p0b, p1b = np.squeeze(p0), np.squeeze(p1), np.squeeze(p0b), np.squeeze(p1b)
    return(p0,p1,p0b,p1b)


# generate a Speck key, return expanded key
def gen_key(nr):
    key = np.frombuffer(urandom(8), dtype=np.uint16)
    ks = sp.expand_key(key, nr)
    return(ks)


def gen_plain(n):
    pt0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    pt1 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    return(pt0, pt1)


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    id1 = [v + word_size * i for i in range(4) for v in id0]
    new_x = raw_x[:, id1]

    return new_x


def create_simulated_plaintext_structures(m=100, valid=1, diff=(0x2800, 0x10)):
    if valid == 1:
        pt0a, pt1a = gen_plain(m)
        pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1]
    else:
        pt0a, pt1a = gen_plain(m)
        pt0b, pt1b = gen_plain(m)

    return pt0a, pt1a, pt0b, pt1b


def create_plaintext_structures_with_neutral_bits(m=100, diff=(0x211, 0xa04), neutral_bits=[14,13,12,11]):
    assert 2**(len(neutral_bits)) >= m
    pt0, pt1 = gen_plain(1)
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)

    return pt0a[0:m], pt1a[0:m], pt0b[0:m], pt1b[0:m]


# nr covers the second differential and the neural distinguisher
def identify_right_simulated_structure(t=1000, valid=1, m=10, diff=(0x40, 0), nr=8, net='./',
                                        c=0.55, tc=10, tp=8, c_bits=[14]):
    key_space = 2**(len(c_bits))
    record = np.zeros((t, key_space), dtype=np.uint32)
    acc = 0
    distinguisher = load_model(net)
    for i in range(t):
        print('cur t is ', i)
        mt0a, mt1a, mt0b, mt1b = create_simulated_plaintext_structures(m=m, valid=valid, diff=diff)
        pt0a, pt1a = sp.dec_one_round((mt0a, mt1a), 0)
        pt0b, pt1b = sp.dec_one_round((mt0b, mt1b), 0)

        ks = gen_key(nr)
        ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
        ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)

        for kg in range(key_space):
            dt0a, dt1a = sp.dec_one_round((ct0a, ct1a), kg)
            dt0b, dt1b = sp.dec_one_round((ct0b, ct1b), kg)
            raw_x = sp.convert_to_binary([dt0a, dt1a, dt0b, dt1b])
            extracted_x = extract_sensitive_bits(raw_x, bits=c_bits)
            Z = distinguisher.predict(extracted_x, batch_size=m)
            record[i][kg] = np.sum(Z > c)

        num = np.sum(record[i] >= tc)
        if num >= tp:
            acc = acc + 1
            print('current Set is a valid structure. the number of surviving keys is ', num)
        else:
            print('current Set is an invalid structure. the number of surviving keys is ', num)

        # for valid structure
        if valid == 1:
            true_key = ks[nr-1] & (key_space - 1)
            dis = [hex(true_key ^ kg) for kg in range(key_space) if record[i][kg] >= tc]
            print('differences between returned kg and sk are ', dis)

    if valid == 1:
        print('acc is ', acc / t)
    else:
        print('acc is ', (t - acc) / t)


# plaintext structures created from neutral bits
def identify_right_structures(t=100, valid=1, m=100, diff=(0x211, 0xa04), m_diff=(0x2800, 0x10), nr=11, net='./',
                              c=0.55, tc=100, tp=8, n_bits=[14], c_bits=[14], special=0):
    key_space = 2 ** (len(c_bits))
    record = np.zeros((t, key_space), dtype=np.uint32)
    acc = 0
    distinguisher = load_model(net)
    index = 0
    while index < t:
        ks = gen_key(nr)
        mt0a, mt1a, mt0b, mt1b = create_plaintext_structures_with_neutral_bits(m=m, diff=diff, neutral_bits=n_bits)
        pt0a, pt1a = sp.dec_one_round((mt0a, mt1a), 0)
        pt0b, pt1b = sp.dec_one_round((mt0b, mt1b), 0)

        # check the true label
        ks_tp = ks[0:2]
        tp_mt0a, tp_mt1a = sp.encrypt((pt0a, pt1a), ks_tp)
        tp_mt0b, tp_mt1b = sp.encrypt((pt0b, pt1b), ks_tp)
        tp_diff0, tp_diff1 = tp_mt0a ^ tp_mt0b ^ m_diff[0], tp_mt1a ^ tp_mt1b ^ m_diff[1]
        tp_diff = tp_diff0 + tp_diff1
        if np.sum(tp_diff == 0) == m:
            # print('current set is a valid homogeneous set')
            label = 1
        else:
            # print('current set is an invalid homogeneous set')
            label = 0

        if label != valid:
            continue

        if special == 1:
            if valid == 1 or np.sum(tp_diff == 0) == 0:
                continue

        print('cur t is ', index)
        print('the number of plaintext pairs comformimg to the differential is ', np.sum(tp_diff == 0))

        ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
        ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)
        for kg in range(key_space):
            dt0a, dt1a = sp.dec_one_round((ct0a, ct1a), kg)
            dt0b, dt1b = sp.dec_one_round((ct0b, ct1b), kg)
            raw_x = sp.convert_to_binary([dt0a, dt1a, dt0b, dt1b])
            extracted_x = extract_sensitive_bits(raw_x, bits=c_bits)
            Z = distinguisher.predict(extracted_x, batch_size=m)
            record[index][kg] = np.sum(Z > c)

        num = np.sum(record[index] >= tc)
        if num >= tp and valid == 1:
            acc = acc + 1
            print('current valid structure is identified. the number of surviving keys is ', num)
        elif num < tp and valid == 0:
            acc = acc + 1
            print('current invalid structure is identified. the number of surviving keys is ', num)
        else:
            print('current identification result is wrong')
            print('the number of surviving keys is  ', num)

        index = index + 1

    print('acc is ', acc / t)
    print('the total number of real plaintext structures with right label is ', t)


net_path = './saved_model/student/0x0040-0x0/hard_label/student_7_distinguisher.h5'
selected_bits = [14 - i for i in range(8)]

# identify simulated right structures
identify_right_simulated_structure(t=100, valid=1, m=37938, diff=(0x2800, 0x10), nr=10, net=net_path,
                                    c=0.55, tc=11103, tp=8, c_bits=selected_bits)
# identify wrong structures
identify_right_simulated_structure(t=100, valid=0, m=37938, diff=(0x2800, 0x10), nr=10, net=net_path,
                                    c=0.55, tc=11103, tp=8, c_bits=selected_bits)

neutral_bits = [11,14,15,20,21,22,0,1,3,23,24,26,4,27,5,28]

# identify right structures created from neutral bits
identify_right_structures(t=100, valid=1, m=37938, diff=(0x211, 0xa04), m_diff=(0x2800, 0x10), nr=11, net=net_path,
                          c=0.55, tc=11103, tp=8, n_bits=neutral_bits, c_bits=selected_bits)

# identify wrong structures created from neutral bits
identify_right_structures(t=100, valid=0, m=37938, diff=(0x211, 0xa04), m_diff=(0x2800, 0x10), nr=11, net=net_path,
                          c=0.55, tc=11103, tp=8, n_bits=neutral_bits, c_bits=selected_bits, special=0)

# if special = 1, we test wrong structures which contain many plaintext pairs conforming to the prepended differential
# This shows why the right keys survive in a few experiments when a wrong plaintext structure passes the identification
identify_right_structures(t=100, valid=0, m=37938, diff=(0x211, 0xa04), m_diff=(0x2800, 0x10), nr=11, net=net_path,
                          c=0.55, tc=11103, tp=8, n_bits=neutral_bits, c_bits=selected_bits, special=1)









