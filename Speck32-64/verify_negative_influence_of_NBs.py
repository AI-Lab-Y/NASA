import speck as sp
import numpy as np

from keras.models import load_model
from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2
from random import sample

word_size = sp.WORD_SIZE()
MASK_VAL = 2 ** word_size - 1


# generate a Speck key, return expanded key
def gen_key(nr):
    key = np.frombuffer(urandom(8), dtype=np.uint16)
    ks = sp.expand_key(key, nr)
    return(ks)


def gen_plain(n):
    pt0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    pt1 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    return(pt0, pt1)


def make_chosen_plaintexts(n=100, diff=(0x211, 0xa04)):
    mt0a, mt1a = gen_plain(n)
    mt0b, mt1b = mt0a ^ diff[0], mt1a ^ diff[1]
    # add an extra round on the top
    pt0a, pt1a = sp.dec_one_round((mt0a, mt1a), 0)
    pt0b, pt1b = sp.dec_one_round((mt0b, mt1b), 0)

    return pt0a, pt1a, pt0b, pt1b


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    id1 = [v + word_size * i for i in range(4) for v in id0]
    new_x = raw_x[:, id1]

    return new_x


# recover sk_11[7~0]
def attack_with_ND7s(ct0a, ct1a, ct0b, ct1b, t1, c1=0.55, net1='./', key_bits=8, c_bits=[14]):
    key_space = 2**key_bits
    distinguisher = load_model(net1)
    surviving_kg = []

    for kg in range(key_space):
        dt0a, dt1a = sp.dec_one_round((ct0a, ct1a), kg)
        dt0b, dt1b = sp.dec_one_round((ct0b, ct1b), kg)
        raw_x = sp.convert_to_binary([dt0a, dt1a, dt0b, dt1b])
        X = extract_sensitive_bits(raw_x, bits=c_bits)
        Z = distinguisher.predict(X, batch_size=100000)
        if np.sum(Z > c1) >= t1:
            surviving_kg.append(kg)

    return surviving_kg


# recover sk_11, filter sk_11[7~0]
def attack_with_ND7t(ct0a, ct1a, ct0b, ct1b, sur_kg, t2, c2=0.55, net2='./', key_bits=8):
    key_space = 2**key_bits
    distinguisher = load_model(net2)
    surviving_kg = []

    for k1 in sur_kg:
        for k2 in range(key_space):
            kg = (k2 << 8) + k1
            dt0a, dt1a = sp.dec_one_round((ct0a, ct1a), kg)
            dt0b, dt1b = sp.dec_one_round((ct0b, ct1b), kg)
            X = sp.convert_to_binary([dt0a, dt1a, dt0b, dt1b])
            Z = distinguisher.predict(X, batch_size=100000)
            if np.sum(Z > c2) >= t2:
                surviving_kg.append(kg)

    return surviving_kg


# recover sk_10[7~0] and filter sk_11
def attack_with_ND6s(ct0a, ct1a, ct0b, ct1b, sur_kg, t3, c3=0.55, net3='./', key_bits=8, c_bits=[14]):
    key_space = 2**key_bits
    distinguisher = load_model(net3)
    surviving_kg = []

    for sk in sur_kg:
        st0a, st1a = sp.dec_one_round((ct0a, ct1a), sk)
        st0b, st1b = sp.dec_one_round((ct0b, ct1b), sk)
        for kg in range(key_space):
            dt0a, dt1a = sp.dec_one_round((st0a, st1a), kg)
            dt0b, dt1b = sp.dec_one_round((st0b, st1b), kg)
            raw_x = sp.convert_to_binary([dt0a, dt1a, dt0b, dt1b])
            X = extract_sensitive_bits(raw_x, bits=c_bits)
            Z = distinguisher.predict(X, batch_size=10000)
            if np.sum(Z > c3) >= t3:
                surviving_kg.append([sk, kg])

    return surviving_kg


# recover sk_10, filter sk_11
def attack_with_ND6t(ct0a, ct1a, ct0b, ct1b, sur_kg, t4, c4=0.55, net4='./', key_bits=8):
    key_space = 2**key_bits
    distinguisher = load_model(net4)
    surviving_kg = []

    for kg_set in sur_kg:
        sk = kg_set[0]
        st0a, st1a = sp.dec_one_round((ct0a, ct1a), sk)
        st0b, st1b = sp.dec_one_round((ct0b, ct1b), sk)
        k1 = kg_set[1]
        for k2 in range(key_space):
            kg = (k2 << 8) + k1
            dt0a, dt1a = sp.dec_one_round((st0a, st1a), kg)
            dt0b, dt1b = sp.dec_one_round((st0b, st1b), kg)
            X = sp.convert_to_binary([dt0a, dt1a, dt0b, dt1b])
            Z = distinguisher.predict(X, batch_size=10000)
            if np.sum(Z > c4) >= t4:
                surviving_kg.append([sk, kg])

    return surviving_kg


# settings for attacking 10-round Speck32/64
c1 = 0.55;    n1 = 22586;    t1 = 6690
c2 = 0.55;    n2 = 5271;     t2 = 1325
c3 = 0.55;    n3 = 5228;     t3 = 1589
c4 = 0.55;    n4 = 829;      t4 = 180

net1 = './saved_model/student/0x0040-0x0/hard_label/student_7_distinguisher.h5'
net2 = './saved_model/teacher/0x0040-0x0/7_distinguisher.h5'
net3 = './saved_model/student/0x0040-0x0/hard_label/student_6_distinguisher.h5'
net4 = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'
selected_bits = [14 - i for i in range(8)]


def recover_two_subkeys(t, nr, n1, diff=(0x211, 0xa04), id=1):
    attack_res = []
    m_res = np.zeros((t, 4), dtype=np.uint16)
    for i in range(t):
        print('cur t is ', i)
        ks = gen_key(nr)
        sk10, sk11 = ks[nr-2], ks[nr-1]
        pt0a, pt1a, pt0b, pt1b = make_chosen_plaintexts(n=n1, diff=diff)
        ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
        ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)
        sur_half_sk11 = attack_with_ND7s(ct0a=ct0a, ct1a=ct1a, ct0b=ct0b, ct1b=ct1b, t1=t1, c1=c1,
                                         net1=net1, c_bits=selected_bits)
        print('the number of surviving sk_r+1_[7~0] is ', len(sur_half_sk11))
        m_res[i][0] = len(sur_half_sk11)
        ct0a = ct0a[:n2]; ct1a=ct1a[:n2]; ct0b=ct0b[:n2]; ct1b=ct1b[:n2]
        sur_sk11 = attack_with_ND7t(ct0a=ct0a, ct1a=ct1a, ct0b=ct0b, ct1b=ct1b,
                                    sur_kg=sur_half_sk11, t2=t2, c2=c2, net2=net2)
        print('the number of surviving sk_r+1_ is ', len(sur_sk11))
        m_res[i][1] = len(sur_sk11)
        ct0a = ct0a[:n3]; ct1a = ct1a[:n3]; ct0b = ct0b[:n3]; ct1b = ct1b[:n3]
        sur_sk11_half_sk10 = attack_with_ND6s(ct0a=ct0a, ct1a=ct1a, ct0b=ct0b, ct1b=ct1b,
                                              sur_kg=sur_sk11, t3=t3, c3=c3, net3=net3, c_bits=selected_bits)
        print('the number of survivng (sk_r+1, sk_r_[7~0]) is ', len(sur_sk11_half_sk10))
        m_res[i][2] = len(sur_sk11_half_sk10)
        ct0a = ct0a[:n4]; ct1a = ct1a[:n4]; ct0b = ct0b[:n4]; ct1b = ct1b[:n4]
        sur_sk11_sk10 = attack_with_ND6t(ct0a=ct0a, ct1a=ct1a, ct0b=ct0b, ct1b=ct1b,
                                         sur_kg=sur_sk11_half_sk10, t4=t4, c4=c4, net4=net4)
        print('the number of surviving (sk_r+1, sk_r) is ', len(sur_sk11_sk10))
        m_res[i][3] = len(sur_sk11_sk10)

        print('distances between true keys and key guesses are: ')
        tp = []
        for kg_set in sur_sk11_sk10:
            if len(kg_set) == 2:
                print('sk_r+1: ', hex(sk11 ^ kg_set[0]), ' sk_r: ', hex(sk10 ^ kg_set[1]))
                tp.append((hex(sk11 ^ kg_set[0]), hex(sk10 ^ kg_set[1])))

        # save the attack res
        attack_res.append(tp)

    folder = './key_recovery_record/bayesian/2_7_6_22586_5271_5228_829/'

    np.save(folder + str(id) + '_attack_res.npy', m_res)
    # write the attack res to txt file
    file = open(folder + str(id) + '_attack_res.txt', 'w')
    index = 0
    for res in attack_res:
        file.write(str(index))
        file.write('\n')
        for kg_set in res:
            file.write(str(kg_set))
            file.write('\n')
        index = index + 1
    file.close()


# verify the negative influence of neutral bits.
# Bayesian key search strategy which has a minor negative influenc is not used.
# If you want to quickly check the negative influence of neutral bits, go to attack_on_11_round.py.
# We provide a fast version that adopts Bayesian key search strategy in attack_on_11_round.py.

recover_two_subkeys(t=50, nr=10, n1=n1, diff=(0x2800, 0x10), id=0)
recover_two_subkeys(t=50, nr=10, n1=n1, diff=(0x2800, 0x10), id=1)

# faster setting
# recover_two_subkeys(t=25, nr=10, n1=n1, diff=(0x2800, 0x10), id=0)
# recover_two_subkeys(t=25, nr=10, n1=n1, diff=(0x2800, 0x10), id=1)
# recover_two_subkeys(t=25, nr=10, n1=n1, diff=(0x2800, 0x10), id=0)
# recover_two_subkeys(t=25, nr=10, n1=n1, diff=(0x2800, 0x10), id=1)
