import speck as sp
import numpy as np

from keras.models import model_from_json
from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time

WORD_SIZE = sp.WORD_SIZE()
MASK_VAL = 2 ** WORD_SIZE - 1


#make a plaintext structure
def make_structure(pt0, pt1, diff=(0x211, 0xa04), neutral_bits=[20, 21, 22, 14, 15]):
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for i in neutral_bits:
        d = 1 << i
        d0 = d >> WORD_SIZE
        d1 = d & MASK_VAL
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return(p0,p1,p0b,p1b)


#generate a Speck key, return expanded key
def gen_key(nr):
    key = np.frombuffer(urandom(8), dtype=np.uint16)
    ks = sp.expand_key(key, nr)
    return(ks)


def gen_plain(n):
    pt0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    pt1 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    return(pt0, pt1)


def check_neutral_bit(n, nr, in_diff=(0x211, 0xa04), out_diff=(0x0040, 0), tested_bit=[0]):
    pt0, pt1 = gen_plain(n)
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=in_diff, neutral_bits=tested_bit)
    key = gen_key(nr)
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), key)
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), key)

    ct_diff0 = np.squeeze(ct0a ^ ct0b)
    ct_diff1 = np.squeeze(ct1a ^ ct1b)
    # print('ct_diif0 is ', ct_diff0, ' ct_diff1 is ', ct_diff1)

    if np.sum(ct_diff0 == out_diff[0]) == 2 and np.sum(ct_diff1 == out_diff[1]) == 2:
        return 2
    elif ct_diff0[0] == out_diff[0] and ct_diff1[0] == out_diff[1]:
        return 1
    else:
        return 0


neutrality = np.zeros(WORD_SIZE*2)
for i in range(WORD_SIZE*2):
    neutral_bit = []
    neutral_bit.append(i)
    passed_num = 0
    neutral_num = 0
    for j in range(2**14):
        res = check_neutral_bit(n=1, nr=1, in_diff=(0x211, 0xa04), out_diff=(0x2800, 0x10), tested_bit=neutral_bit)
        if res == 1:
            passed_num = passed_num + 1
        elif res == 2:
            passed_num = passed_num + 1
            neutral_num = neutral_num + 1
    neutrality[i] = neutral_num / passed_num
    print('i is ', i, ' neutral res is ', neutrality[i])

p_0 = 2**(-4)
selected_bits = [0,1,3,4,5,11,14,15,20,21,22,23,24,26,27,28]
prob = 1.0
for i in selected_bits:
    prob = prob * neutrality[i]
print('the probability that a plaintext structure created from these bits is ', log2(prob * p_0))