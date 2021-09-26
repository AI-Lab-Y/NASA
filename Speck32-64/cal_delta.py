import speck as sp
import numpy as np
import heapq

from pickle import dump
from keras.models import Model, load_model, model_from_json
from os import urandom
import time


MASK_VAL = 2 ** sp.WORD_SIZE() - 1
word_size = sp.WORD_SIZE()


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    id1 = [v + word_size * i for i in range(4) for v in id0]
    new_x = raw_x[:, id1]

    return new_x


def make_target_diff_samples(n=2**12, nr=10, diff=(0x2800, 0x10)):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]

    key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(key, nr)

    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r


def full_speck_encryption_speed(nr, n):
    p0l = np.frombuffer(urandom(2), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2), dtype=np.uint16)
    p0l, p0r = p0l.repeat(n), p0r.repeat(n)
    start = time.time()
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(keys, nr)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    end = time.time()
    print('the total time is ', end - start)
    return end - start


def partial_decryption_NDs_inference_speed(nr, n, bs, net='./', diff=(0x40, 0), bits=[14, 13]):
    ND = load_model(net)
    c0l, c0r, c1l, c1r = make_target_diff_samples(n=n, nr=nr, diff=diff)
    start = time.time()
    key_guess = np.frombuffer(urandom(2*n), dtype=np.uint16)
    t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
    t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
    raw_X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
    X = extract_sensitive_bits(raw_X, bits=bits)
    Z = ND.predict(X, batch_size=bs)
    end = time.time()
    print('the total time is ', end - start)
    return end - start


def partial_decryption_NDt_inference_speed(nr, n, bs, net='./', diff=(0x40, 0)):
    if nr == 8:
        json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
        json_model = json_file.read()
        ND = model_from_json(json_model)
        ND.load_weights('./saved_model/teacher/0x0040-0x0/net8_small.h5')
        ND.compile(optimizer='adam', loss='mse', metrics=['acc'])
    else:
        ND = load_model(net)
    c0l, c0r, c1l, c1r = make_target_diff_samples(n=n, nr=nr, diff=diff)
    start = time.time()
    key_guess = np.frombuffer(urandom(2*n), dtype=np.uint16)
    t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
    t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
    X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
    Z = ND.predict(X, batch_size=bs)
    end = time.time()
    print('the total time is ', end - start)
    return end - start


def cal_delta(nr_enc, nr_nd, n, bs, net='./', diff=(0x40, 0), bits=[14, 13], type=1):
    encryption_speed_of_full_speck = full_speck_encryption_speed(nr=nr_enc, n=n)
    if type == 1:  # teacher distinguisher
        attack_speed_with_NDt = partial_decryption_NDt_inference_speed(nr=nr_nd, n=n, bs=bs, net=net, diff=diff)
        delta = attack_speed_with_NDt / encryption_speed_of_full_speck
    else:
        attack_speed_with_NDs = partial_decryption_NDs_inference_speed(nr=nr_nd, n=n, bs=bs, net=net, diff=diff,
                                                                       bits=bits)
        delta = attack_speed_with_NDs / encryption_speed_of_full_speck
    print('the value of delta is ', delta)
    return delta


# for 13(3+8+2) round Speck32/64
selected_bits = [15 - i for i in range(16)]
cal_delta(nr_enc=13, nr_nd=8, n=2**20, bs=2**15, net='./', diff=(0x40, 0), bits=selected_bits, type=1)
# delta = 10

