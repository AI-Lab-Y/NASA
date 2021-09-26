import numpy as np
from os import urandom
import speck as sp
from keras.models import load_model


selected_bits = [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]
word_size = sp.WORD_SIZE()


def make_target_diff_samples(n, nr, diff=(0x808000, 0x808004)):
    keys = np.frombuffer(urandom(16), dtype=np.uint64).reshape(2, -1)
    ks = sp.expand_key(keys, nr)
    plain0l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    plain0r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    plain0l, plain0r = sp.dec_one_round((plain0l, plain0r), 0)
    plain1l, plain1r = sp.dec_one_round((plain1l, plain1r), 0)

    ctdata0l, ctdata0r = sp.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = sp.encrypt((plain1l, plain1r), ks)

    return ctdata0l, ctdata0r, ctdata1l, ctdata1r, ks[nr-1] & 0x3fff


def extract_sensitive_bits(raw_x, bits=selected_bits):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    id1 = [v + word_size * i for i in range(4) for v in id0]
    new_x = raw_x[:, id1]
    return new_x


def key_recovery_attack(t, n, th, nr, net, c3=0.55, diff=(0x808000, 0x808004), speed_flag=True):
    acc = 0
    for i in range(t):
        print('cur attack_index is ', i)
        surviv_key = []
        c0l, c0r, c1l, c1r, tk = make_target_diff_samples(n=n, nr=nr, diff=diff)
        if speed_flag:
            c0l, c0r, c1l, c1r = np.tile(c0l, 2**14), np.tile(c0r, 2**14), np.tile(c1l, 2**14), np.tile(c1r, 2**14)
            keys = np.array([np.uint64(kg) for kg in range(2**14)])
            ks = keys.repeat(n)
            t0l, t0r = sp.dec_one_round((c0l, c0r), ks)
            t1l, t1r = sp.dec_one_round((c1l, c1r), ks)
            raw_X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            X = extract_sensitive_bits(raw_X, selected_bits)
            Z = net.predict(X, batch_size=10000)
            Z = np.squeeze(Z)
            Z = Z.reshape(2**14, -1)
            cnt = np.sum(Z > c3, axis=1)
            surviv_key = [np.uint64(i) for i in range(2**14) if cnt[i] > th]
        else:
            for kg in range(2 ** 14):
                t0l, t0r = sp.dec_one_round((c0l, c0r), kg)
                t1l, t1r = sp.dec_one_round((c1l, c1r), kg)
                raw_X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
                X = extract_sensitive_bits(raw_X, selected_bits)
                Z = net.predict(X, batch_size=n)
                Z = np.squeeze(Z)
                if np.sum(Z > c3) > th:
                    surviv_key.append(kg)
        if tk in surviv_key:
            acc = acc + 1
            print('true key survives. ')
        print('the number of surviving keys is ', len(surviv_key))
        print('the differnces between surviving kg and sk are')
        for kg in surviv_key:
            print(hex(tk[0] ^ np.uint64(kg)))


# 1 + 7 + 1
# the number of survivng keys should not exceed 16
# This attack is fast
# we have provided an attack record in the dictionary './key_recovery_record'
nd = load_model('./saved_model/student/0x80-0x0/soft_label/21_8_student_{}_distinguisher.h5'.format(7))
key_recovery_attack(t=100, n=86, th=56, nr=9, net=nd, c3=0.5, diff=(0x80, 0x0), speed_flag=True)

# 2 + 7 + 1
# the number of survivng keys should not exceed 16
# This attack is fast
nd = load_model('./saved_model/student/0x80-0x0/soft_label/21_8_student_{}_distinguisher.h5'.format(7))
key_recovery_attack(t=100, n=1240, th=432, nr=10, net=nd, c3=0.5, diff=(0x9000, 0x10), speed_flag=True)