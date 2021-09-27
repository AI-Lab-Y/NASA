import speck as sp
import numpy as np

from pickle import dump
from keras.models import Model, load_model
from os import urandom
import random
# import gc


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [15 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + 16 * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]

    return new_x


def compare_p3_p4(n=10**7, nr=7, c3=0.5, net_path='./', bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    net = load_model(net_path)

    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(keys, nr+1)
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    tk = ks[nr]
    fk = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    t0l, t0r = sp.dec_one_round((c0l, c0r), tk)
    t1l, t1r = sp.dec_one_round((c1l, c1r), tk)
    raw_tx = sp.convert_to_binary([t0l, t0r, t1l, t1r])
    tx = extract_sensitive_bits(raw_tx, bits=bits)
    tZ = net.predict(tx, batch_size=10000)
    p_3 = np.sum(tZ > c3) / n

    f0l, f0r = sp.dec_one_round((c0l, c0r), fk)
    f1l, f1r = sp.dec_one_round((c1l, c1r), fk)
    raw_fx = sp.convert_to_binary([f0l, f0r, f1l, f1r])
    fx = extract_sensitive_bits(raw_fx, bits=bits)
    fZ = net.predict(fx, batch_size=10000)
    p_4 = np.sum(fZ > c3) / n

    print('the number of testing samples is ', n)
    print('p3 is ', p_3)
    print('p4 is ', p_4)


net5_path = './saved_model/teacher/0x0040-0x0/5_distinguisher.h5'
net6_path = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'
net7_path = './saved_model/teacher/0x0040-0x0/7_distinguisher.h5'
selected_bits = [15 - i for i in range(16)]

for i in range(5, 8):
    print('i is ', i)
    if i == 5:
        net_path = net5_path
    elif i == 6:
        net_path = net6_path
    elif i == 7:
        net_path = net7_path
    compare_p3_p4(n=10**7, nr=i, c3=0.55, net_path=net_path, bits=selected_bits)