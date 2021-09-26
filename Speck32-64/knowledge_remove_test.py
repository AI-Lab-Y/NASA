import speck as sp

import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
import random


block_size = 32


def make_diffusion_data(n, nr, diff=(0x8100, 0x8102)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1

    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(keys, nr)

    plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)

    ctdata0l, ctdata0r = sp.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = sp.encrypt((plain1l, plain1r), ks)

    # generate blinding masks
    k0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    k1 = np.frombuffer(urandom(2*n), dtype=np.uint16)

    # apply blinding masks to C0 or C1
    masked_ctdata0l = ctdata0l ^ k0; masked_ctdata0r = ctdata0r ^ k1
    # ctdata1l = ctdata1l ^ k0; ctdata1r = ctdata1r ^ k1

    X = sp.convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    masked_X = sp.convert_to_binary([masked_ctdata0l, masked_ctdata0r, ctdata1l, ctdata1r])

    return(X, masked_X, Y)


def test_knowledge_remove(n=10**7, nr=7, net_path='./', diff=(0x0040, 0x0)):
    print('nr is ', nr)
    X, masked_X, Y = make_diffusion_data(n=n, nr=nr, diff=diff)
    if nr != 8:
        net = load_model(net_path)
    else:
        json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
        json_model = json_file.read()
        net = model_from_json(json_model)
        net.load_weights('./saved_model/teacher/0x0040-0x0/net8_small.h5')
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    loss, acc_0 = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is ', acc_0)

    loss, acc_1 = net.evaluate(masked_X, Y, batch_size=10000, verbose=0)
    print('after removing knowledge, the acc is ', acc_1)


for i in range(5, 9):
    net_path = './saved_model/teacher/0x0040-0x0/' + str(i) + '_distinguisher.h5'
    test_knowledge_remove(n=10**7, nr=i, net_path=net_path, diff=(0x0040, 0x0))