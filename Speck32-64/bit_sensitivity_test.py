import speck as sp

import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
import copy
import random

block_size = 32


# type = 0, cal sen_0
# type = 1, cal sen_1
# type = 2, cal sen_01
def make_target_bit_diffusion_data(X, id=0, type=0):
    n = X.shape[0]
    masks = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
    masked_X = copy.deepcopy(X)
    if type == 0:
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
    elif type == 1:
        masked_X[:, block_size*2 - 1 - id] = X[:, block_size*2 - 1 - id] ^ masks
    else:
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
        masked_X[:, block_size * 2 - 1 - id] = X[:, block_size * 2 - 1 - id] ^ masks

    return masked_X


def test_bits_sensitivity(n=10**7, nr=7, net_path='./', diff=(0x0040, 0x0), folder='./bits_sensitive_res/'):
    acc = np.zeros(block_size+1)
    X, Y = sp.make_train_data(n=n, nr=nr, diff=diff)
    if nr != 8:
        net = load_model(net_path)
    else:
        json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
        json_model = json_file.read()
        net = model_from_json(json_model)
        net.load_weights(net_path)
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    loss, acc[block_size] = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is ', acc[block_size])

    for i in range(block_size):
        masked_X = make_target_bit_diffusion_data(X, id=i, type=0)
        loss, acc[i] = net.evaluate(masked_X, Y, batch_size=10000, verbose=0)
        print('cur bit position is ', i)
        print('the decrease of the acc is ', acc[block_size] - acc[i])

    np.save(folder + str(nr) + '_distinguisher_bit_sensitivity.npy', acc)


# net_path = './saved_model/teacher/0x0040-0x0/5_distinguisher.h5'
# net_path = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'
net_path = './saved_model/teacher/0x0040-0x0/7_distinguisher.h5'
# net_path = './saved_model/teacher/0x0040-0x0/net8_small.h5'
folder = './bits_sensitivity_res/0x0040-0x0/'
test_bits_sensitivity(n=10**6, nr=7, net_path=net_path, diff=(0x0040, 0x0), folder=folder)