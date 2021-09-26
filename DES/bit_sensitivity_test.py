from numpy.core.shape_base import block
import des
import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
import random


block_size = 64


def convert_mode_to_binary(arr):
    a = arr.shape[0]
    b = 32
    binary_arr = np.zeros((a, b), dtype=np.uint8)
    for j in range(b):
        binary_arr[:, j] = (arr >> (31-j)) & 1

    return binary_arr


# if type = 0, randomize all the bits
def make_target_bit_diffusion_mode(id=0, n=10**7, type=1):
    # generate blinding values for target bits of the positive data
    k0 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    k1 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    # randomize the distribution of the (id)th bit
    if type == 1:
        if id < 32:
            k1 = k1 & (1 << id)
            k0 = k0 & 0
        else:
            k0 = k0 & (1 << (id - 32))
            k1 = k1 & 0

    k0 = convert_mode_to_binary(k0)
    k1 = convert_mode_to_binary(k1)

    return k0, k1


def test_bits_sensitivity(nr=7, n=10**6, net_path='./', diff=(0x0400, 0x200008), type=1):
    acc = np.zeros(block_size+1)
    net = load_model(net_path)
    X_eval, Y_eval = des.make_dataset(n=n, nr=nr, diff=diff)
    loss, acc[block_size] = net.evaluate(X_eval, Y_eval, batch_size=10000, verbose=0)
    print('The initial acc is ', acc[block_size])

    tx0, tx1, tx2, tx3 = X_eval[:, 0:32], X_eval[:, 32:64], X_eval[:, 64:96], X_eval[:, 96:128]

    if type == 1:
        for i in range(block_size):
            k0, k1 = make_target_bit_diffusion_mode(id=i, n=X_eval.shape[0], type=type)
            x = np.concatenate((tx0 ^ k0, tx1 ^ k1, tx2, tx3), axis=1)
            # print('new x shape is ', np.shape(x))
            loss, acc[i] = net.evaluate(x, Y_eval, batch_size=10000, verbose=0)
            print('cur bit position is ', i)
            print('the decrease of the acc is ', acc[block_size] - acc[i])
        # sort all bits according to their bit sensitity
        acc_arg = np.argsort(acc[:block_size])
        for arg in acc_arg:
            print('cur bit position is ', arg)
            print('the decrease of the acc is {:.5f}'.format(acc[block_size] - acc[arg]))

        np.save('./bits_sensitivity_res/{}/{}_distinguisher_bit_sensitivity.npy'.format(delta_S, nr), acc)
    else:
        k0, k1 = make_target_bit_diffusion_mode(id=-1, n=X_eval.shape[0], type=type)
        x = np.concatenate((tx0 ^ k0, tx1 ^ k1, tx2, tx3), axis=1)
        print('new x shape is ', np.shape(x))
        loss, acc_random = net.evaluate(x, Y_eval, batch_size=10000, verbose=0)
        print('when we randomize all the bits, the decrease of the acc is ', acc[block_size] - acc_random)


delta_S = '0x19600000-0x0'
net_nr = 5
net_path = './saved_model/teacher/{}/{}_distinguisher.h5'.format(delta_S, net_nr)
test_bits_sensitivity(nr=5, n=10**6, net_path=net_path, diff=(0x19600000, 0x0), type=1)
