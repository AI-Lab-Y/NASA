import des
import numpy as np
from keras.models import Model, load_model
from os import urandom
import random


word_size = 32


# p1 = tp_tk, p2 = tp_fk, p3 = tn_tk, p4 = tn_fk
# p1, p3, p4 are easy to calculate, p2 is related with the hamming distance between tk and fk


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + word_size * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]
    # print('new_x shape is ', np.shape(new_x))

    return new_x

# evaluate accuracy of a distinguisher
def show_distinguisher_acc(n=10**7, nr=7, net_path='', diff=(0x0400, 0x200008), bits=[14, 13, 12, 11, 10, 9, 8]):
    net = load_model(net_path)
    raw_x = des.make_target_diff_samples(n=n, nr=nr, diff_type=1, diff=diff, return_keys=0)
    x = extract_sensitive_bits(raw_x, bits=bits)
    y = net.predict(x, batch_size=10000)
    y = np.squeeze(y)
    tp = np.sum(y > 0.5) / n
    fn = 1 - tp

    raw_x = des.make_target_diff_samples(n=n, nr=nr, diff_type=0, return_keys=0)
    x = extract_sensitive_bits(raw_x, bits=bits)
    y = net.predict(x, batch_size=10000)
    y = np.squeeze(y)
    tn = np.sum(y <= 0.5) / n
    fp = 1 - tn

    print('acc of cur distinguisher is ', (tp+tn)/2)
    print('tp_to_tp: ', tp, ' tp_to_fn: ', fn, ' tn_to_tn: ', tn, ' tn_to_fp: ', fp)

# evaluate p1 and p3
def cal_p1_p3(n=10**7, nr=7, c3=0.5, table_path='', diff=(0x0400, 0x200008), bits=[14, 13, 12, 11, 10, 9, 8]):
    # net = load_model(net_path)
    inference_table = np.load(table_path)
    p1 = 0
    p3 = 0
    acc = [0, 0, 0, 0]
    for i in range(1, 4):
        if i == 2:
            continue
        if i == 1:
            c0l, c0r, c1l, c1r, subkeys = des.make_target_diff_samples(n=n, nr=nr+1, diff_type=1, diff=diff, return_keys=1)
            d0r, d0l = des.dec_one_round(c0r, c0l, subkeys[nr])       # change the ciphertext order just one time
            d1r, d1l = des.dec_one_round(c1r, c1l, subkeys[nr])
        elif i == 3:
            c0l, c0r, c1l, c1r, subkeys = des.make_target_diff_samples(n=n, nr=nr+1, diff_type=0, return_keys=1)
            random_keys = np.frombuffer(urandom(n * 8), dtype=np.uint64)        # .reshape(-1, 2)
            random_subkeys = np.zeros((n, 48), dtype=np.uint8)
            for j in range(48):
                random_subkeys[:, j] = (random_keys >> (47 - j)) & 1
            d0r, d0l = des.dec_one_round(c0r, c0l, random_subkeys)      # change the ciphertext order just one time
            d1r, d1l = des.dec_one_round(c1r, c1l, random_subkeys)
        X = np.concatenate((d0l, d0r, d1l, d1r), axis=1)
        extracted_x = extract_sensitive_bits(X, bits=bits)
        extracted_x = des.array_to_uint([extracted_x])[0]
        z = inference_table[extracted_x]
        acc[i] = np.sum(z > c3) / n
    
    p1 = acc[1]
    p3 = acc[3]

    print('p1 : ', p1, ' p3 : ', p3)

    return p1, p3


# need to be tested
def gen_fk(arr):
    fk = np.zeros(48, dtype=np.uint8)
    for v in arr:
        fk[v] = 1

    return fk

# evaluate p2 according to d1
def cal_p2_d1_for_des(n=10**7, nr=7, c3=0.55, net_path='', diff=(0x0400, 0x200008), bits=[26,16,8,2], k_bits=[0,1,2,3,4,5,6]):
    net = load_model(net_path)
    d1 = len(k_bits)
    p2_d1 = np.zeros(d1+1)

    sample_range = k_bits
    for i in range(d1+1):
        print('cur i is ', i)
        c0l, c0r, c1l, c1r, subkeys = des.make_target_diff_samples(n=n, nr=nr+1, diff_type=1, diff=diff, return_keys=1)
        fks = subkeys[nr]
        random_modes = np.array([gen_fk(random.sample(sample_range, i)) for j in range(n)], dtype=np.uint8)
        fks = fks ^ random_modes

        d0r, d0l = des.dec_one_round(c0r, c0l, fks)
        d1r, d1l = des.dec_one_round(c1r, c1l, fks)

        X = np.concatenate((d0l, d0r, d1l, d1r), axis=1)
        extracted_x = extract_sensitive_bits(X, bits=bits)
        z = net.predict(extracted_x, batch_size=10000)
        p2_d1[i] = np.sum(z > c3) / n
        print('cur p2_d1 is ', p2_d1[i])

    # np.save('./p2_estimation_res/student/{}/{}_box{}_{}_p2_d1.npy'.format(delta_S, nr, s_box, c3), p2_d1)

nr = 5
# s_box = 5
s_box = 8
delta_S = '0x19600000-0x0'
net_path = './saved_model/student/{}/student_{}_box{}_distinguisher.h5'.format(delta_S, nr, s_box)
table_path = './student_inference_table/{}/student_{}_box{}_distinguisher.npy'.format(delta_S, nr, s_box)
selected_bits_1 = des.Sbox_output[s_box]
selected_bits_2 = [word_size - 1 - i for i in range(word_size)]
k_bits_1 = [(s_box-1)*6 + i for i in range(6)]

show_distinguisher_acc(n=10**6, nr=nr, net_path=net_path, diff=(0x19600000,0x0), bits=selected_bits_1)
cal_p1_p3(n=10**6, nr=nr, c3=0.5, table_path=table_path, diff=(0x19600000,0x0), bits=selected_bits_1)
cal_p2_d1_for_des(n=10**6, nr=nr, c3=0.5, net_path=net_path, diff=(0x19600000,0x0), bits=selected_bits_1, k_bits=k_bits_1)