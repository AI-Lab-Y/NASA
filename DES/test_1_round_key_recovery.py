import des
import numpy as np
from os import urandom, path, mkdir
import time


word_size = 32
'''
Due to memory limitation, samples is generated
in batches during attack.
Batch num is the number of batches
'''
# batch_num = 4


# construct input of student distinguisher according to bits
def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + word_size * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]

    return new_x


# get n ciphertext pairs satisfying: P0 ^ P1 == diff
# if mk is not None, encrypt plaintext with mk, otherwise encrypt plaintext with a random master key
# return n ciphertext pairs and subkey of the very last round
def make_target_diff_samples(n=10**7, nr=3, diff=(0x80, 0x0), mk=None):
    x0l = np.frombuffer(urandom(n * 4), dtype=np.uint32)  # .reshape(-1, 1)
    x0r = np.frombuffer(urandom(n * 4), dtype=np.uint32)  # .reshape(-1, 1)
    x1l = x0l ^ diff[0]
    x1r = x0r ^ diff[1]
    p0l = np.zeros((n, 32), dtype=np.uint8)
    p0r = np.zeros((n, 32), dtype=np.uint8)
    p1l = np.zeros((n, 32), dtype=np.uint8)
    p1r = np.zeros((n, 32), dtype=np.uint8)
    for i in range(32):
        off = 31 - i
        p0l[:, i] = (x0l >> off) & 1
        p0r[:, i] = (x0r >> off) & 1
        p1l[:, i] = (x1l >> off) & 1
        p1r[:, i] = (x1r >> off) & 1

    if mk is None:
        master_keys = np.frombuffer(urandom(8), dtype=np.uint32).reshape(-1, 2)
    else:
        master_keys = mk
    keys = np.zeros((n, 64), dtype=np.uint8)
    for i in range(32):
        keys[:, i] = (master_keys[:, 0] >> (31 - i)) & 1
        keys[:, 32 + i] = (master_keys[:, 1] >> (31 - i)) & 1
    subkeys = des.expand_key(keys, nr)

    c0l, c0r = des.encrypt(p0l, p0r, subkeys)
    c1l, c1r = des.encrypt(p1l, p1r, subkeys)
    return c0l, c0r, c1l, c1r, subkeys[nr-1][0]


def extract_true_key(sk, k_bits=[18,19,20,21,22,23]):
    true_key = 0
    k_len = len(k_bits)
    for i in range(k_len):
        true_key = true_key + (sk[k_bits[i]] << (k_len - 1 - i))

    return true_key.astype(np.uint8)


def key_recovery_attack(t=100, n=2**12, th=11070, nr=10, c3=0.55, table_path="", diff=(0x200008, 0x0400), bits=[26,16,8,2], k_bits=[0,1,2,3,4,5,6]):
    global batch_num
    print('batch num: {}'.format(batch_num))
    # accelerate the distinguishing process by looking up the inference table
    inference_table = np.load(table_path)
    acc = 0
    cnt = np.zeros((t, 2 ** 6), dtype=np.uint64)
    tk = np.zeros(t, dtype=np.uint8)
    fks = np.zeros(((n // batch_num) + 1, 48), dtype=np.uint8)
    time_cost = np.zeros(t)
    for i in range(t):
        print('i is ', i)
        nx = n
        # the number of samples that should be generated in each batch
        sub_n = (n // batch_num) + 1
        # generate master key for this attack in advance
        master_keys = np.frombuffer(urandom(8), dtype=np.uint32).reshape(-1, 2)
        keys = np.zeros((1, 64), dtype=np.uint8)
        for j in range(32):
            keys[:, j] = (master_keys[:, 0] >> (31 - j)) & 1
            keys[:, 32 + j] = (master_keys[:, 1] >> (31 - j)) & 1
        subkeys = des.expand_key(keys, nr)[nr-1][0]
        tk[i] = extract_true_key(subkeys, k_bits=k_bits)
        print('true key is ', tk[i])
        start = time.time()
        # in each sample batch
        while(nx > 0):
            print("rest n is", nx)
            if sub_n >= nx:
                sub_n = nx
            nx -= sub_n
            c0l, c0r, c1l, c1r, subkeys = make_target_diff_samples(sub_n, nr=nr, diff=diff, mk=master_keys)
            for key_guess in range(2 ** 6):
                # print("key guess is", key_guess)
                for j in range(6):
                    fks[:sub_n, k_bits[j]] = (key_guess >> (5 - j)) & 1
                d0r, d0l = des.dec_one_round(c0r, c0l, fks[:sub_n])
                d1r, d1l = des.dec_one_round(c1r, c1l, fks[:sub_n])

                X = np.concatenate((d0l, d0r, d1l, d1r), axis=1)
                extracted_x = extract_sensitive_bits(X, bits=bits)

                # look up the inference table
                extracted_x = des.array_to_uint([extracted_x])[0]
                z = inference_table[extracted_x]

                cur_num = np.sum(z > c3)
                cnt[i][key_guess] += cur_num

        # When all batches has been processed, this attack is over.
        end = time.time()
        time_cost[i] = end - start
        bk = np.argmax(cnt[i])
        print('time cost is ', time_cost[i])
        print('difference between bk and true key is ', hex(bk ^ tk[i]))
        print('the number of surviving keys is ', np.sum(cnt[i, :] > th))

        if bk == tk[i]:
            acc += 1
    
    acc = acc / t
    print('acc is ', acc)
    print('average time cost is ', np.mean(time_cost))
    return cnt, tk, time_cost

# 6-round attack for Sbox 5
batch_num = 1
s_box = 5
c2 = 0.5
pre_nr = 0
dis_nr = 5
delta_S = "0x19600000-0x0"
table_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, dis_nr, s_box)
n = 634
th = 351
k_bits = [(s_box-1)*6 + i for i in range(6)]
diff = (0x19600000, 0x0)
cnt, tk, time_cost = key_recovery_attack(t=100, n=n, th=th, nr=pre_nr+dis_nr+1, c3=c2, table_path=table_path, diff=diff, bits=des.Sbox_output[s_box], k_bits=k_bits)
saved_folder = './key_recovery_record/{}_{}_{}_{}'.format(pre_nr, dis_nr, n, th)
if not path.exists(saved_folder):
    mkdir(saved_folder)
saved_folder += '/'
np.save(saved_folder + 'cnt.npy', cnt)
np.save(saved_folder + 'true_key.npy', tk)
np.save(saved_folder + 'time_cost.npy', time_cost)

# 6-round attack for Sbox 8
# batch_num = 1
# s_box = 8
# c2 = 0.5
# pre_nr = 0
# dis_nr = 5
# delta_S = "0x19600000-0x0"
# table_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, dis_nr, s_box)
# n = 1289
# th = 701
# k_bits = [(s_box-1)*6 + i for i in range(6)]
# diff = (0x19600000, 0x0)
# cnt, tk, time_cost = key_recovery_attack(t=100, n=n, th=th, nr=pre_nr+dis_nr+1, c3=c2, table_path=table_path, diff=diff, bits=des.Sbox_output[s_box], k_bits=k_bits)
# saved_folder = './key_recovery_record/{}_{}_{}_{}'.format(pre_nr, dis_nr, n, th)
# if not path.exists(saved_folder):
#     mkdir(saved_folder)
# saved_folder += '/'
# np.save(saved_folder + 'cnt.npy', cnt)
# np.save(saved_folder + 'true_key.npy', tk)
# np.save(saved_folder + 'time_cost.npy', time_cost)

# 8-round attack for Sbox 5
# the time cost of one attack is about 3600s
# if you have enough memory, batch_num can be set smaller
# batch_num = 4
# s_box = 5
# c2 = 0.5
# pre_nr = 2
# dis_nr = 5
# delta_S = "0x19600000-0x0"
# table_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, dis_nr, s_box)
# n = 35537762
# th = 17387770
# k_bits = [(s_box-1)*6 + i for i in range(6)]
# diff = (0x19600000, 0x0)
# cnt, tk, time_cost = key_recovery_attack(t=100, n=n, th=th, nr=pre_nr+dis_nr+1, c3=c2, table_path=table_path, diff=diff, bits=des.Sbox_output[s_box], k_bits=k_bits)
# saved_folder = './key_recovery_record/{}_{}_{}_{}'.format(pre_nr, dis_nr, n, th)
# if not path.exists(saved_folder):
#     mkdir(saved_folder)
# saved_folder += '/'
# np.save(saved_folder + 'cnt.npy', cnt)
# np.save(saved_folder + 'true_key.npy', tk)
# np.save(saved_folder + 'time_cost.npy', time_cost)

# 8-round attack for Sbox 8
# if you have enough memory, batch_num can be set smaller
# batch_num = 4
# s_box = 8
# c2 = 0.5
# pre_nr = 2
# dis_nr = 5
# delta_S = "0x19600000-0x0"
# table_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, dis_nr, s_box)
# n = 71563482
# th = 35096958
# k_bits = [(s_box-1)*6 + i for i in range(6)]
# diff = (0x19600000, 0x0)
# cnt, tk, time_cost = key_recovery_attack(t=1, n=n, th=th, nr=pre_nr+dis_nr+1, c3=c2, table_path=table_path, diff=diff, bits=des.Sbox_output[s_box], k_bits=k_bits)
# saved_folder = './key_recovery_record/{}_{}_{}_{}'.format(pre_nr, dis_nr, n, th)
# if not path.exists(saved_folder):
#     mkdir(saved_folder)
# saved_folder += '/'
# np.save(saved_folder + 'cnt.npy', cnt)
# np.save(saved_folder + 'true_key.npy', tk)
# np.save(saved_folder + 'time_cost.npy', time_cost)

# 10-round attack for Sbox 5
# if you have enough memory, batch_num can be set smaller
# batch_num = int(2**20)
# s_box = 5
# c2 = 0.5
# pre_nr = 4
# dis_nr = 5
# delta_S = "0x19600000-0x0"
# table_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, dis_nr, s_box)
# n = 1946099202926 # 2**40.824
# th = 951644804794
# k_bits = [(s_box-1)*6 + i for i in range(6)]
# diff = (0x19600000, 0x0)
# cnt, tk, time_cost = key_recovery_attack(t=100, n=n, th=th, nr=pre_nr+dis_nr+1, c3=c2, table_path=table_path, diff=diff, bits=des.Sbox_output[s_box], k_bits=k_bits)
# saved_folder = './key_recovery_record/{}_{}_{}_{}'.format(pre_nr, dis_nr, n, th)
# if not path.exists(saved_folder):
#     mkdir(saved_folder)
# saved_folder += '/'
# np.save(saved_folder + 'cnt.npy', cnt)
# np.save(saved_folder + 'true_key.npy', tk)
# np.save(saved_folder + 'time_cost.npy', time_cost)

# 10-round attack for Sbox 8
# if you have enough memory, batch_num can be set smaller
# batch_num = int(2**20)
# s_box = 8
# c2 = 0.5
# pre_nr = 4
# dis_nr = 5
# delta_S = "0x19600000-0x0"
# table_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, dis_nr, s_box)
# n = 3918761966411 # 2**41.834
# th = 1920980986510
# k_bits = [(s_box-1)*6 + i for i in range(6)]
# diff = (0x19600000, 0x0)
# cnt, tk, time_cost = key_recovery_attack(t=100, n=n, th=th, nr=pre_nr+dis_nr+1, c3=c2, table_path=table_path, diff=diff, bits=des.Sbox_output[s_box], k_bits=k_bits)
# saved_folder = './key_recovery_record/{}_{}_{}_{}'.format(pre_nr, dis_nr, n, th)
# if not path.exists(saved_folder):
#     mkdir(saved_folder)
# saved_folder += '/'
# np.save(saved_folder + 'cnt.npy', cnt)
# np.save(saved_folder + 'true_key.npy', tk)
# np.save(saved_folder + 'time_cost.npy', time_cost)