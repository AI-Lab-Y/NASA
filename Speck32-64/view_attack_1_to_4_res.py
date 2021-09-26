import numpy as np

# attack 1
res_path = './key_recovery_record/3_5_0.55_15905_758_cnt_record.npy'
true_key = np.load('./key_recovery_record/3_5_0.55_15905_758_true_keys.npy')
bits_num = 16

# attack 2
# res_path = './key_recovery_record/2_6_0.55_475_101_cnt_record.npy'
# true_key = np.load('./key_recovery_record/2_6_0.55_475_101_true_keys.npy')
# bits_num = 16

# attack 3
# res_path = './key_recovery_record/2_7_0.55_5272_1325_cnt_record.npy'
# true_key = np.load('./key_recovery_record/2_7_0.55_5272_1325_true_keys.npy')
# bits_num = 16

# attack 4
# res_path = './key_recovery_record/1_8_0.5_25680_13064_cnt_record.npy'
# true_key = np.load('./key_recovery_record/1_8_0.5_25680_13064_true_keys.npy')
# bits_num = 16


def hw(v):
  res = np.zeros(v.shape, dtype=np.uint8)
  for i in range(16):
    res = res + ((v >> i) & 1)

  return(res)


low_weight = np.array(range(2**bits_num), dtype=np.uint16)
low_weight = hw(low_weight)
# print('low weight is ', low_weight)
# num = [1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1]
num = [np.sum(low_weight == i) for i in range(bits_num+1)]


# view the result of attack 1 to 4
def analyze_attack_res(t=9701, res_path=res_path, true_keys=true_key):
    res = np.load(res_path)
    print('res shape is : ', np.shape(res))
    true_keys = true_keys.astype(np.uint16)
    print('true keys are: ', true_keys)

    true_key_cnt = np.zeros(true_keys.shape[0])
    for i in range(true_keys.shape[0]):
        true_key_cnt[i] = res[i][true_keys[i]]
    print('true key cnt is ', true_key_cnt)
    print('min value is ', min(true_key_cnt))
    print('max value is ', max(true_key_cnt))

    survive_key_num = np.sum(res > t, axis=1)
    print('survive_key_num shape is ', np.shape(survive_key_num))
    print('survive key num is: ', survive_key_num)

    dis = np.zeros((true_keys.shape[0], bits_num+1))
    flo = np.zeros((true_keys.shape[0], bits_num+1))
    for i in range(true_keys.shape[0]):
        for j in range(2**bits_num):
            if res[i][j] > t:
                dp = true_keys[i] ^ np.uint64(j)
                dp = low_weight[dp]
                dis[i][dp] = dis[i][dp] + 1

        for j in range(bits_num+1):
            flo[i][j] = dis[i][j]

    pass_rate = np.mean(flo, axis=0)
    print('The average number of survived keys is ', np.sum(pass_rate))
    print('The average number of each subset is ', pass_rate)
    for i in range(bits_num+1):
        pass_rate[i] = pass_rate[i] / num[i]
    print('The average pass rate is  ', pass_rate)


def analyze_attack_res_with_reduced_key_space(t1=6676, t2=1066, res_path=res_path, true_keys=true_key):
    res = np.load(res_path)
    print('res shape is : ', np.shape(res))
    true_keys = true_keys.astype(np.uint16)
    print('true keys are: ', true_keys)

    true_key_cnt = np.zeros(true_keys.shape[0])
    for i in range(true_keys.shape[0]):
        true_key_cnt[i] = res[i][true_keys[i]]
    print('true key cnt is ', true_key_cnt)
    print('min value is ', min(true_key_cnt))
    print('max value is ', max(true_key_cnt))

    # calculate the surviving number in the first stage
    tp = np.zeros((true_keys.shape[0], 256), dtype=np.uint16)
    for i in range(true_keys.shape[0]):
        for j in range(2**16):
            if res[i][j] > 0:
                tp[i][j & 0xff] = 1
    print('The average number of survived keys in the first stage is ', np.mean(np.sum(tp, axis=1)))

    survive_key_num = np.sum(res > t2, axis=1) - np.sum(res > t1, axis=1)
    print('survive_key_num shape is ', np.shape(survive_key_num))
    print('survive key num is: ', survive_key_num)

    dis = np.zeros((true_keys.shape[0], bits_num + 1))
    flo = np.zeros((true_keys.shape[0], bits_num + 1))
    for i in range(true_keys.shape[0]):
        for j in range(2 ** bits_num):
            if t2 < res[i][j] and res[i][j]< t1:
                dp = true_keys[i] ^ np.uint64(j)
                dp = low_weight[dp]
                dis[i][dp] = dis[i][dp] + 1

        for j in range(bits_num + 1):
            flo[i][j] = dis[i][j]

    pass_rate = np.mean(flo, axis=0)
    print('The average number of survived keys of the whole attack is ', np.sum(pass_rate))
    print('The average number of each subset is ', pass_rate)
    for i in range(bits_num + 1):
        pass_rate[i] = pass_rate[i] / num[i]
    print('The average pass rate is  ', pass_rate)


# res_path, true_key need to be set correctly (see the top of this file)
# attack 1
# for th = 758, dc = 15905，attack 9 round peck32/64, 2**16 keys, \beta _n = 2**(-16)
analyze_attack_res(t=758, res_path=res_path, true_keys=true_key)

# attack 2
# for th = 101, dc = 475，attack 9 round peck32/64, 2**16 keys, \beta _n = 2**(-16)
# analyze_attack_res(t=101, res_path=res_path, true_keys=true_key)

# attack 3
# for th = 1325, dc = 5272，attack 10 round peck32/64, 2**16 keys, \beta _n = 2**(-16)
# analyze_attack_res(t=1325, res_path=res_path, true_keys=true_key)

# attack 4
# for th = 13064, dc = 25680，attack 10 round peck32/64, 2**16 keys, \beta _n = 2**(-16)
# analyze_attack_res(t=13064, res_path=res_path, true_keys=true_key)
