import numpy as np

bits_num = 6

def hw(v):
  res = np.zeros(v.shape, dtype=np.uint8)
  for i in range(16):
    res = res + ((v >> i) & 1)

  return(res)


low_weight = np.array(range(2**bits_num), dtype=np.uint16)
low_weight = hw(low_weight)
# print('low weight is ', low_weight)
num = [np.sum(low_weight == i) for i in range(bits_num+1)]


def cal_target_cnt(t, res_path, true_keys, time_cost):
    res = np.load(res_path)
    print('res shape is : ', np.shape(res))
    # true_keys = true_keys.astype(np.uint16)
    print('true keys shape is : ', np.shape(true_keys))
    # print(true_keys)

    true_key_cnt = np.zeros(true_keys.shape[0])
    for i in range(true_keys.shape[0]):
        true_key_cnt[i] = res[i][true_keys[i]]
    print('true key cnt is ', true_key_cnt)
    print('min value is ', min(true_key_cnt))
    print('max value is ', max(true_key_cnt))

    key_guess_num = np.sum(res > t, axis=1)
    print('key guess num shape is ', np.shape(key_guess_num))
    print(key_guess_num)

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

    print('average time cost is {}s'.format(np.mean(time_cost)))

# 6-round attack for Sbox 5
print('result for 6-round attack for Sbox 5:')
folder = './key_recovery_record/0_5_634_351/'
res_path = folder + 'cnt.npy'
true_keys = np.load(folder + 'true_key.npy')
time_cost = np.load(folder + 'time_cost.npy')
cal_target_cnt(351, res_path, true_keys, time_cost)