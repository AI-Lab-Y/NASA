import numpy as np


# view 11-round attack res
def analyze_attack_res(attack_time, consumption_num, is_success, is_valid, survive_num):
    average_surviving_num = np.mean(survive_num, axis=0)
    print('attack success time: {}'.format(np.sum(is_success)))
    print('get valid plaintext structure time: {}'.format(np.sum(is_valid)))
    print('average surviving key in stage1: {}'.format(average_surviving_num[0]))
    print('average surviving key in stage2: {}'.format(average_surviving_num[1]))
    print('average surviving key in stage3: {}'.format(average_surviving_num[2]))
    print('average surviving key in stage4: {}'.format(average_surviving_num[3]))
    print('average attack time: {}s'.format(np.mean(attack_time)))
    print('average plaintext structure consumption: {}'.format(np.mean(consumption_num)))


attack_time = np.load('./key_recovery_record/bayesian/3_7_6_22586_5271_5228_829/attack_time.npy')
consumption_num = np.load('./key_recovery_record/bayesian/3_7_6_22586_5271_5228_829/consumption_num.npy')
is_success = np.load('./key_recovery_record/bayesian/3_7_6_22586_5271_5228_829/is_success.npy')
is_valid = np.load('./key_recovery_record/bayesian/3_7_6_22586_5271_5228_829/is_valid.npy')
survive_num = np.load('./key_recovery_record/bayesian/3_7_6_22586_5271_5228_829/survive_num.npy')
analyze_attack_res(attack_time, consumption_num, is_success, is_valid, survive_num)

# if you wanna see which keys survive the attack,
# open './key_recovery_record/bayesian/3_7_6_22586_5271_5228_829/attack_record.txt'
