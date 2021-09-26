import numpy as np

# if you want to see p1, p2, p3 together, open txt file. p3 isn't included in following npy file
ND5_t_p2_d1_path = './p2_estimation_res/teacher/0x0040-0x0/5/5_0.55_p2_d1.npy'
ND6_t_p2_d1_path = './p2_estimation_res/teacher/0x0040-0x0/6/6_0.55_p2_d1.npy'
ND7_t_p2_d1_path = './p2_estimation_res/teacher/0x0040-0x0/7/7_0.55_p2_d1.npy'
ND7_t_p2_d1_d2_path = './p2_estimation_res/teacher/0x0040-0x0/7/7_0.55_p2_d1_d2.npy'
# ND8_t_p2_d1_path = './p2_estimation_res/teacher/0x0040-0x0/8/8_0.5_p2_d1.txt'

# student distinguisher
ND5_s_p2_d1_path = './p2_estimation_res/student/0x0040-0x0/5/5_p2_d1.npy'
ND6_s_p2_d1_path = './p2_estimation_res/student/0x0040-0x0/6/6_p2_d1.npy'
ND7_s_p2_d1_path = './p2_estimation_res/student/0x0040-0x0/7/7_p2_d1.npy'


def view_selected_file(path='./'):
    res = np.load(path)
    num = len(res)
    # print(res)
    for i in range(num):
        print('cur i is ', i)
        print('cur p2_d1 is ', res[i])


view_selected_file(path=ND6_t_p2_d1_path)



