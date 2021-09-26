import numpy as np

def view_selected_file(path='./'):
    res = np.load(path)
    print(res)

ND5_box5_p2_d1_path = './p2_estimation_res/student/0x19600000-0x0/5_box5_0.5_p2_d1.npy'
ND5_box8_p2_d1_path = './p2_estimation_res/student/0x19600000-0x0/5_box8_0.5_p2_d1.npy'

def view_selected_file(path='./'):
    res = np.load(path)
    print(res)


print('p2_d1 for ND5_box5:')
view_selected_file(path=ND5_box5_p2_d1_path)
print('p2_d1 for ND5_box8:')
view_selected_file(path=ND5_box8_p2_d1_path)