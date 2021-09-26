import numpy as np

block_size = 32


def view_bits_sensitivity(folder='./bits_sensitivity_res/', nr=5):
    path = folder + str(nr) + '_distinguisher_bit_sensitivity.npy'
    res = np.load(path)
    print('cur round is ', nr)
    print('initial acc is ', res[block_size])
    acc = res[block_size]
    print('id       bit sensitivity:')
    for i in range(block_size):
        print(i, '  ', acc - res[i])


res_folder = './bits_sensitivity_res/0x0040-0x0/sen0/'
view_bits_sensitivity(folder=res_folder, nr=5)