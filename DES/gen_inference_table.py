import des
import numpy as np
from keras.models import load_model


'''
This script can be used to generate inference table
of a certain net with input shape (*, input_size).
'''
input_size = 2 * 2 * 4


# net_path: the path of the net
# save_path: the path to save the corresponding inference table
def gen_student_inference_table(net_path, save_path):
    net = load_model(net_path)
    x = np.array(range(2**input_size), dtype=np.uint64)
    x = des.uint_to_array([x], bit_len=input_size)[0]
    y = net.predict(x, batch_size=10000).flatten()
    np.save(save_path, y)


if __name__ == "__main__":
    delta_S = "0x19600000-0x0"
    nr = 5

    s_box = 5
    net_path = "./saved_model/student/{}/student_{}_box{}_distinguisher.h5".format(delta_S, nr, s_box)
    save_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, nr, s_box)
    gen_student_inference_table(net_path=net_path, save_path=save_path)

    s_box = 8
    net_path = "./saved_model/student/{}/student_{}_box{}_distinguisher.h5".format(delta_S, nr, s_box)
    save_path = "./student_inference_table/{}/student_{}_box{}_distinguisher.npy".format(delta_S, nr, s_box)
    gen_student_inference_table(net_path=net_path, save_path=save_path)
