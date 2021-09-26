import student_net as tn
import speck as sp


sp.check_testvector()
selected_bits = [14,13,12,11,10,9,8,7]
teacher = './saved_model/teacher/0x0040-0x0/5_distinguisher.h5'
# teacher = './saved_model/teacher/0x0040-0x0/net8_small.h5'
model_folder = './saved_model/student/0x0040-0x0/'
tn.train_speck_distinguisher(10, num_rounds=5, depth=1, diff=(0x0040, 0), bits=selected_bits, teacher=teacher, folder=model_folder)
# tn_s.train_speck_distinguisher(10, num_rounds=7, depth=1, diff=(0x0040, 0), bits=selected_bits, teacher=teacher, folder=model_folder)
