import student_net as tn_s
import speck as sp


sp.check_testvector()
selected_bits = [21,20,19,18,17,16,15,14,13,12,11,10,9,8]
teacher = './saved_model/teacher/0x80-0x0/7_distinguisher.h5'
model_folder = './saved_model/student/0x80-0x0/'
tn_s.train_speck_distinguisher(10, num_rounds=7, depth=1, diff=(0x80, 0x0), bits=selected_bits, teacher=teacher, folder=model_folder)
