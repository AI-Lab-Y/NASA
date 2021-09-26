import teacher_net as tn
import speck as sp


sp.check_testvector()
model_folder = './saved_model/teacher/0x80-0x0/'
tn.train_speck_distinguisher(10, num_rounds=7, depth=2, diff=(0x80, 0x0), folder=model_folder)

