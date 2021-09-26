import teacher_net as tn
import speck as sp


sp.check_testvector()
model_folder = './saved_model/teacher/0x0040-0x0/'
# tn.train_speck_distinguisher(10, num_rounds=7, depth=1, diff=(0x0040, 0x0), folder=model_folder)
# tn.train_speck_distinguisher(10, num_rounds=6, depth=1, diff=(0x0040, 0x0), folder=model_folder)
tn.train_speck_distinguisher(10, num_rounds=5, depth=1, diff=(0x0040, 0x0), folder=model_folder)
# model_folder = './saved_model/teacher/0x8000-0x840a/'
# tn.train_speck_distinguisher(10, num_rounds=5, depth=1, diff=(0x8000, 0x840a), folder=model_folder)
