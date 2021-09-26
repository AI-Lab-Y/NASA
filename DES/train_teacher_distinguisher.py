import teacher_net as tn


tn.train_DES_distinguisher(10, num_rounds=5, depth=1, diff=(0x19600000, 0x0), delta_S='0x19600000-0x0/')