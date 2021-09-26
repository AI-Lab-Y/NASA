import student_net as tn_h
import des

delta_S = '0x19600000-0x0'

# Sbox = 5
# tn_h.train_DES_distinguisher(10, num_rounds=5, depth=1, diff=(0x19600000, 0x0),
#                              bits=des.Sbox_output[Sbox], Sbox=Sbox, delta_S=delta_S)

Sbox = 8
tn_h.train_DES_distinguisher(10, num_rounds=5, depth=1, diff=(0x19600000, 0x0),
                             bits=des.Sbox_output[Sbox], Sbox=Sbox, delta_S=delta_S)

