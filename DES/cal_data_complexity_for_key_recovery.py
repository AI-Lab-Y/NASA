import numpy as np
import scipy.stats as st


def cal_DC(p0=0.5, p1=0.5, p2=0.5, p3=0.5, bp=0.005, bn=0.005):
    z_1_bp = st.norm.ppf(1 - bp)
    z_1_bn = st.norm.ppf(1 - bn)
    mu_p = p0 * p1 + (1 - p0) * p3
    mu_n = p0 * p2 + (1 - p0) * p3
    sig_p = np.sqrt(p0 * p1 * (1 - p1) + (1 - p0) * p3 * (1 - p3))
    sig_n = np.sqrt(p0 * p2 * (1 - p2) + (1 - p0) * p3 * (1 - p3))
    # print('z_1_bp is ', z_1_bp， ' mu_p is ', mu_p， 'sig_p is ', sig_p)
    # print('z_1_bn is ', z_1_bn， ' mu_n is ', mu_n， 'sig_n is ', sig_n)
    x = z_1_bp * sig_p + z_1_bn * sig_n
    y = np.abs(mu_p - mu_n)

    N = (x / y) * (x / y)
    dc = np.log2(N)
    print('N is ', N)
    print('the weight of data complexity is ', dc)

    # calculate the decision threshold t
    sig = sig_p * np.sqrt(N)
    mu = mu_p * N
    t = mu - sig * z_1_bp
    print('t is ', t)


# attack on 6-round DES for Sbox 5
# data complexity for key recovery
print('N for 6-round attack for Sbox 5')
cal_DC(p0=1, p1=0.6041, p2=0.5113, p3=0.4890, bp=0.005, bn=2**(-6))
# dc = 2**(9.30798) = 634, t = 351

# attack on 8-round DES for Sbox 5
# data complexity for key recovery
print('N for 8-round attack for Sbox 5')
cal_DC(p0=234**(-1), p1=0.6041, p2=0.5113, p3=0.4890, bp=0.005, bn=2**(-6))
# dc = 2**(25.08285) = 35537762, t = 17387770

# attack on 10-round DES for Sbox 5
# data complexity for key recovery
print('N for 10-round attack for Sbox 5')
cal_DC(p0=234**(-2), p1=0.6041, p2=0.5113, p3=0.4890, bp=0.005, bn=2**(-6))
# dc = 2**(40.82372) = 1946099202926, t = 951644804794

# attack on 6-round DES for Sbox 8
print('N for 6-round attack for Sbox 8')
cal_DC(p0=1, p1=0.5799, p2=0.5145, p3=0.4902, bp=0.005, bn=2**(-6))
# dc = 2**(10.33181) = 1289, t = 701

# attack on 8-round DES for Sbox 8
print('N for 8-round attack for Sbox 8')
cal_DC(p0=234**(-1), p1=0.5799, p2=0.5145, p3=0.4902, bp=0.005, bn=2**(-6))
# dc = 2**(26.09272) = 71563482, t = 35096958

# attack on 10-round DES for Sbox 8
print('N for 10-round attack for Sbox 8')
cal_DC(p0=234**(-2), p1=0.5799, p2=0.5145, p3=0.4902, bp=0.005, bn=2**(-6))
# dc = 2**(41.83354) = 3918761966411, t = 1920980986510
