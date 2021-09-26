import numpy as np
import scipy.stats as st


def cal_dc_for_searching_right_structure(p0=1, p2=0.5, p3=0.5, bp=0.005, bn=0.005):
    z_1_bp = st.norm.ppf(1 - bp)
    z_1_bn = st.norm.ppf(1 - bn)
    mu_p = p0 * p2 + (1 - p0) * p3
    mu_n = p3
    sig_p = np.sqrt(mu_p * (1 - mu_p))
    sig_n = np.sqrt(mu_n * (1 - mu_n))
    # print('z_1_bp is ', z_1_bp， ' mu_p is ', mu_p, ' sig_p is  ', sig_p)
    # print('z_1_bn is ', z_1_bn， ' mu_n is ', mu_n, ' sig_n is  ', sig_n)
    x = z_1_bp * sig_p + z_1_bn * sig_n
    y = np.abs(mu_p - mu_n)

    N = (x / y) * (x / y)
    dc = np.log2(N)
    print('the weight of data complexity is ', dc)

    # calculate the decision threshold t
    sig = sig_p * np.sqrt(N)
    mu = mu_p * N
    t = mu - sig * z_1_bp
    print('t is ', t)


def cal_DC(p0=0.5, p1=0.5, p2=0.5, p3=0.5, bp=0.005, bn=0.005):
    z_1_bp = st.norm.ppf(1 - bp)
    z_1_bn = st.norm.ppf(1 - bn)
    mu_p = p0 * p1 + (1 - p0) * p3
    mu_n = p0 * p2 + (1 - p0) * p3
    sig_p = np.sqrt(p0 * p1 * (1 - p1) + (1 - p0) * p3 * (1 - p3))
    sig_n = np.sqrt(p0 * p2 * (1 - p2) + (1 - p0) * p3 * (1 - p3))
    x = z_1_bp * sig_p + z_1_bn * sig_n
    y = np.abs(mu_p - mu_n)

    N = (x / y) * (x / y)
    dc = np.log2(N)
    print('the weight of data complexity is ', dc)

    # calculate the decision threshold t
    sig = sig_p * np.sqrt(N)
    mu = mu_p * N
    t = mu - sig * z_1_bp
    print('t is ', t)


# c3 = 0.5, <= 1 bits,    14 bits  21 ~ 8
print('1 + 7-student')      # attack 9-round Speck96/96
cal_DC(p0=2**(0), p1=0.771912, p2=0.450077, p3=0.249348, bp=0.005, bn=2**(-14))
# dc = 2**(6.433610) = 86, dc = 56

# c3 = 0.5, <= 1 bits,    14 bits  21 ~ 8
print('2 + 7-student')      # attack 10-round Speck96/96
cal_DC(p0=2**(-2), p1=0.771912, p2=0.450077, p3=0.249348, bp=0.005, bn=2**(-14))
# dc = 2**(10.276820) = 1240, dc = 432

# c3 = 0.5, <= 1 bits,    14 bits  21 ~ 8
print('3 + 7-student')      # attack 11-round Speck96/96
cal_DC(p0=2**(-6), p1=0.771912, p2=0.450077, p3=0.249348, bp=0.005, bn=2**(-14))
# dc = 2**(18.221606) = 305667, t = 78097

# c3 = 0.5, <= 1 bits,    14 bits  21 ~ 8
print('4 + 7-student')      # attack 12-round Speck96/96
cal_DC(p0=2**(-12), p1=0.771912, p2=0.450077, p3=0.249348, bp=0.005, bn=2**(-14))
# dc = 2**(30.217861) = , t = 311498458

# c3 = 0.5, <= 1 bits,    14 bits  21 ~ 8
print('5 + 7-student')      # attack 13-round Speck96/96
cal_DC(p0=2**(-22), p1=0.771912, p2=0.450077, p3=0.249348, bp=0.005, bn=2**(-14))
# dc = 2**(50.217802) = , t = 326490697722159

# c3 = 0.5, <= 1 bits,    14 bits  21 ~ 8
print('6 + 7-student')      # attack 14-round Speck96/96
cal_DC(p0=2**(-32), p1=0.771912, p2=0.450077, p3=0.249348, bp=0.005, bn=2**(-14))
# dc = 2**(70.217802) = , t = 3.4235033384896135e+20
