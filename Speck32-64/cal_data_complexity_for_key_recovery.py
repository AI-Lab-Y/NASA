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


# when verify NASA, we let p1, p2, p3 keep many decimal places
# to make the estimated data complexity as precise as possible
# If you do not want to keep so many decimal places,
# we advise that let p1 be smaller, let p2 be larger, and let p3 be larger.
# This is based on their impacts on the data complexity.
# We give an example as follows.

# for attack 1, 2, 3, 4
print('for attack 1, 2, 3, 4 ')
# for 9 round Speck32/64, teacher, c3 = 0.55, <= 2 bits
print('3 + 5_teacher')
cal_DC(p0=2**(-6), p1=0.8889885, p2=0.2167858, p3=0.0383838, bp=0.005, bn=2**(-16))
# dc = 2**(13.957141) = 15905, tc = 758

# for 9 round Speck32/64, teacher, c3 = 0.55, <= 2 bits
print('2 + 6_teacher')
cal_DC(p0=2**(-2), p1=0.6784635, p2=0.2393143, p3=0.1161114, bp=0.005, bn=2**(-16))
# dc = 2**(8.892501) = 475,  tc = 101

# for 10 round Speck32/64, teacher, c3 = 0.55, <= 2 bits
print('2 + 7_teacher')
cal_DC(p0=2**(-2), p1=0.4183026, p2=0.260615, p3=0.2162363, bp=0.005, bn=2**(-16))
# dc = 2**(12.364105) = 5272, t = 1325

# for 10 round speck32/64, teacher, c3 = 0.5, <= 2 bits
print('1 + 8_teacher')
cal_DC(p0=2**0, p1=0.5183542, p2=0.4957098, p3=0.4913596, bp=0.001, bn=2**(-16))
# dc = 2**(14.648375) = 25680, t = 13064

print(' ')

# for attack 1, 2, 3 with reduced key space, student distinguishers
print('for attack 1, 2, 3 with reduced key space ')
# for 9 round Speck32/64, student, c3 = 0.55, <= 2 bits
print('3 + 5_student')
cal_DC(p0=2**(-6), p1=0.7192415, p2=0.356625, p3=0.129078, bp=0.005, bn=2**(-8))
# dc = 2**(16.571374) = 97382, tc = 13196

# for 9 round Speck32/64, student, c3 = 0.55, <= 2 bits
print('2 + 6_student')
cal_DC(p0=2**(-2), p1=0.5131719, p2=0.3025028, p3=0.2603019, bp=0.005, bn=2**(-8))
# dc = 2**(10.962513) = 1995,  tc = 593

# for 10 round Speck32/64, student, c3 = 0.55, <= 2 bits
print('2 + 7_student')
cal_DC(p0=2**(-2), p1=0.3575058, p2=0.2939745, p3=0.2862612, bp=0.005, bn=2**(-8))
# dc = 2**(14.463191) = 22586, t = 6690

print(' ')

# for 11 round Speck32/64, ND7s, ND7t, ND6s, ND6t, c3 = 0.55
print('for attack on 11-round Speck32/64 with reduced data complexity')

# stage 1, ND7s, c3 = 0.55, d1 = 1
print('search right structure, 2 + 7_student')
cal_dc_for_searching_right_structure(p0=2**(-2), p2=0.323207, p3=0.2864982, bp=0.1, bn=2**(-8))
# dc = 2**(15.211368) = 37938, t = 11103

# stage 2, ND7s, c3 = 0.55, <= 2 bits
print('2 + 7_student')
cal_DC(p0=2**(-2), p1=0.3575058, p2=0.2939745, p3=0.2862612, bp=0.005, bn=2**(-8))
# dc = 2**(14.463191) = 22586, t = 6690

# stage 3, ND7t, c3 = 0.55, <= 2 bits
print('2 + 7_teacher')
cal_DC(p0=2**(-2), p1=0.4183026, p2=0.260615, p3=0.2162363, bp=0.005, bn=2**(-16))
# dc = 2**(12.364105) = 5271, t = 1325

# stage 4, ND6s, c3 = 0.55, <= 1 bit
print('2 + 6_student')
cal_DC(p0=2**(-2), p1=0.5131719, p2=0.3402422, p3=0.2603019, bp=0.001, bn=2**(-14))
# dc = 2**(12.352067) = 5228,  tc = 1589

# stage 5, ND6t, c3 = 0.55, <= 1 bit
print('2 + 6_teacher')
cal_DC(p0=2**(-2), p1=0.6784635, p2=0.3134763, p3=0.1161114, bp=0.001, bn=2**(-16))
# dc = 2**(9.696623) = 829,  tc = 180

print(' ')
print(' ')
print(' ')

print('for attack on 12 round speck32/64 with reduced data complexity')
# teacher, c3 = 0.5, <= 2 bits
print('2 + 8, teacher')
cal_DC(p0=2**(-2), p1=0.5183955, p2=0.4957098, p3=0.4913596, bp=0.005, bn=2**(-16))
# dc = 2**(18.431427) = 353518, t = 175328

# stage 1, ND7s, c3 = 0.55, d1 = 1
print('search right structure, 2 + 7_student')
cal_dc_for_searching_right_structure(p0=2**(-2), p2=0.323207, p3=0.2864982, bp=0.1, bn=2**(-8))
# dc = 2**(15.211368) = 37938, t = 11103

# stage 2, ND7s, c3 = 0.55, <= 2 bits
print('2 + 7_student')
cal_DC(p0=2**(-2), p1=0.3575058, p2=0.2939745, p3=0.2862612, bp=0.005, bn=2**(-8))
# dc = 2**(14.463191) = 22586, t = 6690




