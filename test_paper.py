# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:34:09 2024

@author: romix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300


# d1 = np.linspace(5, 0, 100)
# d2 = np.linspace(1, 5, 100)

# d = np.hstack([d1, d2])

M_s = 1.9885*1e30
m_p = M_s * 0.001
G = 6.67 * 1e-11
T = 10 * (24*60*60)
a = (G*(M_s+m_p)*T**2/(4*np.pi**2))**(1/3)
e = 0
inc = 0

time = np.linspace(0, 20, 1000)*60*60

n = 2*np.pi / T
t_c = 10*60*60


r_s = 1
r_p = 0.01

# Z = d / r_s
Z = a/(696000000*r_s)*np.sqrt((np.sin(n*(time-t_c)))
                              ** 2+(np.sin(n*(time-t_c)))**2)
p = r_p / r_s

l = np.zeros(len(Z))


def k1(p, z):
    return np.arccos((1-p**2+z**2)/(2*z))


def k0(p, z):
    return np.arccos((p**2+z**2-1)/(2*p*z))


# for i, z in enumerate(Z):

#     if 1+p < z:
#         l[i] = 0
#     elif abs(1-p) < z <= 1+p:
#         sqrt = (4*z**2-(1+z**2-p**2)**2)/4
#         l[i] = 1/np.pi*(p**2*k0(p, z)+k1(p, z)-np.sqrt(sqrt))
#     elif z <= 1-p:
#         l[i] = p**2
#     elif z <= p-1:
#         l[i] = 1
# F = 1 - l

# b = a*np.cos(inc)/(696000000*r_s)
b = 0
tau0 = (696000000*r_s)/(a*n)

# delta = p**2
u = 0.5
delta = p**2 * (9-8*(np.sqrt(1-b**2)-1)*u)/(9-8*u)
TT = 2*tau0*np.sqrt(1-b**2)
tau = 2*tau0*p/np.sqrt(1-b**2)

F = np.zeros(len(time))
for i, t in enumerate(time):
    if abs(t-t_c) <= TT/2-tau/2:
        F[i] = 1 - delta
    elif TT/2-tau/2 < abs(t-t_c) < TT/2+tau/2:
        F[i] = 1-delta+(delta/tau)*(abs(t-t_c)-TT/2+tau/2)
    elif abs(t-t_c) >= TT/2+tau/2:
        F[i] = 1

plt.plot(time/60/60, F)
plt.show()
