import batman as bm
from matplotlib import rc
from pylab import *
import batman
from batman import TransitModel, TransitParams
from batman.openmp import detect
import timeit
import numpy as np
import matplotlib.pyplot as plt

from batman import _nonlinear_ld
from batman import _quadratic_ld
from batman import _uniform_ld
from batman import _logarithmic_ld
from batman import _exponential_ld
from batman import _power2_ld
from batman import _custom_ld
from batman import _rsky
from batman import _eclipse
from math import pi
import multiprocessing
from batman import openmp
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
matplotlib.rcParams.update({'font.size': 14})

M_s = 1.9885*1e30
m_p = M_s * 0.001
G = 6.67 * 1e-11
T = 10 * (24*60*60)
a = (G*(M_s+m_p)*T**2/(4*np.pi**2))**(1/3)
R_s = 6.96e8  # Stellar radius in meters
# a = a / R_s  # Convert semi-major axis to stellar radii
print(a)
params = batman.TransitParams()  # object to store transit parameters
params.t0 = 10/24.             # mid transit point
params.per = 10.  # orbital period
params.rp = 0.1  # planet radius (in units of stellar radii)
params.a = a  # semi-major axis (in units of stellar radii)
params.inc = 90  # orbital inclination (in degrees)
params.ecc = 0.  # eccentricity
params.w = 90.  # longitude of periastron (in degrees)
params.limb_dark = "quadratic"  # limb darkening model
params.u = [.05, 0]  # limb darkening coefficients

# t = np.linspace(-0.025, 0.025, 1000)   #times at which to calculate light curve
t = np.loadtxt('times.txt')
t = t[0]/24
#t = np.linspace(-.25, .25, 1000)

m = batman.TransitModel(params, t)  # initializes model

flux = m.light_curve(params)  # calculates light curve

plt.plot(t, flux, zorder=5)
plt.title('Transit')
plt.xlabel("Time from central transit (hours)")
plt.ylabel("Relative flux")
# plt.ylim((0.988, 1.001))


#m = batman.TransitModel(params, t, supersample_factor = 7, exp_time = 0.001)

# %%
lc_observed = np.loadtxt('lightcurve.txt')
lc_observed = lc_observed[0]

t = np.loadtxt('times.txt')
t = t[0]/24

# mid transit point in hours (at 10/24 hours)
plt.plot(t, lc_observed, zorder=1)
plt.show()
