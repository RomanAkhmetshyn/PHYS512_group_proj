import corner
import batman as bm
import numpy as np
from matplotlib import pyplot as plt
import time
import matplotlib as mpl
from tqdm import trange
mpl.rcParams['figure.dpi'] = 300

global ld_type
ld_type = 'quadratic'


def get_transit(pars, t):
    pars = np.array(pars, dtype=float)
    params = bm.TransitParams()  # object to store transit parameters
    params.t0 = pars[0]  # mid-transit point
    params.per = pars[1]  # orbital period
    params.rp = pars[2]  # planet radius (in units of stellar radii)
    params.a = pars[3]  # semi-major axis (in units of stellar radii)
    params.inc = pars[4]  # orbital inclination (in degrees)
    params.ecc = pars[5]  # eccentricity
    params.w = pars[6]  # longitude of periastron (in degrees)
    params.u = pars[7:9]  # limb darkening coefficients
    params.limb_dark = ld_type  # limb darkening model

    model = bm.TransitModel(params, t)  # initializes model
    light_curve = model.light_curve(params)  # calculates light curve

    return light_curve


def chisq(pars, data, t, Ninv):
    y = get_transit(pars, t)
    r = data-y
    chisq = r@Ninv@r
    return chisq


def run_chain(pars, fun, data, t, Ninv, L, nsamp=100):
    chisq = np.zeros(nsamp)
    npar = len(pars)
    chain = np.zeros([nsamp, npar])
    chain[0, :] = pars
    chisq[0] = fun(pars, data, t, Ninv)
    for i in trange(1, nsamp):
        # pnew = chain[i-1, :]+L@np.random.randn(npar)
        pnew = chain[i-1, :]+L*np.random.randn(npar)
        chi_new = fun(pnew, data, t, Ninv)
        prob = np.exp(0.5*(chisq[i-1]-chi_new))
        # accept if a random number is less than this
        if np.random.rand(1)[0] < prob:
            chain[i, :] = pnew
            chisq[i] = chi_new
        else:
            chain[i, :] = chain[i-1, :]
            chisq[i] = chisq[i-1]
    return chain, chisq


t = np.loadtxt('times.txt')
t = t[0] / 24
lc_observed = np.loadtxt('lightcurve.txt')
flux = lc_observed[0]
errs = np.zeros(len(flux))+1.6e-4

# set initial parameters
real = np.asarray([10/24, 10, 0.1, 19.5177, 90, 0.0, 90, 0.05, 0.0])
pguess = np.asarray([10/24, 15, 0.01, 19.5, 90, 0.04, 90, 0.05, 0.01])  # input
L = pguess*1e-2


Ninv = np.diag(1/errs**2)

# %%
file = False
burnin = 30000
if file:
    chain = np.genfromtxt('chains.txt')
    chainvec = np.genfromtxt('chis.txt')

else:

    chain, chivec = run_chain(pguess, chisq, flux, t, Ninv,
                              L=L, nsamp=100000)
    chain = chain[burnin:, :]
    chivec = chivec[burnin:]
    np.savetxt('chains.txt', chain)
    np.savetxt('chis.txt', chivec)

# %%
titles = ['T0', 'P', 'Rp/Rs',
          'a', 'inc', 'e', 'w', 'u1', 'u2']

# Number of parameters
n_params = len(titles)

# Create a figure and subplots stacked vertically
fig, axes = plt.subplots(n_params, 1, figsize=(8, 2 * n_params), sharex=True)

# Plot each parameter chain on its corresponding subplot
for i, ax in enumerate(axes):
    ax.plot(chain[:, i])  # Plot for the i-th parameter
    ax.set_title(titles[i])
    ax.set_ylabel("Parameter Value")

# Label the x-axis only on the bottom plot
axes[-1].set_xlabel("Step")

# Adjust the layout for better spacing
plt.tight_layout()

# Show the figure
plt.show()

print(np.mean(chain, axis=0))

print(real)

# %%


# Create a corner plot
fig = corner.corner(chain, labels=titles, truths=real, show_titles=True)

# Show the plot
plt.show()
