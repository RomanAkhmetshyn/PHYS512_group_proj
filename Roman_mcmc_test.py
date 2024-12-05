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

global inc
global ecc
global w

global P
P = 10


inc = 90
ecc = 0.0
w = 90

np.random.seed(42)


def get_transit(pars, t):
    pars = np.array(pars, dtype=float)
    params = bm.TransitParams()  # object to store transit parameters
    params.t0 = pars[0]  # mid-transit point
    params.per = P  # orbital period
    params.rp = pars[1]  # planet radius (in units of stellar radii)
    params.a = pars[2]  # semi-major axis (in units of stellar radii)
    params.inc = inc  # orbital inclination (in degrees)
    params.ecc = ecc  # eccentricity
    params.w = w  # longitude of periastron (in degrees)
    params.u = pars[3:5]  # limb darkening coefficients
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

# def run_chain(pars, fun, data, t, Ninv, L, nsamp=100, adapt_steps=10, target_acceptance=0.45):
#     chisq = np.zeros(nsamp)
#     npar = len(pars)
#     chain = np.zeros([nsamp, npar])
#     chain[0, :] = pars
#     chisq[0] = fun(pars, data, t, Ninv)
#     acceptance_count = 0  # Track acceptance count
#     step_size = L  # Initialize step size

#     for i in trange(1, nsamp):
#         pnew = chain[i - 1, :] + step_size * np.random.randn(npar)
#         chi_new = fun(pnew, data, t, Ninv)
#         prob = np.exp(0.5 * (chisq[i - 1] - chi_new))

#         if np.random.rand() < prob:  # Accept step
#             chain[i, :] = pnew
#             chisq[i] = chi_new
#             acceptance_count += 1
#         else:  # Reject step
#             chain[i, :] = chain[i - 1, :]
#             chisq[i] = chisq[i - 1]

#         # Adjust step size every `adapt_steps` iterations
#         if i % adapt_steps == 0:
#             acceptance_rate = acceptance_count / adapt_steps
#             # print(acceptance_rate)
#             if acceptance_rate < target_acceptance:
#                 step_size *= 0.5  # Decrease step size
#             elif acceptance_rate > target_acceptance:
#                 step_size *= 1.5  # Increase step size
#             acceptance_count = 0  # Reset acceptance count

#     return chain, chisq


t = np.loadtxt('times.txt')
t = t[0] / 24
lc_observed = np.loadtxt('lightcurve.txt')
flux = lc_observed[0]
errs = np.zeros(len(flux))+1.6e-3

# set initial parameters
titles = ['T0', 'Rp/Rs',
          'a/Rs',  'u1', 'u2']
real = np.asarray([10/24,  0.1, 19.5177,  0.4, 0.1])
pguess = np.asarray([0.4,  0.08, 10,  0.0, 0.0])  # input

# L = pguess*[1e-2, 1e-2, 1e-2, 1e-3, 1e-3]
L = pguess*1e-3
L[-2] = 0.001
L[-1] = 0.001
# L = np.array([0.005, 0.05, 0.005, 0.005, 0.05, 0.001, 0.05, 0.005, 0.005])*1e-1


Ninv = np.diag(1/errs**2)

# %%
file = False
nsamp = 40000
burnin = 10000
if file:
    chain = np.genfromtxt('chains.txt')
    chainvec = np.genfromtxt('chis.txt')

else:

    chain, chivec = run_chain(pguess, chisq, flux, t, Ninv,
                              L=L, nsamp=nsamp)
    chain = chain[burnin:, :]
    chivec = chivec[burnin:]
    np.savetxt('chains.txt', chain)
    np.savetxt('chis.txt', chivec)

# %%


steps = np.arange(0, nsamp)
steps = steps[burnin:]
# Number of parameters
n_params = len(titles)

# Create a figure and subplots stacked vertically
fig, axes = plt.subplots(n_params, 1, figsize=(8, 2 * n_params), sharex=True)

# Plot each parameter chain on its corresponding subplot
for i, ax in enumerate(axes):
    ax.plot(steps, chain[:, i])  # Plot for the i-th parameter
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
fig = corner.corner(chain, labels=titles, truths=np.median(
    chain, axis=0), quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f', )

# Show the plot
plt.show()


# %%

plt.errorbar(t, flux, errs, fmt='.', c='k', markersize=1)
plt.plot(t, get_transit(np.mean(chain, axis=0), t))
plt.show()
