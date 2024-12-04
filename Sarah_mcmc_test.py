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
a = a / R_s  # Convert semi-major axis to stellar radii

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

plt.plot(t, flux)
plt.title('Transit')
plt.xlabel("Time from central transit (hours)")
plt.ylabel("Relative flux")
plt.ylim((0.988, 1.001))

plt.show()

#m = batman.TransitModel(params, t, supersample_factor = 7, exp_time = 0.001)

# %%
lc_observed = np.loadtxt('lightcurve.txt')
lc_observed = lc_observed[0]

t = np.loadtxt('times.txt')
t = t[0]/24

plt.plot(t, lc_observed)  # mid transit point in hours (at 10/24 hours)
plt.show()

# %%


def model_lc(pars, t, ld_type):
    pars = np.array(pars, dtype=float)
    params = bm.TransitParams()  # object to store transit parameters
    params.t0 = pars[0]              # mid transit point
    params.per = pars[1]  # orbital period
    params.rp = pars[2]  # planet radius (in units of stellar radii)
    params.a = pars[3]  # semi-major axis (in units of stellar radii)
    params.inc = pars[4]  # orbital inclination (in degrees)
    params.ecc = pars[5]  # eccentricity
    params.w = pars[6]  # longitude of periastron (in degrees)
    params.u = pars[7:9]  # ld_coeffs        #limb darkening coefficients
    params.limb_dark = ld_type  # limb darkening model
    model = bm.TransitModel(params, t)  # initializes model
    # params_array =
    light_curve = model.light_curve(params)  # calculates light curve

    return light_curve

# this function computes the gradient of the model w.r.t. the parameters and is used to determine how changes in the parameters affect the predicted spectrum


# f' = [f(x+dx)-f(x-dx)]/2dx, dx = step, x = pars
def num_deriv(model, pars, dx, t, ld_type):
    grad = np.zeros((len(lc_observed), len(pars)))

    for i in range(len(pars)):
        plus_step = pars.copy()
        minus_step = pars.copy()
        plus_step[i] += dx[i]
        minus_step[i] -= dx[i]
        f_plus = model(plus_step, t, ld_type)[:len(lc_observed)]  # f(x+dx)
        f_minus = model(minus_step, t, ld_type)[:len(lc_observed)]  # f(x-dx)
        grad[:, i] = (f_plus - f_minus) / (2 * dx[i])
    return grad


# starting guess [mid-transit point, period (days), r_planet (stellar radii),semi-major axis (stellar radii), inclination (degrees),
pars = np.asarray([10/24, 10, .1, 19.5, 90, 0, 90, 0.05, 0])
# eccentricity, longitude of periastron (degrees)]
ld_type = 'quadratic'
t = np.loadtxt('times.txt')
t = t[0]/24
dx = pars*1e-3 + np.array([0, 0, 0, 0, 0, 0.001, 0, 0, 0.001])

# starry results --> lc_observed; equivalent of true data  ("spec" in the hw)
lc_observed = np.loadtxt('lightcurve.txt')
lc_observed = lc_observed[0]

sigma = 1.6e-4
errs = sigma * np.random.randn(len(t))
Ninv = np.diag(1/errs**2)

for j in range(7):  # trial and error incrementing
    model_pred = model_lc(pars, t, ld_type)  # predicted spectrum
    resid_ar = lc_observed - model_pred  # residuals array
    err = (resid_ar**2).sum()
    resid = np.matrix(resid_ar).transpose()  # residuals matrix
    grad = num_deriv(model_lc, pars, dx, t, ld_type)
    grad = np.matrix(grad)
    lhs = grad.transpose()@Ninv@grad  # (A_T)(N_inv)(A) # THIS BECOMES A SINGULAR MATRIX
    cond = np.linalg.cond(lhs)
    print(f"Condition number of lhs: {cond}")
    # assert(1 == 0)
    rhs = grad.transpose()@Ninv@resid
    dp = np.linalg.inv(lhs)@rhs

    for jj in range(len(pars)):
        pars[jj] = pars[jj]+dp[jj]

#    chisq=np.sum((resid_ar/errs)**2) #measuring the fit of the model
        # relative to the uncertainties in the data
# print(chisq)

# %%

# Define the Batman model light curve


def model_lc(pars, t, ld_type):
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
    params_array = np.array([pars[1], pars[2]])

    model = bm.TransitModel(params, t)  # initializes model
    light_curve = model.light_curve(params)  # calculates light curve

    return light_curve


# Likelihood function (assume Gaussian)
def likelihood(observed, predicted, sigma):
    likelihood = -0.5 * np.sum(((observed - predicted) / sigma) ** 2)
    return likelihood

# Proposal distribution for Metropolis-Hastings algorithm (Gaussian)


def proposal_dist(pars, step_size):
    return np.random.normal(loc=pars, scale=step_size)

# Metropolis-Hastings Sampler


def MH_sampler(num_iter, step_size, pars, t, lc_observed, ld_type, sigma):
    """
    Perform Metropolis-Hastings sampling with Batman model fitting.

    Parameters:
        num_iter (int): Number of iterations.
        step_size (float): Step size for the proposal distribution.
        guess (np.ndarray): Initial guess for the parameters.
        t (np.ndarray): Time values for the light curve.
        lc_observed (np.ndarray): Observed light curve data.
        ld_type (str): Limb darkening model type.
        sigma (float): Standard deviation for the Gaussian likelihood.
    """
    initial = pars
    samples = []
    accepted_samples = 0
    best_likelihood = -np.inf  # Initialize as negative infinity
    best_params = None

    for i in range(num_iter):
        final = proposal_dist(initial, step_size)  # proposed parameters

        # compute likelihood of batman model
        current_model = model_lc(initial, t, ld_type)
        current_likelihood = likelihood(lc_observed, current_model, sigma)
        # compute likelihood of new version of the Batman model that uses the new proposed parameter values
        model_with_proposed_params = model_lc(final, t, ld_type)
        proposed_likelihood = likelihood(
            lc_observed, model_with_proposed_params, sigma)

        # Calculate acceptance ratio
        acceptance_ratio = min(
            1, np.exp(proposed_likelihood - current_likelihood))

        # Decide whether to accept or reject the proposal
        u = np.random.uniform(0, 1)
        if u < acceptance_ratio:
            initial = final
            accepted_samples += 1

        samples.append(initial)

        # Update best parameters if we found a better likelihood
        if proposed_likelihood > best_likelihood:
            best_likelihood = proposed_likelihood
            best_params = final  # Update best parameters with the final proposal

    acceptance_rate = accepted_samples / num_iter

    return np.array(samples), acceptance_rate, best_params

# main code...


# load data
t = np.loadtxt('times.txt')
t = t[0] / 24
lc_observed = np.loadtxt('lightcurve.txt')
lc_observed = lc_observed[0]

# set initial parameters
pars = np.asarray([10/24, 10, 0.1, 19.5, 90, 0, 90, 0.05, 0])  # input
#pars_fit = np.asarray([pars[1],pars[2]])

ld_type = 'quadratic'
sigma = 1.6e-4

# MCMC parameters
num_iter = 100000  # Number of iterations
step_size = 1e-4  # Proposal step size

# Run Metropolis-Hastings sampler
samples, acc_rate, best_params = MH_sampler(
    num_iter, step_size, pars, t, lc_observed, ld_type, sigma)

# Print acceptance rate
print(f'Acceptance Rate: {acc_rate:.2%}')

print(best_params)

# %%

fig, axs = plt.subplots(len(pars), 1, figsize=(20, 40))

for i in range(len(pars)):
    axs[i].plot(samples[:, i])
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('value')
