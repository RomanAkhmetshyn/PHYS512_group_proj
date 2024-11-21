import matplotlib.pyplot as plt
import numpy as np
import starry

np.random.seed(77)
starry.config.lazy = False
starry.config.quiet = True

import numpy as np

# Define the number of rows and columns
rows = 1
cols = (5 + 1)**2

# Use numpy functions to initialize the 2D array
ylm = np.zeros((rows, cols))  # Start with all ones
# ylm[:, :] = np.random.normal(loc=0, scale=0.05, size=(rows, cols))

# ylm[:,0] = 1
# plt.plot(ylm)
# plt.show()

# fig, ax = plt.subplots(1, figsize=(12, 5))
# for Ys in ylm:
ydeg = 1

map = starry.Map(ydeg=ydeg)
A_y = np.array(map.y[1:])
B_y = np.array(map.y[1:])
# map.show(colorbar=True, projection='moll')

A = dict(
    ydeg=ydeg,  # degree of the map
    udeg=2,  # degree of the limb darkening
    inc=90.0,  # inclination in degrees
    amp=1.0,  # amplitude (a value prop. to luminosity)
    r=1,  #  radius in R_sun
    m=1,  # mass in M_sun
    prot=10,  # rotational period in days
    u=[0.05, 0.00],  # limb darkening coefficients
    y=A_y,  # the spherical harmonic coefficients
)

B = dict(
    ydeg=ydeg,  # degree of the map
    udeg=2,  # degree of the limb darkening
    # inc=-39,  # inclination in degrees
    inc=90,
    # amp=0.0165,  # amplitude (a value prop. to luminosity)
    amp=0.001,  # amplitude (a value prop. to luminosity)
    r=0.1,  #  radius in R_sun
    m=0.001,  #  mass in M_sun
    porb=10,  # orbital period in days
    # prot=0.1,  # rotational period in days
    prot=9,
    t0=10/24,  # reference time in days (when it transits A)
    u=[0, 0],  # limb darkening coefficients
    y=B_y,  # the spherical harmonic coefficients
)

pri = starry.Primary(
    starry.Map(ydeg=A["ydeg"], udeg=A["udeg"], inc=A["inc"], amp=A["amp"]),
    r=A["r"],
    m=A["m"],
    prot=A["prot"],
)
pri.map[1:] = A["u"]
pri.map[1:, :] = A["y"]
# pri.map.show()

sec = starry.Secondary(
    starry.Map(ydeg=B["ydeg"], udeg=B["udeg"], inc=B["inc"], amp=B["amp"]),
    r=B["r"],
    m=B["m"],
    porb=B["porb"],
    prot=B["prot"],
    t0=B["t0"],
    inc=90.00,
    ecc=0
)
sec.map[1:] = B["u"]
sec.map[1:, :] = B["y"]
# sec.map.show(colorbar=True)

sys = starry.System(pri, sec)

# sys.show(t=np.linspace(0, 0.3, 100), window_pad=3, 
#           file='sys.gif', figsize=(20,16))



#%%

full_flux = []
full_time = []

rots = 10

pred = [10+rot*10*24 for rot in range(0,rots)]

amplitude = 0.5  # Amplitude of TTV in hours
period = 8  # Period of TTV in number of transits
TTV = amplitude * np.sin(2 * np.pi * np.arange(rots) / period)

plt.plot(np.arange(rots), TTV)
plt.show()


pred_with_TTV = pred + TTV * 24  # Convert TTV from hours to the same time unit as pred

# Simulate and plot data
for rot in range(rots):
    # Generate time array for the current rotation
    t = np.linspace(5 + rot * 10 * 24, 15 + rot * 10 * 24, 1000)
    
    # Simulate true flux
    flux_true = sys.flux((t - TTV[rot]) / 24) - 0.001
    
    # Add noise to flux
    sigma = 1.6e-4
    flux = flux_true + sigma * np.random.randn(len(t))
    
    # Append to full_flux and full_time
    full_flux.append(flux)
    full_time.append(t)
    
    # Plot the results for each rotation
    fig, ax = plt.subplots(1, figsize=(12, 5))  
    ax.vlines(pred[rot], 0.989, 1.005, linestyle='--', color='r', label="TTV mid-transit")
    ax.vlines(pred[rot]+TTV[rot], 0.989, 1.005, linestyle='--', color='g', label="TTV mid-transit")
    ax.plot(t, flux, "k.", alpha=0.3, ms=2, label="Observed Flux")
    ax.plot(t, flux_true, lw=1, label="True Flux")
    ax.set_xlabel("Time [hours]", fontsize=16)
    ax.set_ylabel("Normalized Flux", fontsize=16)
    ax.legend()
    plt.show()

# Stack the arrays into one column of values
full_flux = np.hstack(full_flux)
full_time = np.hstack(full_time)



# np.savetxt('lightcurve.txt', np.column_stack([full_time, full_flux]))
# np.savetxt('real_lightcurve.txt', np.column_stack([full_time, flux_true]))
# Save the result
# np.savez("eb.npz", A=A, B=B, t=full_time / 24, flux=full_flux, sigma=sigma)
