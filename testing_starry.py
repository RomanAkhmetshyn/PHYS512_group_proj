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
ydeg = 4


map = starry.Map(ydeg=ydeg)
A_y = np.array(map.y[1:])
# map.show(colorbar=True)
#2.5 hour base line
map.reset()
# map.add_spot(amp=0.1, sigma=0.25, lat=-20, lon=-50)
# map.add_spot(amp=amp, sigma=sigma, lat=lat, lon=lon)

B_y = np.array(map.y[1:])
# map.show(colorbar=True, projection='moll')

A = dict(
    ydeg=ydeg,  # degree of the map
    udeg=2,  # degree of the limb darkening
    inc=90.0,  # inclination in degrees
    amp=1.0,  # amplitude (a value prop. to luminosity)
    r=1,  #  radius in R_sun
    m=1,  # mass in M_sun
    prot=13,  # rotational period in days
    u=[0.40, 0.25],  # limb darkening coefficients
    y=A_y,  # the spherical harmonic coefficients
)

B = dict(
    ydeg=ydeg,  # degree of the map
    udeg=2,  # degree of the limb darkening
    # inc=-39,  # inclination in degrees
    inc=90,
    # amp=0.0165,  # amplitude (a value prop. to luminosity)
    amp=0.001,  # amplitude (a value prop. to luminosity)
    r=0.08,  #  radius in R_sun
    m=0.08,  #  mass in M_sun
    porb=3,  # orbital period in days
    # prot=0.1,  # rotational period in days
    prot=5,
    t0=2,  # reference time in days (when it transits A)
    u=[0, 0],  # limb darkening coefficients
    y=B_y,  # the spherical harmonic coefficients
)

C = dict(
    ydeg=ydeg,  # degree of the map
    udeg=2,  # degree of the limb darkening
    # inc=-39,  # inclination in degrees
    inc=90,
    # amp=0.0165,  # amplitude (a value prop. to luminosity)
    amp=0.0001,  # amplitude (a value prop. to luminosity)
    r=0.06,  #  radius in R_sun
    m=0.5,  #  mass in M_sun
    porb=7,  # orbital period in days
    # prot=0.1,  # rotational period in days
    prot=5,
    t0=3,  # reference time in days (when it transits A)
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
    inc=90,
    ecc=0
)

sec2 = starry.Secondary(
    starry.Map(ydeg=C["ydeg"], udeg=C["udeg"], inc=C["inc"], amp=C["amp"]),
    r=C["r"],
    m=C["m"],
    porb=C["porb"],
    prot=C["prot"],
    t0=C["t0"],
    inc=90,
    ecc=0,
    Omega=0
    
)

sec.map[1:] = B["u"]
sec.map[1:, :] = B["y"]
# sec.map.show(colorbar=True)

sys = starry.System(pri, sec, sec2)

# sys.show(t=np.linspace(0, 7.5, 500), window_pad=10, 
#           file='sys.gif', figsize=(20,16))

#%%

tot_time = 100
time = np.linspace(0, tot_time, 100000)
flux = sys.flux(time)

plt.plot(time, flux)
# plt.xlim(1.8, 2.2)
plt.show()

#%%

plt.figure(figsize=(10, 6))
for i in range(tot_time//B["porb"]):
    center_time = 2 + i * B["porb"]
    # Get indices within the specified window around the center time
    indices = (time >= center_time - 0.1) & (time <= center_time + 0.1)
    
    # Plot each transit segment, aligning them at the center
    plt.plot(time[indices] - center_time, flux[indices], label=f'Transit {i + 1}')

# Set up the plot
plt.xlabel('Time (days from transit center)')
plt.ylabel('Flux')
plt.title('Overplotted Transits')
plt.legend()
plt.show()

#%%

# full_flux = []
# full_time = []

# for rot in range(0, 1):
#     fig, ax = plt.subplots(1, figsize=(12, 5))  
#     t = np.linspace(0 + rot * 12.713 * 24, 4.9894+2.5 + rot * 12.713 * 24, 5000)
#     flux_true = sys.flux(t / 24)
#     # sigma = 0.011
#     sigma = 1.6e-4
#     flux = flux_true + sigma * np.random.randn(len(t))
    
#     # Append each flux and time into full_flux and full_time
    
#     # flux_binned = bin_data([flux], 0, 50, np.mean)[0]
#     # t_binned = bin_data([t], 0, 50, np.mean)[0]
    
#     full_flux.append(flux)
#     full_time.append(t)
#     # full_flux.append(flux_binned)
#     # full_time.append(t_binned)
    
#     # Plot the results for each rotation
#     ax.plot(t, flux, "k.", alpha=0.3, ms=2)
#     # ax.plot(t_binned, flux_binned, "k.", alpha=0.3, ms=2)
#     ax.plot(t, flux_true, lw=1)
#     ax.set_xlabel("time [hours]", fontsize=24)
#     ax.set_ylabel("normalized flux", fontsize=24)
#     plt.show()

# # Stack the arrays into one column of values instead of separate columns
# full_flux = np.hstack(full_flux)
# full_time = np.hstack(full_time)