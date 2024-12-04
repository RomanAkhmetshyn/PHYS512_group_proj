import numpy as np
from matplotlib import pyplot as plt

G_SI = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
AU = 1.496e11  # 1 Astronomical Unit in meters
Mjup = 1.898e27  # Jupiter mass in kilograms
day = 24 * 3600  # 1 day in seconds

# Gravitational constant in AU^3 Mjup^-1 day^-2
G = G_SI / (AU**3 / (Mjup * (day)**2))


# Function to calculate initial conditions


# Function to calculate initial conditions
def kepler_to_cartesian(T, e, omega, true_anomaly, G, M_central, M_orbiting):
    """
    Convert Keplerian elements to Cartesian coordinates and velocities.

    Parameters:
        T (float): Orbital period in days.
        e (float): Eccentricity (0 for circular orbit).
        omega (float): Argument of periapsis in radians.
        true_anomaly (float): True anomaly in radians.
        G (float): Gravitational constant in AU^3 / Mjup / day^2.
        M_central (float): Mass of the central body in Mjup.
        M_orbiting (float): Mass of the orbiting body in Mjup.

    Returns:
        x (float): Initial x position in AU.
        y (float): Initial y position in AU.
        vx (float): Initial x velocity in AU/day.
        vy (float): Initial y velocity in AU/day.
    """
    # Total mass
    M_total = M_central + M_orbiting

    # Semi-major axis from period (Kepler's Third Law)
    a = (G * M_total * (T / (2 * np.pi))**2)**(1 / 3)

    # Radial distance for the given true anomaly
    r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))

    # Radial and tangential velocities
    v_r = np.sqrt(G * M_total / a) * e * np.sin(true_anomaly) / (1 - e**2)**0.5
    v_theta = np.sqrt(G * M_total / a) * \
        (1 + e * np.cos(true_anomaly)) / (1 - e**2)**0.5

    # Cartesian position in orbital plane
    x_orbit = r * np.cos(true_anomaly)
    y_orbit = r * np.sin(true_anomaly)

    # Cartesian velocity in orbital plane
    vx_orbit = v_r * np.cos(true_anomaly) - v_theta * np.sin(true_anomaly)
    vy_orbit = v_r * np.sin(true_anomaly) + v_theta * np.cos(true_anomaly)

    # Rotate position and velocity to account for argument of periapsis
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    x = cos_omega * x_orbit - sin_omega * y_orbit
    y = sin_omega * x_orbit + cos_omega * y_orbit
    vx = cos_omega * vx_orbit - sin_omega * vy_orbit
    vy = sin_omega * vx_orbit + cos_omega * vy_orbit

    return x, y, vx, vy


# Initial conditions for three particles (in A.U. and A.U./year)
x0, y0, vx0, vy0 = 0, 0, 0, 0  # Particle 0
# Particle 1 (orbital velocity for circular orbit)

# Masses of the particles (in Mjup)
m0 = 1000.0  # Mass of particle 0
m1 = 1.0  # Mass of particle 1
m2 = 1.0  # Mass of particle 2

x1, y1, vx1, vy1 = kepler_to_cartesian(10, 0.0, 0.0, 0.0, G, m0, m1)
x2, y2, vx2, vy2 = kepler_to_cartesian(160, 0.0, 0, np.pi, G, m0, m2)

# vx2 = vx2 / 1.75
# vy2 = vy2 / 1.75


# Simulation parameters
dt = 0.001  # Time step in days
tmax = 500  # Total simulation time in days
dprint = 365 * 2  # Number of sub-steps for smooth plotting
# dt = dt / dprint

# Arrays for tracking positions and detecting transits
num_steps = len(np.arange(0, tmax, dt)) + 1
xs = np.zeros(len(np.arange(0, tmax, dt)) + 1)
xs[0] = x1
ts = []  # Time of transits

positions_0 = np.zeros((num_steps, 2))  # Particle 0 (x, y)
positions_1 = np.zeros((num_steps, 2))  # Particle 1 (x, y)
positions_2 = np.zeros((num_steps, 2))  # Particle 2 (x, y)

positions_0[0] = [x0, y0]
positions_1[0] = [x1, y1]
positions_2[0] = [x2, y2]

# Simulation loop
plt.figure(figsize=(8, 8))
for i, t in enumerate(np.arange(0, tmax, dt)):
    # Calculate distances and forces
    # Particle 0 and Particle 1
    dx01 = x0 - x1
    dy01 = y0 - y1
    r01_square = dx01**2 + dy01**2
    r01 = np.sqrt(r01_square)
    r01_cubed = r01 * r01_square

    # Particle 0 and Particle 2
    dx02 = x0 - x2
    dy02 = y0 - y2
    r02_square = dx02**2 + dy02**2
    r02 = np.sqrt(r02_square)
    r02_cubed = r02 * r02_square

    # Particle 1 and Particle 2
    dx12 = x1 - x2
    dy12 = y1 - y2
    r12_square = dx12**2 + dy12**2
    r12 = np.sqrt(r12_square)
    r12_cubed = r12 * r12_square

    # Forces on Particle 0
    fx0 = G * m1 * dx01 / r01_cubed + G * m2 * dx02 / r02_cubed
    fy0 = G * m1 * dy01 / r01_cubed + G * m2 * dy02 / r02_cubed

    # Forces on Particle 1
    fx1 = -G * m0 * dx01 / r01_cubed + G * m2 * dx12 / r12_cubed
    fy1 = -G * m0 * dy01 / r01_cubed + G * m2 * dy12 / r12_cubed

    # Forces on Particle 2
    fx2 = -G * m0 * dx02 / r02_cubed - G * m1 * dx12 / r12_cubed
    fy2 = -G * m0 * dy02 / r02_cubed - G * m1 * dy12 / r12_cubed

    # Update velocities
    vx0 += -dt * fx0 / m0
    vy0 += -dt * fy0 / m0
    vx1 += -dt * fx1 / m1
    vy1 += -dt * fy1 / m1
    vx2 += -dt * fx2 / m2
    vy2 += -dt * fy2 / m2

    # Update positions
    x0 += dt * vx0
    y0 += dt * vy0
    x1 += dt * vx1
    y1 += dt * vy1
    x2 += dt * vx2
    y2 += dt * vy2

    xs[i + 1] = x1

    positions_0[i + 1] = [x0, y0]
    positions_1[i + 1] = [x1, y1]
    positions_2[i + 1] = [x2, y2]

    # Detect transits
    if y1 < 0 and xs[i] < 0 and xs[i + 1] > 0:
        ts.append(t)

    # Plotting
    if np.abs(t % 0.1) < 1e-6:
        plt.plot(x0, y0, 'r.', markersize=1)
        plt.plot(x1, y1, 'b.', markersize=0.5)
        plt.plot(x2, y2, 'g.', markersize=0.5)
        # plt.ylim(-10, 10)
        # plt.xlim(-10, 10)
        plt.ylim(-0.8, 0.8)
        plt.xlim(-0.8, 0.8)
        # plt.pause(0.001)

    # KE = 0.5 * (m0 * (vx0**2 + vy0**2) + m1 *
    #             (vx1**2 + vy1**2) + m2 * (vx2**2 + vy2**2))
    # PE = -G * (m0 * m1 / r01 + m0 * m2 / r02 + m1 * m2 / r12)
    # print(KE + PE)


plt.show()

np.save("positions_0.npy", positions_0)
np.save("positions_1.npy", positions_1)
np.save("positions_2.npy", positions_2)
# %%
# Post-analysis: Observed minus Calculated (O-C) analysis
ts = np.array(ts)
periods = np.diff(ts)
expected = [ts[0] + np.mean(periods) * n for n in range(len(ts))]

OC = ts - expected

plt.plot(ts, OC*24*60)
plt.xlabel("days")
plt.ylabel("O-C (minutes)")
plt.title("Observed Minus Calculated Analysis")
plt.show()

# plt.plot(periods)
# plt.show()

# %%


# Visualization (optional)
plt.figure(figsize=(8, 8))
plt.plot(positions_0[:, 0], positions_0[:, 1], label="Particle 0")
plt.plot(positions_1[:, 0], positions_1[:, 1], label="Particle 1")
plt.plot(positions_2[:, 0], positions_2[:, 1], label="Particle 2")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.ylim(-0.8, 0.8)
plt.xlim(-0.8, 0.8)
plt.legend()
plt.title("Trajectories of Particles")
plt.show()
