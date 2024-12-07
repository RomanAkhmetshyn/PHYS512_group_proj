import time
import numpy as np
from matplotlib import pyplot as plt
import numba as nb

G_SI = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
AU = 1.496e11  # 1 Astronomical Unit in meters
Mjup = 1.898e27  # Jupiter mass in kilograms
day = 24 * 3600  # 1 day in seconds

# Gravitational constant in AU^3 Mjup^-1 day^-2
G = G_SI / (AU**3 / (Mjup * (day)**2))


# Function to calculate initial conditions
def compute_total_energy(x0, y0, vx0, vy0, m0,
                         x1, y1, vx1, vy1, m1,
                         x2, y2, vx2, vy2, m2, G, epsilon=0):
    # Kinetic energy
    KE = (0.5 * m0 * (vx0**2 + vy0**2) +
          0.5 * m1 * (vx1**2 + vy1**2) +
          0.5 * m2 * (vx2**2 + vy2**2))

    # Potential energy
    r01 = np.sqrt((x0 - x1)**2 + (y0 - y1)**2 + epsilon**2)
    r02 = np.sqrt((x0 - x2)**2 + (y0 - y2)**2 + epsilon**2)
    r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + epsilon**2)
    PE = (-G * m0 * m1 / r01 - G * m0 * m2 / r02 - G * m1 * m2 / r12)

    # Total energy
    return KE + PE


@nb.njit(parallel=False)
def kepler_to_cartesian(T, e, omega, G, M_central):
    """
    Convert Keplerian elements to Cartesian coordinates and velocities.

    Parameters:
        T (float): Orbital period in days.
        e (float): Eccentricity (0 for circular orbit).
        omega (float): Argument of periapsis in radians.
        true_anomaly (float): True anomaly in radians.
        G (float): Gravitational constant in AU^3 / Mjup / day^2.
        M_central (float): Mass of the central body in Mjup.

    Returns:
        x (float): Initial x position in AU.
        y (float): Initial y position in AU.
        vx (float): Initial x velocity in AU/day.
        vy (float): Initial y velocity in AU/day.
    """

    true_anomaly = 0
    # Semi-major axis from period (Kepler's Third Law)
    a = (G * M_central * (T / (2 * np.pi))**2)**(1 / 3)

    # Radial distance for the given true anomaly
    r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))

    # Orbital velocity magnitude (Vis-viva equation)
    v = np.sqrt(G * M_central * (2 / r - 1 / a))

    # Cartesian position in the orbital plane
    x_orbit = r * np.cos(true_anomaly)
    y_orbit = r * np.sin(true_anomaly)

    # Cartesian velocity in the orbital plane
    vx_orbit = -v * np.sin(true_anomaly)
    vy_orbit = v * (e + np.cos(true_anomaly)) / (1 + e * np.cos(true_anomaly))

    # Rotate position and velocity to account for argument of periapsis
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    x = cos_omega * x_orbit - sin_omega * y_orbit
    y = sin_omega * x_orbit + cos_omega * y_orbit
    vx = cos_omega * vx_orbit - sin_omega * vy_orbit
    vy = sin_omega * vx_orbit + cos_omega * vy_orbit

    return x, y, vx, vy


@nb.njit(parallel=False)
def twoPlanetTTV(m0=1000, m1=1, m2=1,
                 T1=10, T2=100, e1=0, e2=0, w1=0, w2=0, Ntransits=10, savepos=True):

    m0 = m0*1047

    # Initial conditions for three particles (in A.U. and A.U./year)
    x0, y0, vx0, vy0 = 0, 0, 0, 0  # Particle 0

    x1, y1, vx1, vy1 = kepler_to_cartesian(T1, e1, w1, G, m0)
    x2, y2, vx2, vy2 = kepler_to_cartesian(T2, e2, w2, G, m0)

    # Simulation parameters
    dt = 0.0001  # Time step in days
    tmax = T1*Ntransits  # Total simulation time in days
    # dprint = 365 * 2  # Number of sub-steps for smooth plotting
    # dt = dt / dprint

    # Arrays for tracking positions and detecting transits
    num_steps = len(np.arange(0, tmax, dt)) + 1
    xs = np.zeros(len(np.arange(0, tmax, dt)) + 1)
    xs[0] = x1
    ts = np.zeros(Ntransits)  # Time of transits
    trans = 0

    # if savepos:

    #     positions_0 = np.zeros((num_steps, 2))  # Particle 0 (x, y)
    #     positions_1 = np.zeros((num_steps, 2))  # Particle 1 (x, y)
    #     positions_2 = np.zeros((num_steps, 2))  # Particle 2 (x, y)

    #     positions_0[0] = [x0, y0]
    #     positions_1[0] = [x1, y1]
    #     positions_2[0] = [x2, y2]

    # energies = []

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
        fx0 = G * m1 * m0 * dx01 / r01_cubed + G * m2 * m0 * dx02 / r02_cubed
        fy0 = G * m1 * m0 * dy01 / r01_cubed + G * m2 * m0 * dy02 / r02_cubed

        # Forces on Particle 1
        fx1 = -G * m0 * m1 * dx01 / r01_cubed + G * m2 * m1 * dx12 / r12_cubed
        fy1 = -G * m0 * m1 * dy01 / r01_cubed + G * m2 * m1 * dy12 / r12_cubed

        # Forces on Particle 2
        fx2 = -G * m0 * m2 * dx02 / r02_cubed - G * m1 * m2 * dx12 / r12_cubed
        fy2 = -G * m0 * m2 * dy02 / r02_cubed - G * m1 * m2 * dy12 / r12_cubed

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

        # x0 = 0
        # y0 = 0

        xs[i + 1] = x1

        # if savepos:
        #     positions_0[i + 1] = [x0, y0]
        #     positions_1[i + 1] = [x1, y1]
        #     positions_2[i + 1] = [x2, y2]

        # Detect transits
        if y1 < 0 and xs[i] < 0 and xs[i + 1] > 0:
            ts[trans] = t
            trans += 1

        # E_total = compute_total_energy(x0, y0, vx0, vy0, m0,
        #                                 x1, y1, vx1, vy1, m1,
        #                                 x2, y2, vx2, vy2, m2, G)
        # energies.append(E_total)

    # if savepos:
    #     np.save("positions_0.npy", positions_0)
    #     np.save("positions_1.npy", positions_1)
    #     np.save("positions_2.npy", positions_2)

    return ts


# %%
# Post-analysis: Observed minus Calculated (O-C) analysis
if __name__ == "__main__":
    start_time = time.time()
    ts = twoPlanetTTV(m0=1, m1=1, m2=3,
                      T1=10, T2=160, e1=0, e2=0.0, w1=0, w2=np.pi/6, Ntransits=10, savepos=True)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    periods = np.diff(ts)

    expected = [ts[0] + np.mean(periods) * n for n in range(len(ts))]

    OC = ts - expected

    plt.plot(ts, OC*24*60)
    plt.xlabel("days")
    plt.ylabel("O-C (minutes)")
    plt.title("Observed Minus Calculated Analysis")
    plt.show()

    np.savetxt('observed_OC(mins).txt', OC*24*60)

# plt.plot(periods)
# plt.show()

# %%
# positions_0 = np.load('positions_0.npy')
# positions_1 = np.load('positions_1.npy')
# positions_2 = np.load('positions_2.npy')

# # Visualization (optional)
# plt.figure(figsize=(8, 8))
# plt.plot(positions_0[:, 0], positions_0[:, 1], label="Particle 0")
# plt.plot(positions_1[:, 0], positions_1[:, 1], label="Particle 1")
# plt.plot(positions_2[:, 0], positions_2[:, 1], label="Particle 2")
# plt.xlabel("x (AU)")
# plt.ylabel("y (AU)")
# plt.ylim(-1, 1)
# plt.xlim(-1, 1)
# plt.legend()
# plt.title("Trajectories of Particles")
# plt.show()

# %%

# energy_change = np.abs((energies - energies[0]) / energies[0])

# Plot energy change
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(0, tmax, dt), energies - energies[0])
# plt.xlabel("Time (days)")
# plt.ylabel("Relative Change in Energy")
# plt.title("Energy Conservation Check")
# plt.grid()
# plt.show()
