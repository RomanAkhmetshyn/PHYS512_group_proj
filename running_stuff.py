# simple_solar_system.py

from testing_stuff import SolarSystem, Sun, Planet
import time


solar_system = SolarSystem(width=2500, height=1400)

sun = Sun(solar_system, mass=10000, velocity=(0, 0))
planets = (
    Planet(
        solar_system,
        mass=1000,
        position=(-600, 0),
        velocity=(0, 4),
    ),
    Planet(
        solar_system,
        mass=0.01,
        position=(-620, -80),
        velocity=(3.6, 4),
    ),
)

while True:
    solar_system.calculate_all_body_interactions()
    solar_system.update_all()
    time.sleep(0.006)
