# solarsystem.py

import turtle


# Solar System Bodies
class SolarSystemBody(turtle.Turtle):
    def __init__(
            self,
            solar_system,
            mass,
            position=(0, 0),
            velocity=(0, 0),
    ):
        super().__init__()
        self.mass = mass
        self.setposition(position)
        self.velocity = velocity

        self.penup()
        self.hideturtle()

        solar_system.add_body(self)


class Sun(SolarSystemBody):
    ...


class Planet(SolarSystemBody):
    ...


# Solar System
class SolarSystem:
    def __init__(self, width, height):
        self.solar_system = turtle.Screen()
        self.solar_system.tracer(0)
        self.solar_system.setup(width, height)
        self.solar_system.bgcolor("black")

        self.bodies = []

    def add_body(self, body):
        self.bodies.append(body)

    def remove_body(self, body):
        self.bodies.remove(body)
