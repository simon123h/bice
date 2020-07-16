import numpy as np


class TimeStepper():

    def __init__(self, dt):
        self.dt = dt

    # perform a timestep on a problem
    def step(self, problem):
        raise NotImplementedError

# Explicit Euler scheme
class Euler(TimeStepper):
    # perform timestep
    def step(self, problem):
        problem.u += self.dt * problem.rhs(problem.u)
        problem.time += self.dt

# Classical Runge-Kutta-4 scheme
class RungeKutta4(TimeStepper):
    # perform timestep
    def step(self, problem):
        k1 = problem.rhs(problem.u)
        problem.time += self.dt/2
        k2 = problem.rhs(problem.u + self.dt / 2 * k1)
        k3 = problem.rhs(problem.u + self.dt / 2 * k2)
        problem.time += self.dt/2
        k4 = problem.rhs(problem.u + self.dt * k3)
        problem.u += self.dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4)