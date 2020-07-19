#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use("GTK3Agg")  # noqa
import matplotlib.pyplot as plt
import sys
sys.path.append("..")  # noqa, needed for relative import of package
from bice import Problem
from bice.time_steppers import Euler, ImplicitEuler, RungeKutta4


# The Lotka-Volterra equations (predator prey model)
class LotkaVolterra(Problem):

    def __init__(self):
        super().__init__()
        # parameters
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1
        # initial condition
        self.u = np.array([1., 0.7])
        # time stepper
        # self.time_stepper = Euler()
        # self.time_stepper = ImplicitEuler()
        self.time_stepper = RungeKutta4()

    def rhs(self, u):
        x = u[0]
        y = u[1]
        return np.array([
            (self.a - self.b * y) * x,
            (self.d * x - self.c) * y
        ])

# create problem
problem = LotkaVolterra()

# time-stepping
data = []
for n in range(10000):
    data.append(problem.u.copy())
    problem.time_step()

data = np.array(data).T
plt.plot(data[0], data[1])
plt.show()
