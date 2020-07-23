#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use("GTK3Agg")  # noqa
import matplotlib.pyplot as plt
import sys
sys.path.append("..")  # noqa, needed for relative import of package
from bice import Problem
from bice.time_steppers import Euler, ImplicitEuler, RungeKutta4, BDF, BDF2
import time


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
        self.time_stepper = BDF(self)
        # self.time_stepper = BDF2(dt=1e-1)
        # self.time_stepper = Euler(dt=5e-4)
        # self.time_stepper = ImplicitEuler(dt=1e-2)
        # self.time_stepper = RungeKutta4(dt=2e-1)

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
print(problem.time_stepper)
# for n in range(10000):
start = time.time()
while problem.time < 1e3:
    print(problem.time / 1e3)
    data.append(problem.u.copy())
    problem.time_step()
end = time.time()
print("# steps:", len(data))
print("time   :", end - start)


data = np.array(data).T
plt.plot(data[0], data[1])
plt.show()
