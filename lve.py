#!/usr/bin/python3
from src.problem import Problem
import numpy as np

import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

# The Lotka-Volterra equations (predator prey model)
class LotkaVolterra(Problem):

    def __init__(self):
        super().__init__()
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1
        self.u = np.array([1., 0.7])

    def rhs(self, u):
        x = u[0]
        y = u[1]
        return np.array([
            (self.a - self.b * y) * x,
            (self.d * x - self.c) * y
        ])

problem = LotkaVolterra()

dat = []

for n in range(10000):
    dat.append(problem.u.copy())
    problem.time_step()

dat = np.array(dat).T
plt.plot(dat[0], dat[1])
plt.show()

