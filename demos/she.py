#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("..")  # noqa, needed for relative import of package
from bice import Problem
from bice.time_steppers import RungeKuttaFehlberg45

# Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
# equation, a nonlinear PDE
# \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3


class SwiftHohenberg(Problem):

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.r = -0.013
        self.kc = 0.5
        self.v = 0.41
        self.g = 1
        # space and fourier space
        self.x = np.linspace(-L/2, L/2, N)
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initialize unknowns
        self.u = 1 * np.cos(2 * np.pi * self.x / 10) * \
            np.exp(-0.005 * self.x**2)
        # initialize time stepper
        self.time_stepper = RungeKuttaFehlberg45(dt=1e-3)
        self.time_stepper.error_tolerance = 1e-1

    def rhs(self, u):
        u_k = np.fft.rfft(u)
        return np.fft.irfft((self.r - (self.kc**2 - self.k**2)**2) * u_k) + self.v * u**2 - self.g * u**3

    def dealias(self, fraction=1./2.):
        u_k = np.fft.rfft(self.u)
        N = len(u_k)
        u_k[-int(N*fraction):] = 0
        self.u = np.fft.irfft(u_k)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = SwiftHohenberg(512, 240)

# time-stepping and plot
fig, ax = plt.subplots()
plotevery = 500
n = 0
while True:
    if n % plotevery == 0:
        ax.plot(problem.x, problem.u)
        # u_k = np.fft.rfft(problem.u)
        # ax.plot(problem.k, np.abs(u_k))
        fig.savefig("out/img/{:05d}.svg".format(n//plotevery))
        ax.clear()
        print("Step #{:05d}".format(n//plotevery))
        print("dt:   {:}".format(problem.time_stepper.dt))
    n += 1
    problem.time_step()
    problem.dealias()
    if np.max(problem.u) > 1e12:
        break
