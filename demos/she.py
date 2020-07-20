#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("..")  # noqa, needed for relative import of package
from bice import Problem
from bice.time_steppers import RungeKuttaFehlberg45, ImplicitEuler
from bice.continuation_steppers import NaturalContinuation, PseudoArclengthContinuation

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
        self.u = np.cos(2 * np.pi * self.x / 10) * np.exp(-0.005 * self.x**2)
        # initialize time stepper
        self.time_stepper = RungeKuttaFehlberg45(dt=1e-3)
        self.time_stepper.error_tolerance = 1e-7
        self.time_stepper.dt = 1e-3

    def rhs(self, u):
        u_k = np.fft.rfft(u)
        return np.fft.irfft((self.r - (self.kc**2 - self.k**2)**2) * u_k) + self.v * u**2 - self.g * u**3

    def dealias(self, fraction=1./2.):
        u_k = np.fft.rfft(self.u)
        N = len(u_k)
        u_k[-int(N*fraction):] = 0
        self.u = np.fft.irfft(u_k)

    # for continuation
    def get_parameter(self):
        return self.r

    def set_parameter(self, v):
        self.r = v

    def L2norm(self):
        return np.linalg.norm(self.u)

    def save(self, filename):
        np.savetxt(filename, problem.u)

    def load(self, filename):
        problem.u = np.loadtxt(filename)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = SwiftHohenberg(N=512, L=240)

# time-stepping and plot
fig, ax = plt.subplots(2, 1, figsize=(12.8,7.2))
plotevery = 1000
n = 0
dudtnorm = 1
if not os.path.exists("state.dat"):
    while dudtnorm > 1e-5:
        # plot
        if n % plotevery == 0:
            ax[0].plot(problem.x, problem.u)
            u_k = np.fft.rfft(problem.u)
            ax[1].plot(problem.k, np.abs(u_k))
            fig.savefig("out/img/{:05d}.svg".format(n//plotevery))
            ax[0].clear()
            ax[1].clear()
            dudtnorm = np.linalg.norm(problem.rhs(problem.u))
            print("Step #{:05d}".format(n//plotevery))
            print("dt:   {:}".format(problem.time_stepper.dt))
            print("time: {:}".format(problem.time))
            print("dudt: {:}".format(dudtnorm))
        n += 1
        # perform timestep
        problem.time_step()
        # perform dealiasing
        problem.dealias()
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            break

    problem.save("state.dat")
else:
    problem.load("state.dat")

problem.continuation_stepper = PseudoArclengthContinuation()
problem.continuation_stepper.ds = 1e-5


norms = []
rs = []

plt.cla()

print("Starting continuation")

n = 0
while problem.r < 1:
    norms.append(problem.L2norm())
    rs.append(problem.get_parameter())
    ax[0].plot(problem.x, problem.u)
    ax[1].plot(rs, norms)
    fig.savefig("out/img/c{:05d}.svg".format(n))
    n += 1
    ax[0].clear()
    ax[1].clear()
    print("r:", problem.r)
    print("norm:", problem.L2norm())
    print("step #:", n)
    problem.continuation_step()
    # perform dealiasing
    problem.dealias()
    n += 1