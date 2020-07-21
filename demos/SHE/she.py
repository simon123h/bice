#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem
from bice.time_steppers import RungeKuttaFehlberg45, ImplicitEuler
from bice.continuation_steppers import NaturalContinuation, PseudoArclengthContinuation

"""
Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
equation, a nonlinear PDE
\partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
"""


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
    def get_continuation_parameter(self):
        return self.r

    def set_continuation_parameter(self, v):
        self.r = v


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = SwiftHohenberg(N=512, L=240)

# time-stepping and plot
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
n = 0
plotevery = 1000
plotID = 0
dudtnorm = 1
if not os.path.exists("initial_state.dat"):
    while dudtnorm > 1e-5:
        # plot
        if n % plotevery == 0:
            ax[0, 0].plot(problem.x, problem.u)
            ax[0, 0].set_xlabel("x")
            ax[0, 0].set_ylabel("solution u(x,t)")
            ax[1, 0].plot(problem.k, np.abs(np.fft.rfft(problem.u)))
            ax[1, 0].set_xlabel("k")
            ax[1, 0].set_ylabel("fourier spectrum u(k,t)")
            ax[0, 1].plot(problem.get_continuation_parameter(),
                          problem.norm(), label="current point", marker="x")
            ax[0, 1].set_xlabel("parameter r")
            ax[0, 1].set_ylabel("L2-norm")
            fig.savefig("out/img/{:05d}.svg".format(plotID))
            for a in ax:
                a.clear()
            plotID += 1
            dudtnorm = np.linalg.norm(problem.rhs(problem.u))
            print("step #: {:}".format(n))
            print("time:   {:}".format(problem.time))
            print("dt:     {:}".format(problem.time_stepper.dt))
            print("|dudt|: {:}".format(dudtnorm))
        n += 1
        # perform timestep
        problem.time_step()
        # perform dealiasing
        problem.dealias()
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            break

    problem.save("initial_state.dat")
else:
    problem.load("initial_state.dat")

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.check_eigenvalues = True

# lists for bifurcation diagram
norms = []
rs = []

n = 0
plotevery = 2
while problem.r < 1:
    # perform continuation step
    sol = problem.continuation_step()
    n += 1
    print("step #:", n)
    print("ds:    ", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        ax[0, 0].plot(problem.x, problem.u)
        ax[0, 0].set_xlabel("x")
        ax[0, 0].set_ylabel("solution u(x,t)")
        ax[1, 0].plot(problem.k, np.abs(np.fft.rfft(problem.u)))
        ax[1, 0].set_xlim((0, problem.k[-1]/2))
        ax[1, 0].set_xlabel("k")
        ax[1, 0].set_ylabel("fourier spectrum u(k,t)")
        for branch in problem.bifurcation_diagram.branches:
            r, norm = branch.data()
            ax[0, 1].plot(r, norm, "--", color="C0")
            r, norm = branch.data(only="stable")
            ax[0, 1].plot(r, norm, color="C0")
            r, norm = branch.data(only="bifurcations")
            ax[0, 1].plot(r, norm, "*", color="C2")
        ax[0, 1].plot(problem.r, problem.norm(),
                      "x", label="current point", color="black")
        ax[0, 1].set_xlabel("parameter r")
        ax[0, 1].set_ylabel("L2-norm")
        ax[0, 1].legend()
        ev_re = np.real(sol.eigenvalues[:20])
        ev_re_n = np.ma.masked_where(
            ev_re > problem.eigval_zero_tolerance, ev_re)
        ev_re_p = np.ma.masked_where(
            ev_re <= problem.eigval_zero_tolerance, ev_re)
        ax[1, 1].plot(ev_re_n, "o", color="C0", label="Re < 0")
        ax[1, 1].plot(ev_re_p, "o", color="C1", label="Re > 0")
        ax[1, 1].axhline(0, color="gray")
        ax[1, 1].legend()
        ax[1, 1].set_ylabel("eigenvalues")
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        for a in ax.flatten():
            a.clear()
        plotID += 1
