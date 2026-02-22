#!/usr/bin/python3
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags

from bice import Problem, time_steppers
from bice.continuation import TranslationConstraint, VolumeConstraint
from bice.measure import LyapunovExponentCalculator
from bice.pde.finite_differences import FiniteDifferencesEquation, PeriodicBC


class NikolaevskiyEquation(FiniteDifferencesEquation):
    r"""
    Finite difference implementation of the 2-dimensional Nikolaevskiy Equation
    equation, a nonlinear PDE
    \partial t h &= -\Delta (r - (1+\Delta)^2) h - 1/2 (\nabla h)^2.
    """

    def __init__(self, Nx, Ny):
        super().__init__()
        # parameters
        self.r = 0.5  # drive
        self.m = 10  # characteristic system length
        self.ratio = 1  # length ratio Ly/Lx
        # space and fourier space
        self.x = [np.linspace(0, 1, Nx), np.linspace(0, 1, Ny)]
        # build finite difference matrices
        self.bc = PeriodicBC()
        self.build_FD_matrices()
        # initial condition
        self.u = 2 * (np.random.rand(Nx, Ny) - 0.5) * 1e-5
        self.u = self.u.ravel()
        # create constraints
        self.volume_constraint = VolumeConstraint(self)
        self.translation_constraint_x = TranslationConstraint(self, direction=0)
        self.translation_constraint_y = TranslationConstraint(self, direction=1)

    # characteristic length scale
    @property
    def L0(self):
        return 2 * np.pi / np.sqrt(1 + np.sqrt(self.r))

    # definition of the Nikolaevskiy equation (right-hand side)
    def rhs(self, u):
        # calculate the system length
        Lx = self.L0 * self.m
        Ly = Lx * self.ratio
        # include length scale in the differentiation operators
        nabla_x = self.nabla[0] / Lx
        nabla_y = self.nabla[1] / Ly
        Delta = self.ddx[2][0] / Lx**2 + self.ddx[2][1] / Ly**2
        Delta2 = Delta.dot(Delta)
        lin = -Delta.dot(self.r * u - (u + 2 * Delta.dot(u) + Delta2.dot(u)))
        nonlin = nabla_x.dot(u) ** 2 + nabla_y.dot(u) ** 2
        return lin - 0.5 * nonlin

    # definition of the Jacobian
    def jacobian(self, u):
        Lx = self.L0 * self.m
        Ly = Lx * self.ratio
        nabla_x = self.nabla[0] / Lx
        nabla_y = self.nabla[1] / Ly
        Delta = self.ddx[2][0] / Lx**2 + self.ddx[2][1] / Ly**2
        Delta2 = Delta.dot(Delta)
        lin = (1 - self.r) * Delta + Delta.dot(2 * Delta + Delta2)
        nonlin = diags(nabla_x.dot(u)) * nabla_x + diags(nabla_y.dot(u)) * nabla_y
        return lin - nonlin

    # plot the solution
    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        x, y = np.meshgrid(self.x[0], self.x[1])
        Lx = self.L0 * self.m
        Ly = Lx * self.ratio
        pcol = ax.pcolor(x * Lx, y * Ly, self.u.reshape(x.shape), cmap="coolwarm", rasterized=True)
        pcol.set_edgecolor("face")
        # put velocity labels into plot
        ax.text(
            0.02,
            0.06,
            f"vx = {self.translation_constraint_x.u[0]:.1g}",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.02,
            f"vy = {self.translation_constraint_y.u[0]:.1g}",
            transform=ax.transAxes,
        )


class NikolaevskiyProblem(Problem):
    def __init__(self, Nx, Ny):
        super().__init__()
        # Add the Nikolaevskiy equation to the problem
        self.ne = NikolaevskiyEquation(Nx, Ny)
        self.add_equation(self.ne)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF(self, dt_max=1e-1)
        # assign the continuation parameter
        self.continuation_parameter = (self.ne, "m")

    # reset zero-mode, for conservation of volume
    def dealias(self):
        # subtract volume
        self.ne.u -= np.mean(self.ne.u)

    # Norm is the L2-norm of the NE
    def norm(self):
        # TODO: divide by Nx*Nx*Lx*Ly
        return np.linalg.norm(self.ne.u)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = NikolaevskiyProblem(Nx=12, Ny=12)
problem.ne.r = 0.5
problem.ne.m = 1.25

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1
T = 100 / problem.ne.r
if not os.path.exists("initial_state.npz"):
    while problem.time < T:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig(f"out/img/{plotID:05d}.svg")
            plotID += 1
            print(f"step #: {n}")
            print(f"time:   {problem.time}")
            print(f"dt:     {problem.time_stepper.dt}")
            print(f"|dudt|: {dudtnorm}")
        n += 1
        # perform timestep
        problem.time_step()
        # perform dealiasing
        problem.dealias()
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            print("diverged")
            break
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

# calculate Lyapunov exponents
problem.time_stepper = time_steppers.BDF2(dt=0.1)
lyapunov = LyapunovExponentCalculator(problem, nexponents=10, epsilon=1e-6, nintegration_steps=1)

while True:
    lyapunov.step()
    problem.dealias()
    ax.clear()
    ax.plot(lyapunov.exponents, marker="o")
    fig.savefig(f"out/img/{plotID:05d}.svg")
    plotID += 1
    print("Lyapunov exponents:", lyapunov.exponents)
