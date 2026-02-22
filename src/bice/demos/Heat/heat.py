"""1D Heat Equation demo using Finite Differences method."""

#!/usr/bin/python3
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from bice import Problem, Profiler, profile, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.pde.finite_differences import (
    PeriodicBC,
)


class HeatEquation(FiniteDifferencesEquation):
    r"""
    Finite difference implementation of the 1-dimensional Heat-Equation.

    equation, a linear PDE
    \partial t u &= -k \Delta u.
    """

    def __init__(self, N, L):
        """Initialize the equation."""
        super().__init__()
        # parameters
        self.k = 1
        # create non-uniform grid
        rng = np.random.default_rng(4)
        x = np.cumsum(rng.random(N) * 0.1 + 0.5)
        x = x / x[-1] * L
        self.x = [x - np.mean(x)]
        # self.x = [np.linspace(-L/2, L/2, N)] # uniform grid
        # initial condition
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * np.exp(-0.005 * self.x[0] ** 2)
        # create boundary conditions
        # self.bc = DirichletBC(vals=(0.1, -0.1))
        # self.bc = NeumannBC(vals=(0.01, 0.01))
        # TODO: inhomogeneous NeumannBC somehow break on non-uniform grids for approx_order > 1
        self.bc = PeriodicBC()
        # build finite difference matrices
        self.build_FD_matrices(approx_order=2)
        # mesh adaption settings
        self.max_refinement_error = 5e-2
        self.min_refinement_error = 1e-3
        self.min_dx = 1e-1

    # definition of the equation (right-hand side)
    def rhs(self, u):
        """Calculate the right-hand side of the equation."""
        # return -self.nabla(u)  # advection equation
        return self.laplace(u)

    # definition of the Jacobian
    @profile
    def jacobian(self, u):
        """Calculate the Jacobian of the equation."""
        return self.laplace()

    def plot(self, ax):
        """Plot the equation's solution."""
        ax.plot(self.x[0], self.u, marker="x")


class HeatProblem(Problem):
    """Problem class for the 1D Heat equation."""

    def __init__(self, N, L):
        """Initialize the problem."""
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.she = HeatEquation(N, L)
        self.add_equation(self.she)
        # initialize time stepper
        # self.time_stepper = time_steppers.BDF(self)
        self.time_stepper = time_steppers.BDF2(dt=1)
        # assign the continuation parameter
        self.continuation_parameter = (self.she, "r")


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = HeatProblem(N=256, L=100)

# create figure
fig, ax = plt.subplots(1, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 1
dudtnorm = 1
Profiler.start()
while dudtnorm > 1e-5:
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        ax.set_ylim((-1, 1))
        fig.savefig(f"out/img/{plotID:05d}.png")
        plotID += 1
        print(f"step #: {n}")
        print(f"time:   {problem.time}")
        print(f"dt:     {problem.time_stepper.dt}")
        print(f"|dudt|: {dudtnorm}")
    n += 1
    # perform timestep
    problem.time_step()
    # adapt the mesh
    problem.adapt()
    # calculate the new norm
    dudtnorm = np.linalg.norm(problem.rhs(problem.u))
    # catch divergent solutions
    if np.max(problem.u) > 1e12:
        break
Profiler.print_summary()
