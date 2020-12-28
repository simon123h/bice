#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
from scipy.sparse import diags
import scipy.sparse as sp
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.pde.finite_differences import DirichletBC, NeumannBC, PeriodicBC
from bice.continuation import ConstraintEquation
from bice import profile, Profiler


class HeatEquation(FiniteDifferencesEquation):
    r"""
    Finite difference implementation of the 1-dimensional Heat-Equation
    equation, a nonlinear PDE
    \partial t u &= -k \Delta u
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.k = 1
        # spatial coordinate
        self.x = [np.linspace(-L/2, L/2, N)]
        # initial condition
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * \
            np.exp(-0.005 * self.x[0] ** 2)
        # create boundary conditions
        dbc = DirichletBC(vals=(0.1, 0.1))
        nbc = NeumannBC(vals=(0.01, 0.01))
        pbc = PeriodicBC()
        # build finite difference matrices
        self.build_FD_matrices(boundary_conditions=dbc, premultiply_bc=False)

    # definition of the equation (right-hand side)
    def rhs(self, u):
        u_pad = self.bc.pad(u)
        # return -self.nabla.dot(u_pad)  # advection equation
        return self.laplace.dot(u_pad)

    # definition of the Jacobian
    @profile
    def jacobian(self, u):
        return self.laplace.dot(self.bc.Q)


class HeatProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.she = HeatEquation(N, L)
        self.add_equation(self.she)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF(self)
        # assign the continuation parameter
        self.continuation_parameter = (self.she, "r")


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = HeatProblem(N=500, L=100)

# create figure
fig, ax = plt.subplots(1, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1
Profiler.start()
while dudtnorm > 1e-5:
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        ax.set_ylim((-1, 1))
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1
        print("step #: {:}".format(n))
        print("time:   {:}".format(problem.time))
        print("dt:     {:}".format(problem.time_stepper.dt))
        print("|dudt|: {:}".format(dudtnorm))
    n += 1
    # perform timestep
    problem.time_step()
    # calculate the new norm
    dudtnorm = np.linalg.norm(problem.rhs(problem.u))
    # catch divergent solutions
    if np.max(problem.u) > 1e12:
        break
Profiler.print_summary()
