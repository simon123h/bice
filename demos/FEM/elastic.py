#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation
from bice.time_steppers import RungeKuttaFehlberg45, RungeKutta4, BDF2, BDF
from bice.constraints import *
from bice.solvers import *
from bice.fem import FiniteElementEquation, OneDimMesh


class ThinFilmEquation(FiniteElementEquation):
    r"""
     Finite element implementation of the 1-dimensional Thin-Film Equation
     equation
     dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
     with the disjoining pressure:
     Pi(h) = 1/h^3 - 1/h^6
     """

    def __init__(self, N, L):
        super().__init__()
        # parameters: none
        # setup the mesh
        self.L = L
        self.nvalue = 2
        self.mesh = OneDimMesh(N, L, -L/2)
        # initial condition
        h0 = 6
        a = 3/20. / (h0-1)
        for node in self.mesh.nodes:
            x = node.x[0]
            xi = 0
            h = np.maximum(-a*x*x + h0, 1)
            node.u = np.array([h, xi])
        self.copy_nodal_values_to_unknowns()
        # build finite element matrices
        self.build_FEM_matrices()

    # definition of the equation, using finite element method
    def rhs(self, u):
        self.copy_unknowns_to_nodal_values(u)
        h = self.nodal_values(0)
        xi = self.nodal_values(1)
        p = 10
        r1 = -self.laplace.dot(h+xi) - self.M.dot(self.djp(h)) - p
        r2 = -self.laplace.dot(h+xi) - 1. * \
            self.laplace.dot(xi) + 1. * self.M.dot(xi)
        return np.vstack((r1, r2)).T.flatten()

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    def first_spatial_derivative(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        h = self.nodal_values(0)
        xi = self.nodal_values(1)
        ax.plot(self.x[0], h+xi, marker="x", label="liquid")
        ax.plot(self.x[0], xi, marker="x", label="substrate")
        ax.legend()


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        self.volume_constraint.fixed_volume = 0
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=200, L=100)

# Impose the constraints
# problem.volume_constraint.fixed_volume = np.trapz(
#     problem.tfe.u, problem.tfe.x[0])
# problem.add_equation(problem.volume_constraint)
# problem.add_equation(problem.translation_constraint)

# refinement thresholds
problem.tfe.mesh.max_refinement_error = 1e-2
problem.tfe.mesh.min_refinement_error = 1e-3

# problem.newton_solver = MyNewtonSolver()
# problem.newton_solver.convergence_tolerance = 1e-6

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# plot
problem.tfe.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
ax.clear()
plotID += 1

for i in range(10):

    # solve
    print("solving")
    problem.newton_solve()
    # plot
    problem.tfe.plot(ax)
    fig.savefig("out/img/{:05d}.png".format(plotID))
    ax.clear()
    plotID += 1
    # adapt
    print("adapting")
    problem.tfe.adapt()
    problem.tfe.adapt()
    # plot
    problem.tfe.plot(ax)
    fig.savefig("out/img/{:05d}.png".format(plotID))
    ax.clear()
    plotID += 1
