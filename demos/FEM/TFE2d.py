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
from bice.fem import FiniteElementEquation, TriangleMesh


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
        self.mesh = TriangleMesh(N, N, L, L, -L/2, -L/2)
        # initial condition
        h0 = 5
        a = 3/20. / (h0-1)
        xsq = self.x[0]**2 + self.x[1]**2
        self.u = np.maximum(-a*xsq + h0, 1)
        # build finite element matrices
        self.build_FEM_matrices()

    # definition of the equation, using finite element method
    def rhs(self, h):
        # k = 2 * np.pi / self.L
        # sin = np.cos(k*self.x[0])
        # return self.laplace.dot(sin) - self.M.dot(h)
        return -self.laplace.dot(h) - self.M.dot(self.djp(h))

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    def first_spatial_derivative(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        x = self.x[0]
        y = self.x[1]
        h = self.u
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tricontourf(x, y, h, 256, cmap="coolwarm")
        ax.scatter(x, y, s=0.1, c="black")


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        # Generate the translation constraints
        self.translation_constraint_x = TranslationConstraint(self.tfe, 0)
        self.translation_constraint_y = TranslationConstraint(self.tfe, 1)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=50, L=40)

# Impose the constraints
problem.add_equation(problem.volume_constraint)
# problem.add_equation(problem.translation_constraint_x)
# problem.add_equation(problem.translation_constraint_y)


# refinement thresholds
problem.tfe.mesh.max_refinement_error = 1e-2
problem.tfe.mesh.min_refinement_error = 1e-3
# problem.tfe.mesh.min_element_dx = 0.2
# problem.tfe.mesh.max_element_dx = 2

# problem.newton_solver = MyNewtonSolver()
# problem.newton_solver.convergence_tolerance = 1e-6

# create figure
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
plotID = 0

# plot
problem.tfe.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
ax.clear()
plotID += 1

for i in range(10):

    # solve
    print("solving")
    # problem.newton_solve()
    # plot
    # problem.tfe.plot(ax)
    # fig.savefig("out/img/{:05d}.png".format(plotID))
    # ax.clear()
    # plotID += 1
    # adapt
    print("adapting")
    problem.tfe.adapt()
    # plot
    problem.tfe.plot(ax)
    fig.savefig("out/img/{:05d}.png".format(plotID))
    ax.clear()
    plotID += 1
