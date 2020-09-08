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
        super().__init__(shape=(2, N))
        # parameters
        self.sigma = 0.1
        self.kappa = -2
        # setup the mesh
        self.L = L
        self.mesh = OneDimMesh(N, L, -L/2)
        # initial condition
        h0 = 60
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
        h, xi = u
        r1 = -self.laplace.dot(h+xi) - self.M.dot(self.djp(h))
        r2 = -self.laplace.dot(h+xi) - self.sigma * \
            self.laplace.dot(xi) + 10**self.kappa * self.M.dot(xi)
        return np.array([r1, r2])

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    def first_spatial_derivative(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        h, xi = self.u
        ax.plot(self.x[0], h+xi, label="liquid")
        ax.plot(self.x[0], xi, label="substrate")
        ax.legend()


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint_h = VolumeConstraint(self.tfe, variable=0)
        self.volume_constraint_xi = VolumeConstraint(self.tfe, variable=1)
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)
        # assign the continuation parameter
        self.continuation_parameter = (self.tfe, "kappa")

    def norm(self):
        h, xi = self.tfe.u
        return np.linalg.norm(h)

# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=200, L=1000)

# Impose the constraints
problem.add_equation(problem.volume_constraint_h)
problem.add_equation(problem.volume_constraint_xi)
problem.add_equation(problem.translation_constraint)

# refinement thresholds
problem.tfe.mesh.max_refinement_error = 1e-3
problem.tfe.mesh.min_refinement_error = 1e-4
problem.tfe.mesh.min_element_dx = 1e-12

# problem.newton_solver = MyNewtonSolver()
# problem.newton_solver.convergence_tolerance = 1e-6

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

# plot
problem.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
plotID += 1

for i in range(1):
    # solve
    print("solving")
    problem.newton_solve()
    # adapt
    print("adapting")
    problem.tfe.adapt()
    # plot
    print("plotting")
    problem.plot(ax)
    fig.savefig("out/img/{:05d}.png".format(plotID))
    plotID += 1



# start parameter continuation
problem.continuation_stepper.ds = -1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.always_check_eigenvalues = False
problem.always_locate_bifurcations = False
problem.neigs = 0


n = 0
plotevery = 10
while problem.tfe.kappa < 0:
    # perform continuation step
    problem.continuation_step()
    # problem.tfe.adapt()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
