#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem
from bice.fem import FiniteElementEquation, OneDimMesh
from bice.continuation import VolumeConstraint, TranslationConstraint
from bice import profile, Profiler


class ThinFilmEquation(FiniteElementEquation):
    r"""
     Finite element implementation of the (steady) 1-dimensional Thin-Film Equation
     equation
     dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
     with dh/dt = 0 and integrated twice.
     with the disjoining pressure:
     Pi(h) = 1/h^3 - 1/h^6
     """

    def __init__(self, N, L):
        super().__init__(N)
        # parameters: none
        # setup the mesh
        self.L = L
        self.mesh = OneDimMesh(N, L, -L/2)
        # initial condition
        h0 = 6
        a = 3/20. / (h0-1)
        self.u = np.maximum(-a*self.x[0]*self.x[0] + h0, 1)
        # build finite element matrices
        self.build_FEM_matrices()

    # definition of the residual integrand
    def residual_def(self, x, u, dudx, test, dtestdx):
        return dudx[0]*dtestdx[0] - self.djp(u)

    # definition of the equation, using finite element method
    def rhs(self, u):
        # call residual assembly loop
        # return self.assemble_residuals(self.residual_def, u)
        return -self.laplace.dot(u) - self.M.dot(self.djp(u))

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    # no dealiasing for the FD version
    def dealias(self, u, real_space=False, ratio=1./2.):
        return u

    def first_spatial_derivative(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        ax.plot(self.x[0], self.u, marker="x", label="solution")
        ax.legend()


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)
        # assign the continuation parameter
        self.continuation_parameter = (self.volume_constraint, "fixed_volume")

    # set higher modes to null, for numerical stability
    def dealias(self, fraction=1./2.):
        self.tfe.u = self.tfe.dealias(self.tfe.u, True)

    def norm(self):
        return np.trapz(self.tfe.u, self.tfe.x[0])


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=200, L=100)

# Impose the constraints
# problem.volume_constraint.fixed_volume = np.trapz(
#     problem.tfe.u, problem.tfe.x[0])
problem.add_equation(problem.volume_constraint)
# problem.add_equation(problem.translation_constraint)

# refinement thresholds
problem.tfe.mesh.max_refinement_error = 1e-2
problem.tfe.mesh.min_refinement_error = 1e-3
# problem.tfe.mesh.min_element_dx = 0.2
# problem.tfe.mesh.max_element_dx = 2

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# plot
problem.tfe.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
ax.clear()
plotID += 1

Profiler.start()

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


Profiler.print_summary()
