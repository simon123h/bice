#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation
from bice.constraints import *
from bice.solvers import *
from bice.time_steppers import *
from bice.fem import FiniteElementEquation, OneDimMesh
from bice.profiling import Profiler

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
        # parameters: none
        # setup the mesh
        self.L = L
        self.mesh = OneDimMesh(N, L, -L/2)
        # initial condition
        h0 = 6
        a = 3/20. / (h0-1)
        self.u[0] = np.maximum(-a*self.x[0]*self.x[0] + h0, 1)
        # build finite element matrices
        self.build_FEM_matrices()

    # definition of the residual integrand
    def residual_def(self, x, u, dudx, test, dtestdx):
        h, dFdh = u
        r1 = 0
        r2 = (self.djp(h) - dFdh)*test
        for d in range(self.mesh.dim):
            # TODO: check signs
            r1 += -h**3 * dudx[1, d] * dtestdx[d]
            r2 += -dudx[0, d] * dtestdx[d]
        return np.array([r2, r1])

    # definition of the equation, using finite element method
    def rhs(self, u):
        # call residual assembly loop
        return self.assemble_residuals(self.residual_def, u)
        # return -self.laplace.dot(u) - self.M.dot(self.djp(u))

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    def first_spatial_derivative(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        h, dFdh = self.u
        ax.plot(self.x[0], h, marker="x", label="solution")
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
        # initialize time stepper
        # self.time_stepper = RungeKutta4()
        # self.time_stepper = RungeKuttaFehlberg45()
        # self.time_stepper.error_tolerance = 1e1
        # self.time_stepper.dt = 3e-5
        self.time_stepper = BDF2(dt=1e-1)
        # self.time_stepper = BDF(self)
        # assign the continuation parameter
        self.continuation_parameter = (self.volume_constraint, "fixed_volume")

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
# problem.add_equation(problem.volume_constraint)
# problem.add_equation(problem.translation_constraint)

# refinement thresholds
problem.tfe.mesh.max_refinement_error = 1e-2
problem.tfe.mesh.min_refinement_error = 1e-3
# problem.tfe.mesh.min_element_dx = 0.2
# problem.tfe.mesh.max_element_dx = 2

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

problem.tfe.mesh.max_refinement_error = 1e-2
problem.tfe.mesh.min_refinement_error = 1e-3
problem.tfe.mesh.min_element_dx = 0.1
problem.tfe.mesh.max_element_dx = 1e10

Profiler.start()

dudtnorm = 999
n = 0
plotevery = 1
while dudtnorm > 1e-8:
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
        print("step #: {:}".format(n))
        print("time:   {:}".format(problem.time))
        print("dt:     {:}".format(problem.time_stepper.dt))
        print("|dudt|: {:}".format(dudtnorm))
    n += 1
    # perform timestep
    problem.time_step()
    # perform mesh adaption
    # problem.tfe.adapt()
    # calculate the new norm
    dudtnorm = np.linalg.norm(problem.rhs(problem.u))
    # catch divergent solutions
    if np.max(problem.u) > 1e12:
        print("Aborted.")
        break

Profiler.print_summary()