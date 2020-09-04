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
        # k = 4*np.pi/self.L
        # sin = np.cos(k*self.x[0])
        # return -np.matmul(self.laplace, sin) - np.matmul(self.M, h)
        return -self.laplace.dot(h) - self.M.dot(self.djp(h))

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    def plot(self, ax):
        x = self.x[0]
        y = self.x[1]
        h = self.u
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tricontourf(x, y, h, 256, cmap="coolwarm")

    def first_spatial_derivative(self, u, direction=0):
        return self.nabla[direction].dot(u)


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        self.volume_constraint.fixed_volume = 0
        # Generate the translation constraints
        self.translation_constraint_x = TranslationConstraint(self.tfe, 0)
        self.translation_constraint_y = TranslationConstraint(self.tfe, 1)
        # initialize time stepper
        # self.time_stepper = RungeKutta4()
        # self.time_stepper = RungeKuttaFehlberg45()
        # self.time_stepper.error_tolerance = 1e1
        # self.time_stepper.dt = 3e-5
        # self.time_stepper = BDF(self)  # better for FD
        # assign the continuation parameter
        # self.continuation_parameter = (self.volume_constraint, "fixed_volume")


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=40, L=40)

# Impose the constraints
problem.volume_constraint.fixed_volume = np.trapz(
    problem.tfe.u, problem.tfe.x[0])
problem.add_equation(problem.volume_constraint)
problem.add_equation(problem.translation_constraint_x)
problem.add_equation(problem.translation_constraint_y)


# create figure
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
plotID = 0


plotID = 0
problem.tfe.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
plotID += 1

problem.newton_solver = MyNewtonSolver()
problem.newton_solver.convergence_tolerance = 1e-6

# TODO: matrices become VERY large! At this point, we should really switch to using sparse matrices
problem.newton_solve()

problem.tfe.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))

exit()


# # time-stepping
# n = 0
# plotevery = 1
# dudtnorm = 1
# if not os.path.exists("initial_state.dat"):
#     while dudtnorm > 1e-8:
#         # plot
#         if n % plotevery == 0:
#             problem.plot(ax)
#             fig.savefig("out/img/{:05d}.png".format(plotID))
#             plotID += 1
#             print("step #: {:}".format(n))
#             print("time:   {:}".format(problem.time))
#             print("dt:     {:}".format(problem.time_stepper.dt))
#             print("|dudt|: {:}".format(dudtnorm))
#         n += 1
#         # perform timestep
#         problem.time_step()
#         # perform dealiasing
#         problem.dealias()
#         # calculate the new norm
#         dudtnorm = np.linalg.norm(problem.rhs(problem.u))
#         # catch divergent solutions
#         if np.max(problem.u) > 1e12:
#             print("Aborted.")
#             break
#     # save the state, so we can reload it later
#     problem.save("initial_state.dat")
# else:
#     # load the initial state
#     problem.load("initial_state.dat")

# # start parameter continuation
# problem.continuation_stepper.ds = 1e-2
# problem.continuation_stepper.ndesired_newton_steps = 3
# problem.always_check_eigenvalues = True

# # Impose the constraints
# problem.volume_constraint.fixed_volume = np.trapz(problem.tfe.u, problem.tfe.x[0])
# problem.add_equation(problem.volume_constraint)
# problem.add_equation(problem.translation_constraint)

# problem.continuation_stepper.convergence_tolerance = 1e-10

# n = 0
# plotevery = 1
# while problem.volume_constraint.fixed_volume < 1000:
#     # perform continuation step
#     problem.continuation_step()
#     # perform dealiasing
#     problem.dealias()
#     n += 1
#     print("step #:", n, " ds:", problem.continuation_stepper.ds)
#     #print('largest EVs: ', problem.latest_eigenvalues[:3])
#     # plot
#     if n % plotevery == 0:
#         problem.plot(ax)
#         fig.savefig("out/img/{:05d}.png".format(plotID))
#         plotID += 1
