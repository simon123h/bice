#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation, FiniteElementEquation
from bice.time_steppers import RungeKuttaFehlberg45, RungeKutta4, BDF2, BDF
from bice.constraints import *
from bice.solvers import *


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

        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N, endpoint=False)]
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        self.L = L
        # initial condition
        h0 = 5
        a = 3/20. / (h0-1)
        self.u = np.maximum(-a*self.x[0]*self.x[0] + h0, 1)
        # build finite element matrices
        self.build_FEM_matrices()

    # definition of the equation, using finite difference method
    def rhs(self, h):
        return -np.matmul(self.laplace, h) - np.matmul(self.M, self.djp(h))

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    # no dealiasing for the FD version
    def dealias(self, u, real_space=False, ratio=1./2.):
        return u

    def first_spatial_derivative(self, u):
        return np.matmul(self.nabla, u)


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
        # initialize time stepper
        # self.time_stepper = RungeKutta4()
        # self.time_stepper = RungeKuttaFehlberg45()
        # self.time_stepper.error_tolerance = 1e1
        # self.time_stepper.dt = 3e-5
        self.time_stepper = BDF(self)  # better for FD
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
problem = ThinFilm(N=1024, L=100)

# Impose the constraints
problem.volume_constraint.fixed_volume = np.trapz(
    problem.tfe.u, problem.tfe.x[0])
problem.add_equation(problem.volume_constraint)
problem.add_equation(problem.translation_constraint)


# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0


plotID = 0
problem.plot(ax)
fig.savefig("out/img/{:05d}.svg".format(plotID))
plotID += 1

# problem.newton_solver = MyNewtonSolver()
# problem.newton_solver.convergence_tolerance = 1e-6

problem.newton_solve()

problem.plot(ax)
fig.savefig("out/img/{:05d}.svg".format(plotID))

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
#             fig.savefig("out/img/{:05d}.svg".format(plotID))
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
#         fig.savefig("out/img/{:05d}.svg".format(plotID))
#         plotID += 1
