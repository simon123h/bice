#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.continuation import VolumeConstraint, TranslationConstraint


class ThinFilmEquationFD(FiniteDifferencesEquation):
    r"""
     Finite difference implementation of the 1-dimensional Thin-Film Equation
     equation
     dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
     with the disjoining pressure:
     Pi(h) = 1/h^3 - 1/h^6
     """

    def __init__(self, N, L):
        super().__init__(shape=N)
        # parameters: none
        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N)]
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initial condition
        self.u = 2 * np.cos(self.x[0] * 2 * np.pi / L) + 3
        # build finite difference matrices
        self.build_FD_matrices()
        # self.u = np.ones(N) * 3
        #self.x = self.x[0]
        # self.u = np.maximum(10 * np.cos(self.x / 5), 1)

    # definition of the equation, using finite difference method
    def rhs(self, h):
        djp = 1./h**6 - 1./h**3
        dFdh = -np.matmul(self.laplace, h) - djp
        return np.matmul(self.nabla, h**3 * np.matmul(self.nabla, dFdh))

    def du_dx(self, u, direction=0):
        return np.matmul(self.nabla, u)


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        #self.tfe = ThinFilmEquation(N, L)
        self.tfe = ThinFilmEquationFD(N, L)
        self.add_equation(self.tfe)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF(self)  # better for FD
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        self.volume_constraint.fixed_volume = 0
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)
        # assign the continuation parameter
        self.continuation_parameter = (self.volume_constraint, "fixed_volume")

    def norm(self):
        return np.trapz(self.tfe.u, self.tfe.x[0])


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=256, L=100)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 1
dudtnorm = 1
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
    # calculate the new norm
    dudtnorm = np.linalg.norm(problem.rhs(problem.u))

# save the state, so we can reload it later
problem.save("initial_state.dat")

exit()

# # load the initial state
# problem.load("initial_state.dat")

# # start parameter continuation
# problem.continuation_stepper.ds = 1e-2
# problem.continuation_stepper.ndesired_newton_steps = 3

# # Impose the constraints
# problem.volume_constraint.fixed_volume = np.trapz(
#     problem.tfe.u, problem.tfe.x[0])
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
#     #print('largest EVs: ', problem.eigen_solver.latest_eigenvalues[:3])
#     # plot
#     if n % plotevery == 0:
#         problem.plot(ax)
#         fig.savefig("out/img/{:05d}.svg".format(plotID))
#         plotID += 1
