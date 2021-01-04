#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.continuation import VolumeConstraint, TranslationConstraint
from bice.core.profiling import Profiler


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

    # definition of the equation, using finite difference method
    def rhs(self, h):
        djp = 1./h**6 - 1./h**3
        dFdh = -self.laplace.dot(h) - djp
        return self.nabla.dot(h**3 * self.nabla.dot(dFdh))

    def jacobian(self, h):
        Q = diags(h**3)
        dQdh = diags(3 * h**2)
        djp = 1./h**6 - 1./h**3
        dFdh = diags(-self.laplace.dot(h) - djp)
        ddjpdh = diags(3./h**4 - 6./h**7)
        ddFdhdh = -self.laplace - ddjpdh
        return self.nabla.dot(dQdh * self.nabla.dot(dFdh) + Q * self.nabla.dot(ddFdhdh))

    def du_dx(self, u, direction=0):
        return self.nabla(u)


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        #self.tfe = ThinFilmEquation(N, L)
        self.tfe = ThinFilmEquationFD(N, L)
        self.add_equation(self.tfe)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF(self)
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

Profiler.start()

# time-stepping
n = 0
plotevery = 999
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

Profiler.print_summary()

# save the state, so we can reload it later
problem.save("initial_state.npz")

# # load the initial state
# problem.load("initial_state.npz")

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
