#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation, FiniteDifferenceEquation, PseudospectralEquation
from bice.time_steppers import RungeKutta4, RungeKuttaFehlberg45, BDF2
from bice.constraints import TranslationConstraint
from bice.profiling import Profiler, profile
from bice.bifurcations import BifurcationConstraint


class SwiftHohenbergEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.r = -0.013
        self.kc = 0.5
        self.v = 0.41
        self.g = 1
        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N)]
        self.build_kvectors()
        # initial condition
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * np.exp(-0.005 * self.x[0]**2)

    # definition of the SHE (right-hand side)
    @profile
    def rhs(self, u):
        u_k = np.fft.fft(u)
        return np.fft.ifft((self.r - (self.kc**2 - self.k[0]**2)**2) * u_k).real + self.v * u**2 - self.g * u**3


class SwiftHohenbergEquationFD(FiniteDifferenceEquation):
    r"""
    Finite difference implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.r = -0.013
        self.kc = 0.5
        self.v = 0.41
        self.g = 1
        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N)]
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initial condition
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * np.exp(-0.005 * self.x[0] ** 2)
        # build finite difference matrices
        self.build_FD_matrices()
        self.linear_op = (self.kc**2 + self.laplace)
        self.linear_op = self.r - np.matmul(self.linear_op, self.linear_op)

    # definition of the SHE (right-hand side)
    def rhs(self, u):
        return np.matmul(self.linear_op, u) + self.v * u**2 - self.g * u**3


class SwiftHohenbergProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.she = SwiftHohenbergEquation(N, L)
        # self.she = SwiftHohenbergEquationFD(N, L)
        self.add_equation(self.she)
        # initialize time stepper
        self.time_stepper = RungeKuttaFehlberg45(dt=1e-3)
        self.time_stepper.error_tolerance = 1e-7
        # self.time_stepper = BDF2(dt=1e-3) # better for FD
        # assign the continuation parameter
        self.continuation_parameter = (self.she, "r")

    # set higher modes to null, for numerical stability
    @profile
    def dealias(self, fraction=1./2.):
        u_k = np.fft.fft(self.she.u)
        N = len(u_k)
        k = int(N*fraction)
        u_k[k+1:-k] = 0
        self.she.u = np.fft.ifft(u_k).real


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = SwiftHohenbergProblem(N=512, L=240)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 1000
dudtnorm = 1
if not os.path.exists("initial_state.dat"):
    Profiler.start()
    while dudtnorm > 1e-5:
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
        # perform dealiasing
        #problem.dealias()
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            break
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state.dat")
else:
    # load the initial state
    problem.load("initial_state.dat")

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.always_check_eigenvalues = True
problem.neigs = 20

constraint = TranslationConstraint(problem.she)
problem.add_equation(constraint)

# continuation until first bifurcation
n = 0
plotevery = 1
while len(problem.bifurcation_diagram.current_branch().bifurcations()) < 1:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1


free_parameter = (problem.she, "kc")
bifurcation_constraint = BifurcationConstraint(problem.latest_eigenvectors[0], free_parameter)
problem.add_equation(bifurcation_constraint)
problem.continuation_stepper.always_check_eigenvalues = False


n = 0
plotevery = 5
Profiler.start()
while problem.she.r > -0.016:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1

Profiler.print_summary(nested=False)

# load the initial state and add extra dof for translation constraint
problem.remove_equation(constraint)
problem.load("initial_state.dat")
problem.add_equation(constraint)

# continuation in reverse direction
problem.new_branch()
problem.she.r = -0.013
problem.continuation_stepper.ds = -1e-2
while problem.she.r < -0.002:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
