#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import CollocationEquation
from bice.continuation import TranslationConstraint
from bice import profile, Profiler


class SwiftHohenbergEquation(CollocationEquation):
    r"""
    Collocation method implementation of the 1-dimensional Swift-Hohenberg Equation
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
        # space coordinates
        self.x = [np.linspace(-L/2, L/2, N)]
        # initial condition
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * \
            np.exp(-0.005 * self.x[0] ** 2)
        # convert to polynomial coefficients
        self.u = self.u2poly(self.u)
        print(self.u)
        # build spatial derivative matrices
        self.build_ddx_matrices()
        self.linear_op = (self.kc**2 + self.laplace)
        self.linear_op = self.r - np.matmul(self.linear_op, self.linear_op)

    # definition of the SHE (right-hand side)
    def rhs(self, u):
        # TODO: this is work in progress
        return np.matmul(self.linear_op, u) + self.v * u**2 - self.g * u**3


class SwiftHohenbergProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.she = SwiftHohenbergEquation(N, L)
        self.add_equation(self.she)
        # initialize time stepper
        self.time_stepper = time_steppers.RungeKuttaFehlberg45(dt=1e-3)
        self.time_stepper.error_tolerance = 1e-7
        # self.time_stepper = time_steppers.BDF2(dt=1e-3) # better for FD
        # assign the continuation parameter
        self.continuation_parameter = (self.she, "r")

    # set higher modes to null, for numerical stability
    @profile
    def dealias(self, fraction=1./2.):
        return
        u_k = np.fft.rfft(self.she.u)
        N = len(u_k)
        k = int(N*fraction)
        u_k[k+1:-k] = 0
        self.she.u = np.fft.irfft(u_k)


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
problem.settings.always_locate_bifurcations = False
problem.settings.neigs = 20

constraint = TranslationConstraint(problem.she)
problem.add_equation(constraint)

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
