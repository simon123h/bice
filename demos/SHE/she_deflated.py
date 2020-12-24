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
from bice.continuation import TranslationConstraint, DeflatedContinuation
from bice import profile, Profiler
from bice import MyNewtonSolver


class SwiftHohenbergEquationFD(FiniteDifferencesEquation):
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
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * \
            np.exp(-0.005 * self.x[0] ** 2)
        # build finite difference matrices
        self.build_FD_matrices(sparse=True)
        laplace = self.laplace
        self.linear_op = -2 * self.kc**2 * laplace - laplace.dot(laplace)

    # definition of the SHE (right-hand side)
    def rhs(self, u):
        return (self.r - self.kc**4) * u + self.linear_op.dot(u) + self.v * u**2 - self.g * u**3

    # definition of the Jacobian
    def jacobian(self, u):
        return self.linear_op + diags(self.r - self.kc**4 + self.v * 2 * u - self.g * 3 * u**2)


class SwiftHohenbergProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.she = SwiftHohenbergEquationFD(N, L)
        self.add_equation(self.she)
        # initialize time stepper
        # self.time_stepper = time_steppers.RungeKuttaFehlberg45(dt=1e-3)
        # self.time_stepper.error_tolerance = 1e-7
        # self.time_stepper = time_steppers.BDF2(dt=1e-3) # better for FD
        self.time_stepper = time_steppers.BDF(self)
        # assign the continuation parameter
        self.continuation_parameter = (self.she, "r")


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
plotevery = 100
dudtnorm = 1
if not os.path.exists("initial_state.dat"):
    Profiler.start()
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
problem.newton_solver = MyNewtonSolver()
# problem.newton_solver.convergence_tolerance = 1e-2
# problem.newton_solver.verbosity = 1
# problem.newton_solver.method = "krylov"
problem.continuation_stepper = DeflatedContinuation()
problem.continuation_stepper.ds = 1e-5
# problem.continuation_stepper.max_solutions = 10
problem.settings.always_locate_bifurcations = False
problem.settings.neigs = 20

constraint = TranslationConstraint(problem.she)
problem.add_equation(constraint)

n = 0
plotevery = 1
Profiler.start()
while problem.she.r < -0.012:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " #solutions:", len(
        problem.continuation_stepper.known_solutions))
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
problem.continuation_stepper.ds = -1e-5
while problem.she.r > -0.016:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " #solutions:", len(
        problem.continuation_stepper.known_solutions))
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
