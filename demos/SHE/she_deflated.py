#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import PseudospectralEquation
from bice.continuation import TranslationConstraint, DeflatedContinuation
from bice import profile, Profiler
from she import SwiftHohenbergEquation


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
problem.continuation_stepper.ds = 1e-3
problem.settings.always_locate_bifurcations = False
problem.settings.neigs = 20

constraint = TranslationConstraint(problem.she)
problem.add_equation(constraint)
problem.continuation_stepper = DeflatedContinuation()

n = 0
plotevery = 1
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
    print("step #:", n, " #sols:", len(problem.continuation_stepper.known_solutions))
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
