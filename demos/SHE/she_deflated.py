#!/usr/bin/python3
"""
Continuation of localized states in the 1d Swift-Hohenberg Equation using Finite Differences as
spatial discretization and Deflated Continuation instead of Pseudo-Arclength Continuation.
"""
import shutil
import os
import matplotlib.pyplot as plt
from she_fd import SwiftHohenbergProblem, TranslationConstraint
import sys
import numpy as np
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Profiler
from bice.continuation import DeflatedContinuation
from bice import MyNewtonSolver

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
dudtnorm = 1
while dudtnorm > 1e-5:
    n += 1
    # perform timestep
    problem.time_step()
    # calculate the new norm
    dudtnorm = np.linalg.norm(problem.rhs(problem.u))

# save the state, so we can reload it later
problem.save("initial_state.npz")

# start parameter continuation
problem.newton_solver = MyNewtonSolver()
# problem.newton_solver.convergence_tolerance = 1e-2
# problem.newton_solver.verbosity = 1
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
        problem.continuation_stepper.deflation.solutions))
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1

Profiler.print_summary(nested=False)

# load the initial state and add extra dof for translation constraint
problem.remove_equation(constraint)
problem.load("initial_state.npz")
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
        problem.continuation_stepper.deflation.solutions))
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1
