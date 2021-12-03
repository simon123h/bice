#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from bice import Profiler
from adaptive_problem import AdaptiveSubstrateProblem


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = AdaptiveSubstrateProblem(N=512, L=6)
problem.tfe.U = -0.05
# problem.tfe.M = 4.5e-4
problem.tfe.M = 1e-4
# problem.tfe.M = 4.4e-4 # for stip-slick motion
problem.continuation_parameter = (problem.tfe, "M")

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

Profiler.start()

# time-stepping
n = 0
plotevery = 100
if not os.path.exists("initial_state.npz"):
    while problem.time_stepper.dt < 1e12 and problem.time < 1000:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig(f"out/img/{plotID:05d}.svg")
            plotID += 1
        print("step #:{:6d},  time:{:9.2e},  dt:{:9.2e},  norm:{:9.2e}".format(
            n, problem.time, problem.time_stepper.dt, problem.norm()))
        n += 1
        # perform timestep
        problem.time_step()
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

# start parameter continuation
problem.continuation_stepper.ds = 1e-3
problem.continuation_stepper.ndesired_newton_steps = 3
# problem.continuation_stepper.convergence_tolerance = 1e-8
# problem.continuation_stepper.max_newton_iterations = 30
problem.settings.neigs = 50

# Impose the constraint
problem.add_equation(problem.volume_constraint)

n = 0
plotevery = 1
while True:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig(f"out/img/{plotID:05d}.svg")
        plotID += 1
    # save bifurcation points
    if problem.bifurcation_diagram.current_solution().is_bifurcation():
        problem.save(f"sav/sol{n}.npz")
    # mesh refinement
    # problem.adapt()

problem.save("final_state.npz")
