#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from ne1d import NikolaevskiyProblem
from bice.continuation import TranslationConstraint, VolumeConstraint

# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = NikolaevskiyProblem(N=64)
problem.ne.r = 0.5
problem.ne.m = 1.1

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1
T = 100 / problem.ne.r
if not os.path.exists("initial_state.npz"):
    while problem.time < T:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig(f"out/img/{plotID:05d}.svg")
            plotID += 1
            print(f"step #: {n}")
            print(f"time:   {problem.time}")
            print(f"dt:     {problem.time_stepper.dt}")
            print(f"|dudt|: {dudtnorm}")
        n += 1
        # perform timestep
        problem.time_step()
        # perform dealiasing
        problem.dealias()
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            print("diverged")
            break
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.settings.neigs = 10

# add constraints
volume_constraint = VolumeConstraint(problem.ne)
problem.add_equation(volume_constraint)
translation_constraint = TranslationConstraint(problem.ne)
problem.add_equation(translation_constraint)

# create new figure
plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

n = 0
plotevery = 10
while problem.ne.m > 0:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig(f"out/img/{plotID:05d}.svg")
        plotID += 1
