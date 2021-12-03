#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from ne1d import NikolaevskiyProblem
from bice import time_steppers
from bice.measure import LyapunovExponentCalculator

# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = NikolaevskiyProblem(N=32)
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

# calculate Lyapunov exponents
problem.time_stepper = time_steppers.BDF2(dt=0.1)
lyapunov = LyapunovExponentCalculator(
    problem, nexponents=10, epsilon=1e-6, nintegration_steps=1)

while True:
    lyapunov.step()
    problem.dealias()
    ax.clear()
    ax.plot(lyapunov.exponents, marker="o")
    fig.savefig(f"out/img/{plotID:05d}.svg")
    plotID += 1
    print("Lyapunov exponents:", lyapunov.exponents)
