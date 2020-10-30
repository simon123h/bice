#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from acPFC1d import acPFCProblem
from bice import time_steppers
from bice.measure import LyapunovExponentCalculator


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = acPFCProblem(N=256, L=16*np.pi)

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1
T = 1000.
if not os.path.exists("initial_state.dat"):
    while problem.time < T:
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
            print("diverged")
            break
    # save the state, so we can reload it later
    problem.save("initial_state.dat")
else:
    # load the initial state
    problem.load("initial_state.dat")

# calculate Lyapunov exponents
problem.time_stepper = time_steppers.BDF2(dt=0.1)
lyapunov = LyapunovExponentCalculator(
    problem, nexponents=10, epsilon=1e-6, nintegration_steps=1)

last10 = np.zeros((10, 10))

while True:
    lyapunov.step()
    problem.dealias()
    last10[:-1] = last10[1:]
    last10[-1] = lyapunov.exponents
    ax.clear()
    ccount = 0.
    for exponents in last10:
        ax.plot(exponents, marker='.', color='{:.1f}'.format(1 - ccount/10.), ls='')
        ccount += 1
    ax.plot(lyapunov.exponents, marker="o")
    fig.savefig("out/img/{:05d}.svg".format(plotID))
    plotID += 1
    print("Lyapunov exponents:", lyapunov.exponents)
