#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from acPFC1d import acPFCProblem
from bice import time_steppers
from bice.measure import LyapunovExponentCalculator
from bice import profile, Profiler
import sys


# create output folder

phi01 = float(sys.argv[1])

filepath = "acPFC_lyapunov_phi01{:+01.4f}/".format(phi01).replace('.', '')
shutil.rmtree(filepath + "out", ignore_errors=True)
os.makedirs(filepath + "out/img", exist_ok=True)

# create problem
problem = acPFCProblem(N=256, L=16*np.pi)
problem.acpfc.phi01 = phi01

# create figure
fig = plt.figure(figsize=(16, 9))
ax_sol = fig.add_subplot(121)
ax_largest = fig.add_subplot(122)
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1

T = 10000.
while problem.time < T:
    # plot
    if n % plotevery == 0:
        problem.plot(ax_sol)
        fig.savefig(filepath + "out/img/{:07d}.svg".format(plotID))
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
        print("diverged")
        break
# save the state, so we can reload it later
#problem.save("initial_state.dat")


# calculate Lyapunov exponents
problem.time_stepper = time_steppers.BDF2(dt=0.01)
lyapunov = LyapunovExponentCalculator(problem, nexponents=1, epsilon=1e-6, nintegration_steps=1)

last10 = []
largest = []
L2norms = [problem.norm()]
times = [problem.time]


# Profiler.start()

n = 1
plotevery = 10
while True:
    # perform Lyapunov exponent calculation step
    lyapunov.step()
    # store last10 and largest Lyapunov exponents
    last10 = [lyapunov.exponents] + last10[:9]
    largest += [np.max(lyapunov.exponents)]

    L2norms += [problem.norm()]
    times += [problem.time]

    ax_largest.clear()
    ax_sol.clear()
    ccount = 0.
    problem.plot(ax_sol)
    ax_largest.plot(largest)
    ax_largest.set_xlabel('iterations')
    ax_largest.set_ylabel('largest lyapunov exponent')
    fig.savefig(filepath + "out/img/{:07d}.svg".format(plotID))
    plotID += 1
    print("Lyapunov exponent(s):", lyapunov.exponents)
    print("L2-norm: ", L2norms[-1])
    np.savetxt(filepath[:-1] + "_exponents.dat", largest)
    np.savetxt(filepath[:-1] + "_L2norms.dat", L2norms)
    np.savetxt(filepath[:-1] + "_times.dat", times)
