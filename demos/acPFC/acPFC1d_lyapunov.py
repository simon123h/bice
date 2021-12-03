#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from acPFC1d import acPFCProblem
from bice import time_steppers
from bice.measure import LyapunovExponentCalculator
from bice import profile, Profiler


# create output folder

filepath = "/local0/m_holl20/biceresults/acPFC_lyapunov_phi01-062/"
shutil.rmtree(filepath + "out", ignore_errors=True)
os.makedirs(filepath + "out/img", exist_ok=True)

# create problem
problem = acPFCProblem(N=256, L=16*np.pi)
problem.acpfc.phi01 = -0.62

# create figure
fig = plt.figure(figsize=(16, 9))
ax_sol = fig.add_subplot(121)
ax_largest = fig.add_subplot(122)
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1

T = 1000.
if not os.path.exists("initial_state.npz"):
    while problem.time < T:
        # plot
        if n % plotevery == 0:
            problem.plot(ax_sol)
            fig.savefig(f"out/img/{plotID:05d}.svg")
            plotID += 1
            print(f"step #: {n}")
            print(f"time:   {problem.time}")
            print(f"dt:     {problem.time_stepper.dt}")
            print(f"|dudt|: {dudtnorm}")
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
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")


# calculate Lyapunov exponents
problem.time_stepper = time_steppers.BDF2(dt=0.1)
lyapunov = LyapunovExponentCalculator(
    problem, nexponents=1, epsilon=1e-6, nintegration_steps=1)

last10 = []
largest = []
L2norms = [problem.norm()]
times = [problem.time()]


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
    fig.savefig(filepath + f"out/img/{plotID:07d}.svg")
    plotID += 1
    print("Lyapunov exponent(s):", lyapunov.exponents)
    print("L2-norm: ", L2norms[-1])
    np.savetxt(filepath[:-1] + "_exponents.dat", largest)
    np.savetxt(filepath[:-1] + "_L2norms.dat", L2norms)
    np.savetxt(filepath[:-1] + "_times.dat", times)
