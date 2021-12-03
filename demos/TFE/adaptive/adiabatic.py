#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from bice import Profiler, Solution
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
        print(
            f"step #:{n:6d},  time:{problem.time:9.2e},  dt:{problem.time_stepper.dt:9.2e},  norm:{problem.norm():9.2e}")
        n += 1
        # perform timestep
        problem.time_step()
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")


problem.time_stepper.factory_reset()
n = 0
plotevery = 1
while True:
    # perform continuation step
    for i in range(100):
        problem.time_step()
    n += 1
    vol = problem.tfe.liquid_volume()
    print(
        f"step  # : {n}, M: {problem.tfe.M}, dt: {problem.time_stepper.dt:9.2e}, vol: {vol}")
    problem.tfe.M = problem.tfe.M * 1.0001

    # add to branch
    branch = problem.bifurcation_diagram.active_branch
    # add the solution to the branch
    sol = Solution(problem)
    branch.add_solution_point(sol)
    # # solve the eigenproblem
    # eigenvalues, _ = problem.solve_eigenproblem()
    # # count number of positive eigenvalues
    # tol = problem.settings.eigval_zero_tolerance
    # sol.nunstable_eigenvalues = len(
    #     [ev for ev in eigenvalues if np.real(ev) > tol])
    # sol.nunstable_imaginary_eigenvalues = len(
    #     [ev for ev in eigenvalues if np.real(ev) > tol and abs(np.imag(ev)) > tol])

    # plot
    problem.plot(ax)
    fig.savefig(f"out/img/{plotID:05d}.svg")
    plotID += 1

problem.save("final_state.npz")
