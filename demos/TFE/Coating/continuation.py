#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from coating_problem import CoatingProblem
from bice import Profiler

matplotlib.use("Tkagg")


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = CoatingProblem(N=100, L=2)

# create figure
fig, ax = plt.subplots(1, figsize=(16, 9))
plotID = 0

Profiler.start()

# time-stepping
n = 0
plotevery = 20
dudtnorm = 1
if not os.path.exists("initial_state.npz"):
    while dudtnorm > 1e-5:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig(f"out/img/{plotID:05d}.png")
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
            print("Aborted.")
            break
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(16*0.6, 9*0.6))

# start parameter continuation
problem.continuation_stepper.ds = -1e-4
problem.continuation_stepper.ds_max = 2e-3
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.convergence_tolerance = 1e-10
problem.continuation_stepper.max_newton_iterations = 100
problem.continuation_parameter = (problem.tfe, "q")
problem.settings.neigs = 10

h_p = problem.tfe.h_p
U = problem.tfe.U

# generate bifurcation diagram
# problem.bifurcation_diagram.xlim = (h_p*U, 0.05)
problem.generate_bifurcation_diagram(
    ax=ax,
    parameter_lims=(h_p * U, U),
    max_recursion=0,
    plotevery=50
)

print((h_p * U, U))
print(problem.get_continuation_parameter())

fig.savefig("out/bifurcation_diagram.png")

Profiler.print_summary()

# get Hopf bifurcations
hopfs = [bif for bif in problem.bifurcation_diagram.branches[0].bifurcations()
         if bif.bifurcation_type() == "HP"]
print("#HPs:", len(hopfs))

# store them to disk
for n, hopf in enumerate(hopfs):
    problem.u = hopf.u
    problem.set_continuation_parameter(hopf.p)
    problem.history.clear()
    problem.save(f"hopf_point_{n}.npz")
