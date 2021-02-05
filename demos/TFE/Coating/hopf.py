#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from coating_problem import CoatingProblem
from bice import Profiler
from bice.continuation import TimePeriodicOrbitHandler

matplotlib.use("Tkagg")

# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = CoatingProblem(N=100, L=2)

Profiler.start()

fig, ax = plt.subplots(2, 2, figsize=(16*0.6, 9*0.6))

# start parameter continuation
problem.continuation_stepper.ds = -1e-4
problem.continuation_stepper.ds_max = 2e-3
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.convergence_tolerance = 1e-10
problem.continuation_stepper.max_newton_iterations = 100
problem.continuation_parameter = (problem.tfe, "q")
problem.settings.neigs = 10

# load the bifurcation point into the problem
problem.load("hopf_point_0.npz")
# attempt branch switching to new branch
converged = problem.switch_branch(locate=False)

# period length and number of timesteps
T = 6.34
Nt = 60

# create TimePeriodicOrbitHandler
orbitHandler = TimePeriodicOrbitHandler(problem.eq, T, Nt)
# add initial condition to the TimePeriodicOrbitHandler
# TODO: deflate steady solution?
uu = []
problem.time_stepper.dt = T / Nt
for i in range(Nt):
    uu.append(problem.tfe.u.copy())
    problem.time_step()
orbitHandler.u = np.append(uu, T)

problem.remove_equation(problem.eq)
problem.add_equation(orbitHandler)


Profiler.print_summary()
