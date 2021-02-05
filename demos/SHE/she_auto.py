#!/usr/bin/python3
"""
Continuation of localized states in the 1d Swift-Hohenberg Equation using Finite Differences as
spatial discretization and Pseudo-Arclength Continuation and automated bifurcation diagram
generation.
"""
import matplotlib
import shutil
import os
import matplotlib.pyplot as plt
from she_fd import SwiftHohenbergProblem, TranslationConstraint
import numpy as np
from bice import Profiler

# figures won't steal window focus if the right backend is chosen
# matplotlib.use("QT5Agg")
matplotlib.use("Tkagg")

# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = SwiftHohenbergProblem(N=256, L=240)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16*0.6, 9*0.6))
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

# start parameter continuation
problem.settings.always_locate_bifurcations = True
problem.settings.neigs = 6

constraint = TranslationConstraint(problem.she)
problem.add_equation(constraint)

Profiler.start()

# automatically generate bifurcation diagram
problem.generate_bifurcation_diagram(parameter_lims=(-0.016, -0.012),
                                     max_recursion=1,
                                     max_steps=1e3,
                                     ax=ax,
                                     plotevery=60)


Profiler.print_summary()
