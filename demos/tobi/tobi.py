#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation, PseudospectralEquation
from bice.time_steppers import RungeKutta4, RungeKuttaFehlberg45, BDF2
from bice.constraints import TranslationConstraint
from bice.profiling import Profiler, profile


"""
IMPORTANT NOTE: this code is incomplete and should be removed
"""


class TobisEquation(Equation):

    def __init__(self):
        super().__init__()
        # parameters
        self.a = 0.
        self.T = 1
        self.minmax = "min"
        # initial condition
        self.u = np.array([4])

    def f(self, a, b):
        return (-self.T/2. + np.sqrt(self.T**2/4. + a))**2

    def g(self, a, b):
        return self.f(a, b)**2 / b**2

    # definition of the right-hand side
    def rhs(self, u):
        if self.minmax == "min":
            return min(self.f(self.a, u[0]), self.g(self.a, u[0])) + self.a
        return max(self.f(self.a, u[0]), self.g(self.a, u[0])) + self.a

    def mass_matrix(self):
        return np.zeros((1, 1))

    def plot(self, ax):
        pass


class TobisProblem(Problem):

    def __init__(self):
        super().__init__()
        # Add the equation to the problem
        self.tobis_eq = TobisEquation()
        self.add_equation(self.tobis_eq)
        self.continuation_parameter = (self.tobis_eq, "a")

    def norm(self):
        return self.tobis_eq.u[0]


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = TobisProblem()

# set parameters
problem.tobis_eq.minmax = "max"

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.always_locate_bifurcations = False
problem.neigs = 0

n = 0
plotevery = 1
while problem.tobis_eq.a < 1:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.bifurcation_diagram.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
