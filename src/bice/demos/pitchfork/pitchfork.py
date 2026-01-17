r"""
Perturbed pitchfork problem

in parts stolen from BifurcationKit.jl's Example 1: "solving the perturbed pitchfork equation"

https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/gettingstarted/#Example-3:-continuing-periodic-orbits

"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bice import Equation, Problem

# figures won't steal window focus if the right backend is chosen
matplotlib.use("Tkagg")


class PitchforkEquation(Equation):

    def __init__(self):
        super().__init__()
        self.u = np.zeros(1)
        self.mu = 0.0

    def rhs(self, u):
        self.x = np.full(self.shape, self.mu)
        return self.mu + u - u**3 / 3.0

    def jacobian(self, u):
        return 1.0 - u**2


class Pitchfork(Problem):
    def __init__(self):
        super().__init__()
        self.pitchfork = PitchforkEquation()
        self.add_equation(self.pitchfork)
        self.continuation_parameter = (self.pitchfork, "mu")

    def norm(self):
        return np.linalg.norm(self.pitchfork.u)


problem = Pitchfork()

# some settings
problem.settings.neigs = 0
problem.continuation_stepper.ds = 1e-1

# compute bifurcation diagram
fig, ax = plt.subplots(2, 1)
problem.generate_bifurcation_diagram(
    parameter_lims=(-1, 1), norm_lims=(0, 2), max_recursion=0, ax=ax, plotevery=1
)


fig.savefig(Path(__file__).with_suffix(".png"))

problem.plot(ax)
plt.show(block=True)
