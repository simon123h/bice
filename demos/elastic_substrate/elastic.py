#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem
from bice.pde.finite_differences import FiniteDifferencesEquation, NeumannBC
from bice.continuation import TranslationConstraint, VolumeConstraint


class ThinFilmEquation(FiniteDifferencesEquation):
    r"""
     Finite differences implementation of the 1-dimensional Thin-Film Equation
     on an elastic substrate.
     """

    def __init__(self, N, L):
        super().__init__(shape=(2, N))
        # parameters
        self.sigma = 0.1
        self.kappa = -2
        # setup the mesh
        self.x = [np.linspace(0, L/2, N)]
        # initial condition
        h0 = 60
        a = 3/20. / (h0-1)
        x = self.x[0]
        h = np.maximum(-a*x*x + h0, 1)
        xi = h*0
        self.u = np.array([h, xi])
        # build finite differences matrices
        self.bc = NeumannBC()
        self.build_FD_matrices(approx_order=2)

    # definition of the equation, using finite element method
    def rhs(self, u):
        h, xi = u
        djp = 1./h**6 - 1./h**3
        # h equation
        eq1 = -self.laplace(h+xi) - djp
        # xi equation
        eq2 = -self.laplace(h+xi) - self.sigma * \
            self.laplace(xi) + 10**self.kappa * xi
        return np.array([eq1, eq2])

    # definition of the equation, using finite element method
    def jacobian(self, u):
        h, xi = u
        ddjp_dh = 3./h**4 - 6./h**7
        deq1_dh = -self.laplace() - sp.diags(ddjp_dh)
        deq2_dh = -self.laplace()
        deq1_dxi = -self.laplace()
        deq2_dxi = -self.laplace() - self.sigma * self.laplace() + sp.diags(10**self.kappa+xi*0)
        return sp.bmat([[deq1_dh, deq1_dxi], [deq2_dh, deq2_dxi]])

    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        h, xi = self.u
        ax.plot(self.x[0], h+xi, marker="+", markersize=5, label="liquid")
        ax.plot(self.x[0], xi, marker="+", markersize=5, label="substrate")
        ax.legend()


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint_h = VolumeConstraint(self.tfe, variable=0)
        self.volume_constraint_xi = VolumeConstraint(self.tfe, variable=1)
        # assign the continuation parameter
        self.continuation_parameter = (self.tfe, "kappa")

    def norm(self):
        # calculate the L2-norm by integration
        h, xi = self.tfe.u
        x = self.tfe.x[0]
        L = np.max(x) - np.min(x)
        return np.trapz(h**2, x) / L


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=200, L=1000)

# impose the constraints
problem.add_equation(problem.volume_constraint_h)
problem.add_equation(problem.volume_constraint_xi)

# mesh refinement settings
problem.tfe.max_refinement_error = 1e-1
problem.tfe.min_refinement_error = 1e-2
problem.tfe.min_dx = 1

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# plot initial condition
problem.tfe.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
ax.clear()
plotID += 1

# do some solving
for i in range(5):
    # solve
    print("solving")
    problem.newton_solve()
    # adapt
    print("adapting")
    problem.tfe.adapt()
    # plot
    print("plotting")
    problem.tfe.plot(ax)
    fig.savefig("out/img/{:05d}.png".format(plotID))
    ax.clear()
    plotID += 1

# create new figure
plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

# start parameter continuation
print("starting continuation")
problem.continuation_stepper.ds = -1e-2
problem.settings.always_locate_bifurcations = False
problem.settings.neigs = 0

n = 0
plotevery = 10
while problem.tfe.kappa < 0:
    # perform continuation step
    problem.continuation_step()
    problem.tfe.adapt()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds,
          " kappa:", problem.tfe.kappa)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
