#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem
from bice.fem import FiniteElementEquation, TriangleMesh
from bice.continuation import VolumeConstraint, TranslationConstraint
from bice import profile, Profiler


class ThinFilmEquation(FiniteElementEquation):
    r"""
     Finite element implementation of the (steady) 2-dimensional Thin-Film Equation
     equation
     dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
     with dh/dt = 0 and integrated twice.
     with the disjoining pressure:
     Pi(h) = 1/h^3 - 1/h^6
     """

    def __init__(self, N, L):
        super().__init__(N*N)
        # parameters: none
        # setup the mesh
        self.L = L
        self.mesh = TriangleMesh(N, N, L, L, -L/2, -L/2)
        # initial condition
        h0 = 5
        a = 3/20. / (h0-1)
        xsq = self.x[0]**2 + self.x[1]**2
        self.u = np.maximum(-a*xsq + h0, 1)
        # build finite element matrices
        self.build_FEM_matrices()

    # definition of the equation, using finite element method
    def rhs(self, h):
        return -self.laplace.dot(h) - self.M.dot(self.djp(h))

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    def du_dx(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        x = self.x[0]
        y = self.x[1]
        h = self.u
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # e = self.refinement_error_estimate()
        # e = np.minimum(e, self.mesh.max_refinement_error)
        # e = np.maximum(e, self.mesh.min_refinement_error)
        # ax.tricontourf(x, y, e, levels=np.linspace(0, 1.1*self.mesh.max_refinement_error, 256, endpoint=True), cmap="Reds")
        ax.tricontourf(x, y, h, 256, cmap="coolwarm")
        ax.scatter(x, y, s=0.4, c="black")

        # plot the mesh
        triangles = []
        for i, n in enumerate(self.mesh.nodes):
            n.index = i
        for element in self.mesh.elements:
            triangles.append([n.index for n in element.nodes])
        triangulation = matplotlib.tri.Triangulation(
            self.x[0], self.x[1], triangles)
        ax.triplot(triangulation, lw=0.2, color='black')


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        # Generate the translation constraints
        self.translation_constraint_x = TranslationConstraint(
            self.tfe, direction=0)
        self.translation_constraint_y = TranslationConstraint(
            self.tfe, direction=1)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=60, L=60)

# Impose the constraints
problem.add_equation(problem.volume_constraint)
problem.add_equation(problem.translation_constraint_x)
problem.add_equation(problem.translation_constraint_y)


# refinement thresholds
problem.tfe.mesh.max_refinement_error = 0.2
problem.tfe.mesh.min_refinement_error = 0.07
problem.tfe.mesh.min_element_dx = 0.5
problem.tfe.mesh.max_element_dx = 1e10

# create figure
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
plotID = 0

# plot
print("plotting")
problem.tfe.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
ax.clear()
plotID += 1

Profiler.start()
for i in range(30):

    # solve
    print("solving")
    problem.newton_solve()
    # plot
    print("plotting")
    problem.tfe.plot(ax)
    fig.savefig("out/img/{:05d}.png".format(plotID))
    ax.clear()
    plotID += 1
    # adapt
    print("adapting")
    problem.tfe.adapt()
    # plot
    print("plotting")
    problem.tfe.plot(ax)
    fig.savefig("out/img/{:05d}.png".format(plotID))
    ax.clear()
    plotID += 1

Profiler.print_summary()
