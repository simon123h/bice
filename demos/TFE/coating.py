#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import findiff
import matplotlib.pyplot as plt
import scipy.sparse as sp
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.pde.finite_differences import NeumannBC, DirichletBC, RobinBC, NoBoundaryConditions
from bice import profile, Profiler
from bice.core.solvers import NewtonKrylovSolver, MyNewtonSolver


class CoatingEquation(FiniteDifferencesEquation):
    r"""
     Finite differences implementation of the 1-dimensional coating problem
     """

    def __init__(self, N, L):
        super().__init__(shape=(N,))
        # parameters:
        self.U = 0.5  # substrate velocity
        self.q = 0.1  # influx
        self.h_p = 0.04
        self.theta = np.sqrt(0.6)
        print("h_LL =", self.q/self.U)
        # setup the mesh
        self.L = L
        self.x = [np.linspace(0, L, N)]
        # initial condition
        x = self.x[0]
        incl = 2
        hLL = self.q / self.U
        h1 = 1-incl*(x[1] - x[0])
        self.u = (h1 - hLL) * (1 + np.tanh(-incl * x)) + hLL
        # self.u = np.ones(N)
        # self.u = np.maximum(1 - 0.0*x, self.h_p)
        # build finite differences matrices
        self.build_FD_matrices(approx_order=1)

    # overload building of FD matrices, because this equation has a more complicated set up
    def build_FD_matrices(self, approx_order):
        # build finite differences matrices...
        # (i) including the flux boundary conditions for h^3 * dF/dh
        self.bc = DirichletBC(vals=(1, 0))
        super().build_FD_matrices(approx_order)
        self.nabla_F = self.nabla
        # (ii) including the mixed boundary conditions for h (left Dirichlet, right Neumann)
        self.bc = RobinBC(a=(1, 0), b=(0, 1), c=(1, 0))
        super().build_FD_matrices(approx_order)
        self.nabla_h = self.nabla
        self.laplace_h = self.laplace
        # (iii) differentiation operators no specific boundary effects
        self.bc = NoBoundaryConditions()
        super().build_FD_matrices(approx_order)
        self.nabla0 = self.nabla

    # definition of the equation

    def rhs(self, h):
        h3 = h**3
        # disjoining pressure
        djp = 5/3*(self.theta*self.h_p)**2 * (self.h_p**3/h3**2 - 1./h3)
        # free energy variation
        dFdh = -self.laplace_h(h) - djp
        # bulk flux
        flux = h3 * self.nabla0.dot(dFdh)
        # boundary flux
        j_in = self.U-self.q
        # dynamics equation, scale boundary condition with j_in
        dhdt = self.nabla_F(flux, j_in)
        # advection term
        dhdt -= self.U * self.nabla_h.dot(h)
        return dhdt

    def jacobian2(self, h):
        # disjoining pressure
        h3 = h**3
        djp = 5/3*(self.theta*self.h_p)**2 * (self.h_p**3/h3**2 - 1./h3)
        ddjpdh = 5/3*(self.theta*self.h_p)**2 * \
            sp.diags(3.*self.h_p**3/h**4 - 6./h**7)
        # free energy variation
        dFdh = -self.laplace_h(h) - djp
        ddFdhdh = -self.laplace_h() - ddjpdh
        # d(Qh^3*nabla*dFdh)/dh
        flux = sp.diags(3*h**2 * self.nabla0.dot(dFdh)) + \
            sp.diags(h3) * self.nabla0.dot(ddFdhdh)
        # dynamics equation, boundary condition is a constant --> scale with zero
        jac = self.nabla_F(flux, 0)
        jac -= self.U * self.nabla_h()
        return jac

    def du_dx(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        global problem
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        x = self.x[0]
        h = self.u
        # add left ghost point (Dirichlet BC)
        x = np.concatenate(([x[0]-x[1]], x))
        h = np.concatenate(([1], h))
        x -= self.U*problem.time
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0, 1.1*np.max(h))
        ax.plot(x, h, marker="x")


class Coating(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = CoatingEquation(N, L)
        self.add_equation(self.tfe)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF2(dt=1)
        # self.time_stepper = time_steppers.ImplicitEuler(dt=1e-2)
        # self.time_stepper = time_steppers.BDF(self)
        # self.newton_solver = MyNewtonSolver()
        # self.newton_solver.convergence_tolerance = 1e-2
        # self.newton_solver.max_newton_iterations = 100
        # self.newton_solver.verbosity = 0

    def norm(self):
        return np.trapz(self.tfe.u, self.tfe.x[0])


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = Coating(N=100, L=10)

# create figure
fig, ax = plt.subplots(1, figsize=(16, 9))
plotID = 0

Profiler.start()

# time-stepping
n = 0
plotevery = 20
dudtnorm = 1
if not os.path.exists("initial_state_coating.dat"):
    while dudtnorm > 1e-5:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig("out/img/{:05d}.png".format(plotID))
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
            print("Aborted.")
            break
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state_coating.dat")
else:
    # load the initial state
    problem.load("initial_state_coating.dat")

plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.convergence_tolerance = 1e-10
problem.continuation_stepper.max_newton_iterations = 100
problem.continuation_parameter = (problem.tfe, "q")
problem.settings.neigs = 10

n = 0
plotevery = 1
while problem.tfe.h_p < problem.tfe.q / problem.tfe.U < 1:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1

Profiler.print_summary()
