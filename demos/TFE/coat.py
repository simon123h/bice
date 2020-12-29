#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.pde.finite_differences import RobinBC, NeumannBC, DirichletBC
from bice.continuation import VolumeConstraint, TranslationConstraint
from bice import profile, Profiler
from bice.core.solvers import NewtonKrylovSolver, MyNewtonSolver


class ThinFilmEquation(FiniteDifferencesEquation):
    r"""
     Finite differences implementation of the 1-dimensional Thin-Film Equation
     equation
     dh/dt = d/dx (h^3 d/dx dF/dh )
     with the variation of the free energy:
     dF/dh = -d^2/dx^2 h - Pi(h)
     and the disjoining pressure:
     Pi(h) = 1/h^3 - 1/h^6
     """

    def __init__(self, N, L):
        super().__init__(shape=(2, N))
        # parameters:
        self.U = 0.1  # substrate velocity
        self.q = 0  # influx
        # setup the mesh
        self.L = L
        self.x = [np.linspace(0, L, N)]
        # initial condition
        h0 = 6
        a = 3/20. / (h0-1)
        x = self.x[0]
        self.u[0] = 20 + 0*x
        # self.u[0] = np.maximum(-a*x**2 + h0, 1)
        # build finite differences matrices
        self.bc_F = RobinBC(a=(0, 0), b=(1, 1), c=(0, -1e-2))
        self.build_FD_matrices(boundary_conditions=self.bc_F, premultiply_bc=False)
        self.nabla_F = self.nabla
        self.laplace_F = self.laplace
        self.bc_h = RobinBC(a=(1, 0), b=(0, 1), c=(20, 0))
        self.build_FD_matrices(boundary_conditions=self.bc_h, premultiply_bc=False)
        self.nabla_h = self.nabla
        self.laplace_h = self.laplace
        self.bc_n = DirichletBC()
        self.build_FD_matrices(boundary_conditions=self.bc_n, premultiply_bc=True)


    # definition of the equation
    def rhs(self, u):
        h, dFdh = u
        # do boundary transformation
        h_pad = self.bc_h.pad(h)
        dFdh_pad = self.bc_F.pad(dFdh)
        # disjoining pressure
        h3 = h**3
        djp = 1./h3**2 - 1./h3
        # equations
        # TODO: fix boundary conditions for dFdh
        eq1 = self.nabla_F.dot(self.bc_F.pad(h3 * self.nabla.dot(dFdh)))
        eq1 -= self.U * self.nabla_h.dot(h_pad)
        eq2 = -self.laplace_h.dot(h_pad) - djp - dFdh
        return np.array([eq1, eq2])

    def jacobian2(self, u):
        h, dFdh = u
        ddjpdh = 3./h**4 - 6./h**7
        eq1dh = self.nabla.dot(sp.diags(3 * h**2 * self.nabla.dot(dFdh)))
        eq1dF = self.nabla.dot(sp.diags(h**3) * self.nabla)
        eq2dh = -self.laplace - sp.diags(ddjpdh)
        eq2dF = -self.ddx[0]
        return sp.bmat([[eq1dh, eq1dF], [eq2dh, eq2dF]])

    def du_dx(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        global problem
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        x = self.x[0] - self.U*problem.time
        h = self.u[0]
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0, 1.1*np.max(h))
        ax.plot(x, h)

    def mass_matrix(self):
        nvar, ngrid = self.shape
        dynamics = np.eye(nvar)
        dynamics[1, 1] = 0
        I = sp.eye(ngrid)
        return sp.kron(dynamics, I)


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe, variable=0)
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF2(dt=100)
        # self.time_stepper = time_steppers.ImplicitEuler(dt=1e-2)
        # self.time_stepper = time_steppers.BDF(self)
        # assign the continuation parameter
        self.continuation_parameter = (self.volume_constraint, "fixed_volume")
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
problem = ThinFilm(N=200, L=100)

# create figure
fig, ax = plt.subplots(1, figsize=(16, 9))
plotID = 0

Profiler.start()

# time-stepping
n = 0
plotevery = 1
dudtnorm = 1
if not os.path.exists("initial_state2.dat"):
    while dudtnorm > 1e-8:
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
    problem.save("initial_state.dat")
else:
    # load the initial state
    problem.load("initial_state.dat")

exit()


# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3

# Impose the constraints
problem.volume_constraint.fixed_volume = np.trapz(
    problem.tfe.u, problem.tfe.x[0])
problem.add_equation(problem.volume_constraint)
problem.add_equation(problem.translation_constraint)

problem.continuation_stepper.convergence_tolerance = 1e-10

n = 0
plotevery = 1
while problem.volume_constraint.fixed_volume < 1000:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    #print('largest EVs: ', problem.eigen_solver.latest_eigenvalues[:3])
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1

Profiler.print_summary()
