#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.pde.finite_differences import RobinBC, PeriodicBC, NeumannBC, DirichletBC, NoBoundaryConditions
from bice.continuation import VolumeConstraint, TranslationConstraint
from bice import profile, Profiler
from bice.core.solvers import NewtonSolver, MyNewtonSolver, NewtonKrylovSolver


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
        self.U = 0
        # setup the mesh
        self.L = L
        self.x = [np.linspace(-L/2, L/2, N, endpoint=False)]
        # initial condition
        h0 = 6
        a = 3/20. / (h0-1)
        x = self.x[0]
        self.u[0] = 2 + np.sin(4*np.pi*x/L)
        # self.u[0] = np.maximum(-a*x**2 + h0, 1)
        # build the boundary conditions
        self.bc = PeriodicBC()
        # self.bc = NeumannBC()
        # build finite differences matrices
        self.build_FD_matrices()

    # definition of the equation
    def rhs(self, u):
        h, dFdh = u
        h3 = h**3
        djp = 1./h3**2 - 1./h3
        eq1 = self.nabla(h3 * self.nabla(dFdh))
        eq2 = -self.laplace(h) - djp - dFdh
        return np.array([eq1, eq2])

    def jacobian(self, u):
        h, dFdh = u
        ddjpdh = 3./h**4 - 6./h**7
        eq1dh = self.nabla(sp.diags(3 * h**2 * self.nabla(dFdh)))
        eq1dF = self.nabla(sp.diags(h**3) * self.nabla())
        eq2dh = -self.laplace() - sp.diags(ddjpdh)
        eq2dF = -self.ddx[0]()
        return sp.bmat([[eq1dh, eq1dF], [eq2dh, eq2dF]])

    def du_dx(self, u, direction=0):
        return self.nabla[direction](u)

    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        h, dFdh = self.u
        ax.plot(self.x[0], h, marker="x", label="solution")
        ax.plot(self.x[0], dFdh, marker="x", label="dFdh")
        ax.legend()

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
        self.time_stepper = time_steppers.BDF2(dt=1e-1)
        # self.time_stepper = time_steppers.ImplicitEuler(dt=1e-2)
        # self.time_stepper = time_steppers.BDF(self)
        # assign the continuation parameter
        self.continuation_parameter = (self.volume_constraint, "fixed_volume")
        # self.newton_solver = MyNewtonSolver()
        # self.newton_solver = NewtonSolver()
        self.newton_solver = NewtonKrylovSolver()
        # self.newton_solver.approximate_jacobian = True
        self.newton_solver.convergence_tolerance = 6e-6
        self.newton_solver.max_newton_iterations = 100
        self.newton_solver.verbosity = 0

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
plotevery = 10
dudtnorm = 1
if not os.path.exists("initial_state2.dat"):
    while dudtnorm > 1e-8:
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
        # adapt the mesh
        if n % 10 == 0:
            problem.adapt()
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
        fig.savefig(f"out/img/{plotID:05d}.png")
        plotID += 1

Profiler.print_summary()
