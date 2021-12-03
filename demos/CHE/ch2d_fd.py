#!/usr/bin/python3
import shutil
import os
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from bice import Problem, time_steppers
from bice.pde.finite_differences import FiniteDifferencesEquation, PeriodicBC
from bice.continuation import TranslationConstraint, VolumeConstraint
from bice import profile, Profiler


class CahnHilliardEquation(FiniteDifferencesEquation):
    r"""
    Finite difference implementation of the 2-dimensional Cahn-Hilliard Equation
    equation, a nonlinear PDE
    \partial t c &= \Delta (c^3 + a * c - \kappa * \Delta c)
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.a = -0.5
        self.kappa = 1.
        # list of spatial coordinate. list is important,
        # to deal with several dimensions with different discretization/lengths
        self.x = [np.linspace(-L/2, L/2, N), np.linspace(-L/2, L/2, N)]
        # build finite difference matrices
        self.bc = PeriodicBC()
        self.build_FD_matrices()
        # initial condition
        self.u = (np.random.random(N**2)-0.5)*0.02
        # mx, my = np.meshgrid(*self.x)
        # self.u = np.cos(np.sqrt(mx**2 + my**2)/(L/4)) - 0.1
        self.u = self.u.ravel()

    # definition of the CHE (right-hand side)
    @profile
    def rhs(self, u):
        Delta = self.laplace
        return Delta.dot(u**3 + self.a*u - self.kappa * Delta.dot(u))

    # definition of the Jacobian
    @profile
    def jacobian(self, u):
        Delta = self.laplace
        return Delta.dot(3*diags(u**2) - self.kappa * Delta) + self.a * Delta


class CahnHilliardProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Cahn-Hilliard equation to the problem
        self.che = CahnHilliardEquation(N, L)
        self.add_equation(self.che)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF(self)
        self.time_stepper.dt = 1e-3
        # assign the continuation parameter
        self.continuation_parameter = (self.che, "a")


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = CahnHilliardProblem(N=64, L=64)

# time-stepping
n = 0
plotID = 0
plotevery = 5
dudtnorm = 1
mx, my = np.meshgrid(problem.che.x[0], problem.che.x[1])

Profiler.start()

if not os.path.exists("initial_state2D.npz"):
    while dudtnorm > 1e-6:
        # plot
        if n % plotevery == 0:
            plt.cla()
            plt.pcolormesh(mx, my, problem.che.u.reshape(
                mx.shape), edgecolors='face')
            plt.colorbar()
            plt.savefig(f"out/img/{plotID:05d}.png")
            plt.close()
            # problem.plot(ax)
            # fig.savefig(f"out/img/{plotID:05d}.svg")
            plotID += 1
            print(f"step #: {n}")
            print(f"dt:     {problem.time_stepper.dt}")
            print(f"|dudt|: {dudtnorm}")

        n += 1
        # perform timestep
        problem.time_step()
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            break

    # save the state, so we can reload it later
    problem.save("initial_state2D.npz")
else:
    # load the initial state
    problem.load("initial_state2D.npz")

Profiler.print_summary()

# start parameter continuation
problem.continuation_stepper.ds = -1e-2
problem.continuation_stepper.ndesired_newton_steps = 3

volume_constraint = VolumeConstraint(problem.che)
problem.add_equation(volume_constraint)
translation_constraint_x = TranslationConstraint(problem.che, direction=0)
problem.add_equation(translation_constraint_x)
translation_constraint_y = TranslationConstraint(problem.che, direction=1)
problem.add_equation(translation_constraint_y)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

Profiler.start()

n = 0
plotevery = 1
while problem.che.a < 2 and n < 10:
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig(f"out/img/{plotID:05d}.png")
        plotID += 1
    # perform continuation step
    problem.continuation_step()
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    n += 1

Profiler.print_summary()
