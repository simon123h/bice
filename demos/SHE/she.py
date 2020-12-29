#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
from scipy.sparse import diags
import scipy.sparse as sp
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde.finite_differences import FiniteDifferencesEquation, PeriodicBC
from bice.continuation import ConstraintEquation
from bice import profile, Profiler


class SwiftHohenbergEquation(FiniteDifferencesEquation):
    r"""
    Finite difference implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.r = -0.013
        self.kc = 0.5
        self.v = 0.41
        self.g = 1
        # spatial coordinate
        self.x = [np.linspace(-L/2, L/2, N)]
        # initial condition
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * \
            np.exp(-0.005 * self.x[0] ** 2)
        # build finite difference matrices
        self.bc = PeriodicBC()
        self.build_FD_matrices(approx_order=3)
        laplace = self.laplace()
        self.linear_op = -2 * self.kc**2 * laplace - laplace.dot(laplace)

    # definition of the SHE (right-hand side)
    def rhs(self, u):
        return self.linear_op.dot(u) + (self.r - self.kc**4) * u + self.v * u**2 - self.g * u**3

    # definition of the Jacobian
    @profile
    def jacobian(self, u):
        return self.linear_op + diags(self.r - self.kc**4 + self.v * 2 * u - self.g * 3 * u**2)


class SwiftHohenbergProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.she = SwiftHohenbergEquation(N, L)
        self.add_equation(self.she)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF(self)
        # assign the continuation parameter
        self.continuation_parameter = (self.she, "r")


class TranslationConstraint(ConstraintEquation):

    def __init__(self, reference_equation):
        # call parent constructor
        super().__init__(shape=(1,))
        # on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        # initialize unknowns (velocity vector) to zero
        self.u = np.zeros(1)

    def rhs(self, u):
        # set up the vector of the residual contributions
        res = np.zeros((u.size))
        # reference to the equation, shape and indices of the unknowns that we work on
        eq = self.ref_eq
        eq_idx = self.group.idx[eq]
        self_idx = self.group.idx[self]
        # obtain the unknowns
        eq_u = u[eq_idx]
        eq_u_old = self.group.u[eq_idx]
        velocity = u[self_idx]
        # add constraint to residuals of reference equation (velocity is the lagrange multiplier)
        res[eq_idx] = velocity * eq.nabla(eq_u)
        # add the constraint equation
        res[self_idx] = np.dot(eq.x[0], eq_u-eq_u_old)
        # res[self_idx] = np.dot(eq_dudx, (eq_u - eq_u_old))
        return res

    @profile
    def jacobian(self, u):
        # reference to the equation, shape and indices of the unknowns that we work on
        eq = self.ref_eq
        eq_idx = self.group.idx[eq]
        self_idx = self.group.idx[self]
        # obtain the unknowns
        eq_u = u[eq_idx]
        # eq_u_old = self.group.u[eq_idx]
        velocity = u[self_idx][0]
        # contribution of d(eq) / du
        deq_du = velocity * eq.nabla()
        # contribution of d(eq) / dvelocity
        deq_dv = eq.nabla(eq_u).reshape((eq_u.size, 1))
        # contribution of d(constraint) / du
        dcnstr_du = eq.x[0].reshape((1, eq_u.size))
        # contribution of d(constraint) / dvelocity
        dcnstr_dv = None
        # stack everything together and return
        return sp.bmat([[deq_du, deq_dv], [dcnstr_du, dcnstr_dv]])


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = SwiftHohenbergProblem(N=512, L=240)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 1000
dudtnorm = 1
if not os.path.exists("initial_state.dat"):
    Profiler.start()
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
            break
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state.dat")
else:
    # load the initial state
    problem.load("initial_state.dat")

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.settings.always_locate_bifurcations = False
problem.settings.neigs = 20

constraint = TranslationConstraint(problem.she)
problem.add_equation(constraint)

n = 0
plotevery = 20
Profiler.start()
while problem.she.r > -0.016:
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

# load the initial state and add extra dof for translation constraint
problem.remove_equation(constraint)
problem.load("initial_state.dat")
problem.add_equation(constraint)

# continuation in reverse direction
problem.new_branch()
problem.she.r = -0.013
problem.continuation_stepper.ds = -1e-2
while problem.she.r < -0.002:
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
