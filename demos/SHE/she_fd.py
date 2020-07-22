#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, FiniteDifferenceEquation
from bice.time_steppers import RungeKuttaFehlberg45, ImplicitEuler, RungeKutta4


class SwiftHohenberg(Problem, FiniteDifferenceEquation):
    """
    Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
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
        # impose a translation constraint?
        self.translation_constraint = False
        # space and fourier space
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initial condition
        self.u = np.cos(2 * np.pi * self.x / 10) * np.exp(-0.005 * self.x**2)
        # initialize time stepper
        self.time_stepper = RungeKutta4()
        self.time_stepper.dt = 1e-4
        # plotting
        self.plotID = 0
        # build finite difference matrices
        self.build_FD_matrices(N)
        self.linear_op = (self.kc**2 + self.laplace)
        self.linear_op = self.r - np.matmul(self.linear_op, self.linear_op)

    # definition of the equation, using pseudospectral method
    def rhs(self, u):
        if self.translation_constraint:
            vel = u[-1]
            u = u[:-1]
        else:
            vel = 0
        # res
        res = np.zeros(self.dim)
        # definition of the SHE
        res[:u.size] = np.matmul(self.linear_op, u) + self.v * \
            u**2 - self.g * u**3

        # definition of the constraint
        if self.translation_constraint:
            # add advection term with velocity as lagrange multiplier (additional degree of freedom)
            res[:u.size] += vel*np.matmul(self.nabla, u)
            u_old = self.u[:-1]
            # this is the classical constraint: du/dx * du/dp = 0
            # res[u.size] = np.dot(
            #     np.fft.irfft(-1j*self.k*np.fft.rfft(u_old)), u-u_old)
            # this is the alternative center-of-mass constraint, requires less Fourier transforms :-)
            res[u.size] = np.dot(self.x, u-u_old)
        return res

    # The mass matrix determines the linear relation of the rhs to the temporal derivatives dudt
    def mass_matrix(self):
        mm = np.eye(self.dim)
        # if constraint is enabled, the constraint equation has no time evolution
        if self.translation_constraint:
            mm[-1, -1] = 0
        return mm

    # set higher modes to null, for numerical stability
    def dealias(self, fraction=1./2.):
        u_k = np.fft.rfft(self.u)
        N = len(u_k)
        u_k[-int(N*fraction):] = 0
        self.u = np.fft.irfft(u_k)

    # return the value of the continuation parameter
    def get_continuation_parameter(self):
        return self.r

    # set the value of the continuation parameter
    def set_continuation_parameter(self, v):
        self.r = v

    # plot everything
    def plot(self, fig, ax, sol=None):
        u = self.u
        if self.translation_constraint:
            u = u[:-1]
        ax[0, 0].plot(self.x, u)
        ax[0, 0].set_xlabel("x")
        ax[0, 0].set_ylabel("solution u(x,t)")
        if sol and len(sol.eigenvectors) > 0:
            ax[1, 0].plot(np.real(sol.eigenvectors[0]))
            ax[1, 0].set_ylabel("eigenvector")
        else:
            ax[1, 0].plot(self.k, np.abs(np.fft.rfft(u)))
            ax[1, 0].set_xlim((0, self.k[-1]/2))
            ax[1, 0].set_xlabel("k")
            ax[1, 0].set_ylabel("fourier spectrum u(k,t)")
        for branch in self.bifurcation_diagram.branches:
            r, norm = branch.data()
            ax[0, 1].plot(r, norm, "--", color="C0")
            r, norm = branch.data(only="stable")
            ax[0, 1].plot(r, norm, color="C0")
            r, norm = branch.data(only="bifurcations")
            ax[0, 1].plot(r, norm, "*", color="C2")
        ax[0, 1].plot(np.nan, np.nan, "*", color="C2", label="bifurcations")
        ax[0, 1].plot(self.r, self.norm(),
                      "x", label="current point", color="black")
        ax[0, 1].set_xlabel("parameter r")
        ax[0, 1].set_ylabel("L2-norm")
        ax[0, 1].legend()
        if sol:
            ev_re = np.real(sol.eigenvalues[:20])
            ev_re_n = np.ma.masked_where(
                ev_re > self.eigval_zero_tolerance, ev_re)
            ev_re_p = np.ma.masked_where(
                ev_re <= self.eigval_zero_tolerance, ev_re)
            ax[1, 1].plot(ev_re_n, "o", color="C0", label="Re < 0")
            ax[1, 1].plot(ev_re_p, "o", color="C1", label="Re > 0")
            ax[1, 1].axhline(0, color="gray")
            ax[1, 1].legend()
            ax[1, 1].set_ylabel("eigenvalues")
        fig.savefig("out/img/{:05d}.svg".format(self.plotID))
        self.plotID += 1
        for a in ax.flatten():
            a.clear()


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = SwiftHohenberg(N=512, L=240)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

# time-stepping
n = 0
plotevery = 1000
dudtnorm = 1
if not os.path.exists("initial_state2.dat"):
    while dudtnorm > 1e-5:
        # plot
        if n % plotevery == 0:
            problem.plot(fig, ax)
            print("step #: {:}".format(n))
            print("time:   {:}".format(problem.time))
            print("dt:     {:}".format(problem.time_stepper.dt))
            print("|dudt|: {:}".format(dudtnorm))
        n += 1
        # perform timestep
        problem.time_step()
        # perform dealiasing
        problem.dealias()
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            break
    # save the state, so we can reload it later
    problem.save("initial_state.dat")
else:
    # load the initial state
    problem.load("initial_state.dat")

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.always_check_eigenvalues = True

problem.translation_constraint = True
problem.u = np.append(problem.u, [0])

n = 0
plotevery = 5
while problem.r > -0.016:
    # perform continuation step
    sol = problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(fig, ax, sol)

# continuation in reverse direction
# load the initial state
problem.new_branch()
problem.load("initial_state.dat")
problem.r = -0.013
problem.continuation_stepper.ds = -1e-2
while problem.r < -0.002:
    # perform continuation step
    sol = problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(fig, ax, sol)
