#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, FiniteDifferenceEquation
from bice.time_steppers import RungeKuttaFehlberg45, ImplicitEuler, RungeKutta4


class ThinFilm(Problem, FiniteDifferenceEquation):
    """
    Finite difference implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.fixed_volume = 0
        # impose a volume constraint?
        self.volume_constraint = False
        # space and fourier space
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initial condition
        self.u = np.ones(N) * 3
        self.u = 2 * np.cos(self.x*2*np.pi/L) + 3
        # self.u = np.maximum(10 * np.cos(self.x / 5), 1)
        # initialize time stepper
        self.time_stepper = RungeKutta4()
        # self.time_stepper = RungeKuttaFehlberg45()
        self.time_stepper.error_tolerance = 1e-1
        self.time_stepper.dt = 4e-5
        # plotting
        self.plotID = 0
        # build finite difference matrices
        self.build_FD_matrices(N)

    # definition of the equation, using finite difference method
    def rhs(self, h):
        # definition of the TFE
        # dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
        dFdh = -np.matmul(self.laplace, h) - self.djp(h)
        res = np.matmul(self.nabla, h**3 * np.matmul(self.nabla, dFdh))
        # definition of the constraint
        if self.volume_constraint:
            h_old = self.u
            # this is a parametric constraint, parameter fixed_volume needs to be set accordingly
            res[0] = np.trapz(h, self.x) - self.fixed_volume
            # alternative constraint that does not depend on parameter would be:
            # res[0] = np.trapz(h-h_old, self.x)
        return res

    # The mass matrix determines the linear relation of the rhs to the temporal derivatives dudt
    def mass_matrix(self):
        mm = np.eye(self.dim)
        # if constraint is enabled, the constraint equation has no time evolution
        if self.volume_constraint:
            mm[0, 0] = 0
        return mm

    # disjoining pressure and derivatives
    def djp(self, h):
        return 1./h**6 - 1./h**3

    # return the value of the continuation parameter
    def get_continuation_parameter(self):
        return self.fixed_volume

    # set the value of the continuation parameter
    def set_continuation_parameter(self, v):
        self.fixed_volume = v

    # plot everything
    def plot(self, fig, ax, sol=None):
        h = self.u
        ax[0, 0].plot(self.x, h)
        ax[0, 0].set_xlabel("x")
        ax[0, 0].set_ylabel("solution h(x,t)")
        ax[0, 0].set_ylim((0, np.max(h)*1.1))
        if sol and len(sol.eigenvectors) > 0:
            ax[1, 0].plot(np.real(sol.eigenvectors[0]))
            ax[1, 0].set_ylabel("eigenvector")
        else:
            ax[1, 0].plot(self.k, np.abs(np.fft.rfft(h)))
            ax[1, 0].set_xlim((0, self.k[-1]))
            ax[1, 0].set_xlabel("k")
            ax[1, 0].set_ylabel("fourier spectrum h(k,t)")
            ax[1, 0].set_yscale("log")
        for branch in self.bifurcation_diagram.branches:
            r, norm = branch.data()
            ax[0, 1].plot(r, norm, "--", color="C0")
            r, norm = branch.data(only="stable")
            ax[0, 1].plot(r, norm, color="C0")
            r, norm = branch.data(only="bifurcations")
            ax[0, 1].plot(r, norm, "*", color="C2")
        ax[0, 1].plot(np.nan, np.nan, "*", color="C2", label="bifurcations")
        ax[0, 1].plot(self.fixed_volume, self.norm(),
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
        else:
            ax[1, 1].plot(problem.x, problem.rhs(problem.u), label="dh/dt")
            ax[1, 1].set_xlabel("x")
            ax[1, 1].set_ylabel("dh/dt")
        fig.savefig("out/img/{:05d}.svg".format(self.plotID))
        self.plotID += 1
        for a in ax.flatten():
            a.clear()


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=256, L=100)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

# time-stepping
n = 0
plotevery = 1000
dudtnorm = 1
if not os.path.exists("initial_state2.dat"):
    while dudtnorm > 1e-8:
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
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            print("Aborted.")
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

problem.volume_constraint = True
problem.fixed_volume = np.trapz(problem.u, problem.x)

# n = 0
# plotevery = 5
# while problem.r > -0.016:
#     # perform continuation step
#     sol = problem.continuation_step()
#     n += 1
#     print("step #:", n, " ds:", problem.continuation_stepper.ds)
#     # plot
#     if n % plotevery == 0:
#         problem.plot(fig, ax, sol)

# # continuation in reverse direction
# # load the initial state
# problem.new_branch()
# problem.load("initial_state.dat")
# problem.r = -0.013
# problem.continuation_stepper.ds = -1e-2
# while problem.r < -0.002:
#     # perform continuation step
#     sol = problem.continuation_step()
#     n += 1
#     print("step #:", n, " ds:", problem.continuation_stepper.ds)
#     # plot
#     if n % plotevery == 0:
#         problem.plot(fig, ax, sol)
