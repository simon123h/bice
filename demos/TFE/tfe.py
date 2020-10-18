#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation, FiniteDifferenceEquation
from bice.time_steppers import RungeKuttaFehlberg45, RungeKutta4, BDF2, BDF
from bice.constraints import *


class ThinFilmEquation(Equation):
    r"""
    Pseudospectral implementation of the 1-dimensional Thin-Film Equation
    equation
    dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
    with the disjoining pressure:
    Pi(h) = 1/h^3 - 1/h^6
    """

    def __init__(self, N, L):
        super().__init__()
        # we have only a single variable h, so the shape is just (N,)
        # Note: self.shape = (1, N) would also be possible, it's a matter of taste
        self.shape = (N,)
        # parameters: none

        # space and fourier space
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initial condition
        # self.u = np.ones(N) * 3
        self.u = 2 * np.cos(self.x*2*np.pi/L) + 1
        # self.u = np.maximum(10 * np.cos(self.x / 5), 1)

    # definition of the equation, using pseudospectral method
    def rhs(self, h):
        h_k = np.fft.rfft(h)
        djp_k = self.dealias(np.fft.rfft(self.djp(h)))
        dhhh_dx = np.fft.irfft(self.dealias(
            self.dealias(np.fft.rfft(h**3)) * 1j * self.k))
        term1 = np.fft.irfft(self.dealias(
            1j * self.k * (-self.k**2 * h_k + djp_k)))
        term2 = np.fft.irfft(self.dealias(
            self.k**2 * (self.k**2 * h_k - djp_k)))
        return -dhhh_dx * term1 - h**3 * term2

    # disjoining pressure
    def djp(self, h):
        return 1./h**6 - 1./h**3

    # set higher modes to null, for numerical stability
    def dealias(self, u, real_space=False, ratio=1./2.):
        if real_space:
            u_k = np.fft.rfft(u)
        else:
            u_k = u
        k_F = (1-ratio) * self.k[-1]
        u_k *= np.exp(-36*(4. * self.k / 5. / k_F)**36)
        if real_space:
            return np.fft.irfft(u_k)
        return u_k

    def first_spatial_derivative(self, u, direction=0):
        du_dx = 1j*self.k*np.fft.rfft(u)
        return np.fft.irfft(du_dx)


class ThinFilmEquationFD(FiniteDifferenceEquation):
    r"""
     Finite difference implementation of the 1-dimensional Thin-Film Equation
     equation
     dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
     with the disjoining pressure:
     Pi(h) = 1/h^3 - 1/h^6
     """

    def __init__(self, N, L):
        super().__init__()
        # we have only a single variable h, so the shape is just (N,)
        # Note: self.shape = (1, N) would also be possible, it's a matter of taste
        self.shape = (N,)

        # parameters: none

        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N)]
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initial condition
        self.u = 2 * np.cos(self.x[0] * 2 * np.pi / L) + 3
        # build finite difference matrices
        self.build_FD_matrices()
        # self.u = np.ones(N) * 3
        #self.x = self.x[0]
        # self.u = np.maximum(10 * np.cos(self.x / 5), 1)

    # definition of the equation, using finite difference method
    def rhs(self, h):
        dFdh = -np.matmul(self.laplace, h) - self.djp(h)
        return np.matmul(self.nabla, h**3 * np.matmul(self.nabla, dFdh))

    # disjoining pressure

    def djp(self, h):
        return 1./h**6 - 1./h**3

    # no dealiasing for the FD version
    def dealias(self, u, real_space=False, ratio=1./2.):
        return u

    def first_spatial_derivative(self, u, direction=0):
        return np.matmul(self.nabla, u)


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        #self.tfe = ThinFilmEquation(N, L)
        self.tfe = ThinFilmEquationFD(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        self.volume_constraint.fixed_volume = 0
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)
        # initialize time stepper
        # self.time_stepper = RungeKutta4()
        # self.time_stepper = RungeKuttaFehlberg45()
        # self.time_stepper.error_tolerance = 1e1
        # self.time_stepper.dt = 3e-5
        self.time_stepper = BDF(self)  # better for FD
        # assign the continuation parameter
        self.continuation_parameter = (self.volume_constraint, "fixed_volume")

    # set higher modes to null, for numerical stability
    def dealias(self, fraction=1./2.):
        self.tfe.u = self.tfe.dealias(self.tfe.u, True)

    def norm(self):
        return np.trapz(self.tfe.u, self.tfe.x[0])


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=256, L=100)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 1
dudtnorm = 1
if not os.path.exists("initial_state.dat"):
    while dudtnorm > 1e-8:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig("out/img/{:05d}.svg".format(plotID))
            plotID += 1
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
    # perform dealiasing
    problem.dealias()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    #print('largest EVs: ', problem.latest_eigenvalues[:3])
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
