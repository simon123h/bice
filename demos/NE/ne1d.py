#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, PseudospectralEquation
from bice.time_steppers import RungeKutta4, RungeKuttaFehlberg45, BDF2, BDF
from bice.constraints import TranslationConstraint, VolumeConstraint


class NikolaevskiyEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Nikolaevskiy Equation
    equation, a nonlinear PDE
    \partial t h &= -\Delta (r - (1+\Delta)^2) h - 1/2 (\nabla h)^2
    """

    def __init__(self, N):
        super().__init__()
        # make sure N is even
        N = int(np.floor(N/2)*2)
        # we have only a single variable h
        self.shape = (N,)
        # parameters
        self.r = 0.5  # drive
        self.m = 10  # characteristic system length
        # space and fourier space
        L = 1
        self.x = [np.linspace(0, L, N)]
        self.k = [np.fft.rfftfreq(N, L / (2. * N * np.pi))]
        self.ksquare = self.k[0]**2
        # initial condition
        self.u = 2*(np.random.rand(N)-0.5) * 1e-5
        # self.u = np.fft.rfft(self.u)

    # characteristic length scale
    @property
    def L0(self):
        return 2*np.pi / np.sqrt(1+np.sqrt(self.r))

    # definition of the Nikolaevskiy equation (right-hand side)
    def rhs(self, u):
        # calculate the system length
        L = self.L0 * self.m
        # include length scale in spatial derivatives
        k = self.k[0] / L
        ksq = self.ksquare / L**2
        # fourier transform
        u_k = np.fft.rfft(u)
        # calculate linear part (in fourier space)
        lin = ksq * (self.r - (1-ksq)**2) * u_k
        # calculate nonlinear part (in real space)
        nonlin = - 0.5 * np.fft.irfft(1j * k * u_k)**2
        # sum up and return
        return np.fft.irfft(lin) + nonlin

    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("h(x,t)")
        L = self.L0 * self.m
        ax.plot(self.x[0] * L, self.u)


class NikolaevskiyProblem(Problem):

    def __init__(self, N):
        super().__init__()
        # Add the Nikolaevskiy equation to the problem
        self.ne = NikolaevskiyEquation(N)
        self.add_equation(self.ne)
        # initialize time stepper
        # self.time_stepper = RungeKutta4(dt=1e-7)
        # self.time_stepper = RungeKuttaFehlberg45(dt=1e-7, error_tolerance=1e-4)
        # self.time_stepper = BDF2(dt=1e-3)
        self.time_stepper = BDF(self, dt_max=1e-1)
        # assign the continuation parameter
        self.continuation_parameter = (self.ne, "m")

    # set higher modes to null, for numerical stability
    def dealias(self, fraction=1./2.):
        u_k = np.fft.rfft(self.ne.u)
        N = len(u_k)
        k = int(N*fraction)
        u_k[k+1:] = 0
        u_k[0] = 0
        self.ne.u = np.fft.irfft(u_k)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = NikolaevskiyProblem(N=64)
problem.ne.r = 0.5
problem.ne.m = 1.1

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1
T = 60
if not os.path.exists("initial_state.dat"):
    while problem.time < T:
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
            print("diverged")
            break
    # save the state, so we can reload it later
    problem.save("initial_state.dat")
else:
    # load the initial state
    problem.load("initial_state.dat")

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 3
problem.settings.always_check_eigenvalues = True
problem.settings.neigs = 10

# add constraints
volume_constraint = VolumeConstraint(problem.ne)
problem.add_equation(volume_constraint)
translation_constraint = TranslationConstraint(problem.ne)
problem.add_equation(translation_constraint)

# create new figure
plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

n = 0
plotevery = 10
while problem.ne.m > 0:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
