#!/usr/bin/python3
import shutil
import os
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation, PseudospectralEquation
from bice.continuation import TranslationConstraint
from bice import profile, Profiler


class SwiftHohenbergEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__(shape=N)
        # parameters
        self.r = -0.013
        self.kc = 0.5
        self.v = 0.41
        self.g = 1
        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N)]
        self.build_kvectors(real_fft=True)
        # initial condition
        self.u = np.cos(
            2 * np.pi * self.x[0] / 10) * np.exp(-0.005 * self.x[0]**2)

    # definition of the SHE (right-hand side)
    @profile
    def rhs(self, u):
        u_k = np.fft.rfft(u)
        return np.fft.irfft((self.r - (self.kc**2 - self.k[0]**2)**2) * u_k) + self.v * u**2 - self.g * u**3

    # definition of spatial derivative for translation constraint
    def du_dx(self, u, direction=0):
        du_dx = 1j*self.k[direction]*np.fft.rfft(u)
        return np.fft.irfft(du_dx)


class SwiftHohenbergProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.she = SwiftHohenbergEquation(N, L)
        self.add_equation(self.she)
        # initialize time stepper
        self.time_stepper = time_steppers.RungeKuttaFehlberg45(dt=1e-3)
        self.time_stepper.error_tolerance = 1e-7
        # assign the continuation parameter
        self.continuation_parameter = (self.she, "r")

    # set higher modes to null, for numerical stability
    @profile
    def dealias(self, fraction=1./2.):
        u_k = np.fft.rfft(self.she.u)
        N = len(u_k)
        k = int(N*fraction)
        u_k[k+1:-k] = 0
        self.she.u = np.fft.irfft(u_k)


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
if not os.path.exists("initial_state.npz"):
    Profiler.start()
    while dudtnorm > 1e-5:
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
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            break
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

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
        fig.savefig(f"out/img/{plotID:05d}.png")
        plotID += 1

Profiler.print_summary(nested=True)

# load the initial state and add extra dof for translation constraint
problem.remove_equation(constraint)
problem.load("initial_state.npz")
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
        fig.savefig(f"out/img/{plotID:05d}.png")
        plotID += 1
