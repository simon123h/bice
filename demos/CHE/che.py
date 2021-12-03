#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from bice import Problem, time_steppers
from bice.pde import PseudospectralEquation
from bice.continuation import TranslationConstraint, VolumeConstraint


class CahnHilliardEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Cahn-Hilliard Equation
    equation, a nonlinear PDE
    \partial t c &= \Delta (c^3 + a * c - \kappa * \Delta c)
    """

    def __init__(self, N, L):
        # we have only a single variable h, so the shape is just (N,)
        super().__init__(shape=N)
        # parameters
        self.a = -0.5
        self.kappa = 1.
        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N)]
        self.build_kvectors(real_fft=True)
        # initial condition
        # self.u = (np.random.random(N)-0.5)*0.02
        self.u = np.cos(self.x[0]/(L/4)) - 0.1
        # calculate linear part beforehand

    # definition of the CHE (right-hand side)
    def rhs(self, u):
        u_k = np.fft.rfft(u)
        u3_k = np.fft.rfft(u**3)
        result_k = -self.ksquare * \
            (self.kappa*self.ksquare*u_k + self.a*u_k + u3_k)
        return np.fft.irfft(result_k)

    def du_dx(self, u, direction=0):
        du_dx = 1j*self.k[direction]*np.fft.rfft(u)
        return np.fft.irfft(du_dx)


class CahnHilliardProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Cahn-Hilliard equation to the problem
        self.che = CahnHilliardEquation(N, L)
        self.add_equation(self.che)
        # initialize time stepper
        # self.time_stepper = time_steppers.RungeKuttaFehlberg45(dt=1e-3)
        # self.time_stepper.error_tolerance = 1e-7
        self.time_stepper = time_steppers.RungeKutta4(dt=1e-2)
        # assign the continuation parameter
        self.continuation_parameter = (self.che, "a")

    # set higher modes to null, for numerical stability
    def dealias(self, fraction=1./2.):
        u_k = np.fft.rfft(self.che.u)
        N = len(u_k)
        k = int(N * fraction)
        u_k[k + 1:-k] = 0
        self.che.u = np.fft.irfft(u_k)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = CahnHilliardProblem(N=128, L=128)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 1000
dudtnorm = 1

if not os.path.exists("initial_state.npz"):
    while dudtnorm > 1e-6:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig(f"out/img/{plotID:05d}.svg")
            plotID += 1
            print(f"step #: {n}")
            print(f"time:   {problem.time}")
            print(f"dt:     {problem.time_stepper.dt}")
            print(f"|dudt|: {dudtnorm}")
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
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

# start parameter continuation
problem.continuation_stepper.ds = 1e-3
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.ds_max = 1e0
problem.settings.neigs = 50
translation_constraint = TranslationConstraint(problem.che)
problem.add_equation(translation_constraint)
problem.volume_constraint = VolumeConstraint(problem.che)
problem.add_equation(problem.volume_constraint)
problem.volume_constraint.fixed_volume = np.trapz(
    problem.che.u, problem.che.x[0])
problem.continuation_parameter = (problem.volume_constraint, 'fixed_volume')

n = 0
plotevery = 1
while problem.che.a < 1.:
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig(f"out/img/{plotID:05d}.svg")
        plotID += 1
    # perform continuation step
    problem.continuation_step()
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    n += 1
