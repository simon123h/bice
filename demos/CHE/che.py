#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation, FiniteDifferenceEquation, PseudospectralEquation
from bice.time_steppers import RungeKutta4, RungeKuttaFehlberg45, BDF2
from bice.constraints import TranslationConstraint, VolumeConstraint


class CahnHilliardEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.a = -0.5
        self.kappa = 1.
        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N)]
        self.build_kvectors()
        # initial condition
        #self.u = (np.random.random(N)-0.5)*0.02
        self.u = np.cos(self.x[0]/(L/4)) - 0.1
        # calculate linear part beforehand

    # definition of the CHE (right-hand side)
    def rhs(self, u):
        u_k = np.fft.fft(u)
        u3_k = np.fft.fft(u**3)
        result_k = -self.ksquare*(self.kappa*self.ksquare*u_k + self.a*u_k + u3_k)
        result = np.fft.ifft(result_k).real
        return result

    def first_spatial_derivative(self, u, direction=0):
        du_dx = 1j*self.k[direction]*np.fft.fft(u)
        return np.fft.ifft(du_dx).real


class CahnHilliardProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.che = CahnHilliardEquation(N, L)
        self.add_equation(self.che)
        # initialize time stepper
        #self.time_stepper = RungeKuttaFehlberg45(dt=1e-3)
        self.time_stepper.error_tolerance = 1e-7
        self.time_stepper = RungeKutta4(dt=1e-2)
        # assign the continuation parameter
        self.continuation_parameter = (self.che, "a")

    # set higher modes to null, for numerical stability
    def dealias(self, fraction=1./2.):
        u_k = np.fft.fft(self.che.u)
        N = len(u_k)
        k = int(N * fraction)
        u_k[k + 1:-k] = 0
        self.che.u = np.fft.ifft(u_k).real


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

if not os.path.exists("initial_state.dat"):
    while dudtnorm > 1e-6:
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
            break
    # save the state, so we can reload it later
    problem.save("initial_state.dat")
else:
    # load the initial state
    problem.load("initial_state.dat")

# start parameter continuation
problem.continuation_stepper.ds = 1e-3
problem.continuation_stepper.ndesired_newton_steps = 3
problem.continuation_stepper.ds_max = 1e0
problem.always_check_eigenvalues = True
problem.neigs = 50
translation_constraint = TranslationConstraint(problem.che)
problem.add_equation(translation_constraint)
problem.volume_constraint = VolumeConstraint(problem.che)
problem.add_equation(problem.volume_constraint)
problem.volume_constraint.fixed_volume = np.trapz(problem.che.u, problem.che.x[0])
problem.continuation_parameter = (problem.volume_constraint, 'fixed_volume')

n = 0
plotevery = 1
while problem.che.a < 1.:
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
    # perform continuation step
    problem.continuation_step()
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    n += 1
