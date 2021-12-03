#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from bice import Problem, time_steppers
from bice.pde import PseudospectralEquation
from bice.continuation import VolumeConstraint, TranslationConstraint


class ThinFilmEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Thin-Film Equation
    equation
    dh/dt = d/dx (h^3 d/dx ( - d^2/dx^2 h - Pi(h) ))
    with the disjoining pressure:
    Pi(h) = 1/h^3 - 1/h^6
    """

    def __init__(self, N, L):
        super().__init__(shape=N)
        # parameters: none
        # space and fourier space
        self.x = [np.linspace(-L/2, L/2, N, endpoint=False)]
        x = self.x[0]
        self.build_kvectors(real_fft=True)
        # initial condition
        # self.u = np.ones(N) * 3
        self.u = 2 * np.cos(x*2*np.pi/L) + 1
        # self.u = np.maximum(10 * np.cos(x / 5), 1)

    # definition of the equation, using pseudospectral method
    def rhs(self, h):
        h_k = np.fft.rfft(h)
        k = self.k[0]
        djp_k = self.dealias(np.fft.rfft(self.djp(h)))
        dhhh_dx = np.fft.irfft(self.dealias(
            self.dealias(np.fft.rfft(h**3)) * 1j * k))
        term1 = np.fft.irfft(self.dealias(1j * k * (-k**2 * h_k + djp_k)))
        term2 = np.fft.irfft(self.dealias(k**2 * (k**2 * h_k - djp_k)))
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
        k = self.k[0]
        k_F = (1-ratio) * k[-1]
        u_k *= np.exp(-36*(4. * k / 5. / k_F)**36)
        if real_space:
            return np.fft.irfft(u_k)
        return u_k

    def du_dx(self, u, direction=0):
        du_dx = 1j*self.k*np.fft.rfft(u)
        return np.fft.irfft(du_dx)


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = ThinFilmEquation(N, L)
        self.add_equation(self.tfe)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        self.volume_constraint.fixed_volume = 0
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)
        # initialize time stepper
        # self.time_stepper = time_steppers.RungeKutta4()
        # self.time_stepper = time_steppers.RungeKuttaFehlberg45()
        # self.time_stepper.error_tolerance = 1e1
        # self.time_stepper.dt = 3e-5
        self.time_stepper = time_steppers.BDF(self)
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
if not os.path.exists("initial_state.npz"):
    while dudtnorm > 1e-8:
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
            print("Aborted.")
            break
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

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
    # print('largest EVs: ', problem.eigen_solver.latest_eigenvalues[:3])
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig(f"out/img/{plotID:05d}.svg")
        plotID += 1
