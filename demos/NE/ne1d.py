#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from bice import Problem, time_steppers
from bice.pde import PseudospectralEquation


class NikolaevskiyEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Nikolaevskiy Equation
    equation, a nonlinear PDE
    \partial t h &= -\Delta (r - (1+\Delta)^2) h - 1/2 (\nabla h)^2
    """

    def __init__(self, N):
        # make sure N is even
        N = int(np.floor(N/2)*2)
        super().__init__(shape=N)
        # parameters
        self.r = 0.5  # drive
        self.m = 10  # characteristic system length
        # space and fourier space
        self.x = [np.linspace(0, 1, N)]
        self.build_kvectors(real_fft=True)
        # initial condition
        self.u = 2*(np.random.rand(N)-0.5) * 1e-5

    # characteristic length scale
    @property
    def L0(self):
        return 2*np.pi / np.sqrt(1+np.sqrt(self.r))

    # definition of the Nikolaevskiy equation (right-hand side)
    def rhs(self, u):
        # calculate the system length
        L = self.L0 * self.m
        # include length scale in the k-vector
        k = self.k[0] / L
        ksq = k**2
        # fourier transform
        u_k = np.fft.rfft(u)
        # calculate linear part (in fourier space)
        lin = ksq * (self.r - (1-ksq)**2) * u_k
        # calculate nonlinear part (in real space)
        nonlin = np.fft.irfft(1j * k * u_k)**2
        # sum up and return
        return np.fft.irfft(lin) - 0.5 * nonlin

    # calculate the spatial derivative
    def du_dx(self, u, direction=0):
        du_dx = 1j*self.k[direction]*np.fft.rfft(u)
        return np.fft.irfft(du_dx)

    # plot the solution
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
        # self.time_stepper = time_steppers.RungeKutta4(dt=1e-7)
        # self.time_stepper = time_steppers.RungeKuttaFehlberg45(dt=1e-7, error_tolerance=1e-4)
        # self.time_stepper = time_steppers.BDF2(dt=1e-3)
        self.time_stepper = time_steppers.BDF(self, dt_max=1e-1)
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

    # Norm is the L2-norm of the NE
    def norm(self):
        return np.linalg.norm(self.ne.u)


if __name__ == "__main__":
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
    T = 100 / problem.ne.r
    while problem.time < T:
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
            print("diverged")
            break
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
