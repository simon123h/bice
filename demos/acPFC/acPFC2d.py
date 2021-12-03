#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import PseudospectralEquation


class acPFCEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional semi-active coupled Phase Field Crystal Equation, a nonlinear PDE
    \partial_t psi_1 &= \Delta((r + (q_1^2 + \Delta)^2)\psi_1 + (\psi_1 + \bar\phi_1)^3 + c\psi_2) - v0\nabla P
    \partial_t psi_2 &= \Delta((r + (q_2^2 + \Delta)^2)\psi_2 + (\psi_2 + \bar\phi_2)^3 + c\psi_1)
    \partial_t P &= = C_1\Delta P - D_r C_1 P - v0\nabla\psi_1
    """

    def __init__(self, Nx, Lx, Ny, Ly, particles1, particles2):
        super().__init__(shape=(4, Nx, Ny))

        # parameters
        self.r = -1.5

        self.phi01 = -0.55
        self.q1 = 1.

        self.v0 = 0.23
        self.c = -0.2

        self.phi02 = -0.55
        self.q2 = 1.

        self.C1 = 0.1
        self.Dr = 0.5

        # space and fourier space
        x = np.linspace(-Lx/2, Lx/2, Nx)
        y = np.linspace(-Ly/2, Ly/2, Ny)
        x, y = np.meshgrid(x, y)
        self.x = [x, y]
        self.k = [np.fft.rfftfreq(Nx, Lx / (2. * Nx * np.pi)), np.fft.rfftfreq(Ny, Ly / (2. * Ny * np.pi))]
        # self.build_kvectors(real_fft=True)
        self.ksquare = self.k[0]**2 + self.k[1]**2

        # initial guess
        if os.path.exists("initial_state_2D.npz"):
            self.u = np.loadtxt("initial_state_2D.npz").reshape(self.shape)
        else:
            for i in range(particles1):
                u1 = 2.7*np.exp(-((self.x[0]-Lx*np.random.random())**2+(self.x[1]-Ly*np.random.random())**2)/5)
            
            for i in range(particles2):
                u2 = 2.7*np.exp(-((self.x[0]-Lx*np.random.random())**2+(self.x[1]-Ly*np.random.random())**2)/5)
            u3 = (np.random.random((Ny,Nx))-0.5)
            u4 = (np.random.random((Ny,Nx))-0.5)
            self.u = np.array([u1, u2, u3, u4])


    def rhs(self, u):
        u_k = np.fft.rfft(u)
        psi1_k3 = np.fft.rfft((u[0] + self.phi01)**3)
        psi2_k3 = np.fft.rfft((u[1] + self.phi02)**3)
        r1 = -self.ksquare * ((self.r + (self.q1**2 - self.ksquare)**2)*u_k[0] + psi1_k3 + self.c*u_k[1]) - 1j*self.v0*(self.k[0]*u_k[2] + self.k[1]*u_k[3])
        r2 = -self.ksquare * ((self.r + (self.q2**2 - self.ksquare)**2)*u_k[1] + psi2_k3 + self.c*u_k[0])
        r3 = -self.C1*self.ksquare * u_k[2] - self.Dr*self.C1*u_k[2] - 1j*self.v0*self.k[0]*u_k[0]
        r4 = -self.C1*self.ksquare * u_k[2] - self.Dr*self.C1*u_k[2] - 1j*self.v0*self.k[1]*u_k[0]

        res = np.array([r1, r2, r3, r4])
        res = np.fft.irfft(res)
        return res

    def plot(self, ax):
        for i in range(3):
            ax[i].imshow(self.u[i])


class acPFCProblem(Problem):

    def __init__(self, Nx, Lx, Ny, Ly, particles1=10, particles2=10):
        super().__init__()
        # add the acPFC equation to the problem
        self.acpfc = acPFCEquation(Nx, Lx, Ny, Ly, particles1, particles2)
        self.add_equation(self.acpfc)
        # initialize time stepper
        self.time_stepper = time_steppers.RungeKuttaFehlberg45(self, 1e-2)
        self.continuation_parameter = (self.acpfc, "phi01")

    # Norm is the L2-norm of the three fields
    def norm(self):
        u = self.acpfc.u
        N = self.acpfc.shape[-1]
        return np.sqrt(np.sum(u[0]**2/N + u[1]**2/N + u[2]**2/N))


if __name__ == "__main__":
    # create output folder
    shutil.rmtree("out", ignore_errors=True)
    os.makedirs("out/img", exist_ok=True)

    # create problem
    problem = acPFCProblem(Nx=128, Lx=16*np.pi, Ny=128, Ly=16*np.pi)

    # create figure

    fig = plt.figure(figsize=(16, 9))
    
    ax_sol1 = fig.add_subplot(411)
    ax_sol2 = fig.add_subplot(412)
    ax_sol3 = fig.add_subplot(421)
    ax_sol = [ax_sol1, ax_sol2, ax_sol3]
    ax_norm = fig.add_subplot(422)
    plotID = 0

    # time-stepping
    n = 0
    plotevery = 1000
    dudtnorm = 1
    L2norms = [problem.norm()]
    times = [problem.time]
    us = [problem.acpfc.u]
    problem.acpfc.v0 = 0.2
    T = 250.
    while problem.time < T:
        # plot
        if not n % plotevery:
            problem.plot(sol_ax=ax_sol)
            ax_norm.clear()
            ax_norm.plot(times, L2norms)
            fig.savefig(f"out/img/{plotID:05d}.svg")
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
        # add new norm and time and u
        L2norms += [problem.norm()]
        times += [problem.time]
        us += [problem.acpfc.u]
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            print("diverged")
            break
        # save the state, so we can reload it later
    #problem.save("initial_state.npz")
    plt.close()
    problem.acpfc.v0 = 0.3
    problem.time = 0
    problem.time_stepper = time_steppers.BDF(problem, dt_max=1e-2)
    T = 500.
    while problem.time < T:
        # plot
        if not n % plotevery:
            problem.plot(ax_sol)
            ax_norm.clear()
            ax_norm.plot(times, L2norms)
            fig.savefig(f"out/img/{plotID:05d}.svg")
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
        # add new norm and time and u
        L2norms += [problem.norm()]
        times += [problem.time]
        us += [problem.acpfc.u]
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            print("diverged")
            break
        # save the state, so we can reload it later
    #problem.save("initial_state.npz")
    plt.close()

    us = np.array(us)
    psi1s = us[:, 0, :]
    psi2s = us[:, 1, :]
    Ps = us[:, 2, :]

    fig = plt.figure(1, figsize=(16, 9))
    ax_psi1 = fig.add_subplot(221)
    ax_psi2 = fig.add_subplot(222)
    ax_norm = fig.add_subplot(223)
    ax_P = fig.add_subplot(224)

    ax_psi1.pcolormesh(times, problem.acpfc.x[0], psi1s.T, cmap='Reds')
    ax_psi2.pcolormesh(times, problem.acpfc.x[0], psi2s.T, cmap='Blues')
    ax_P.pcolormesh(times, problem.acpfc.x[0], Ps.T, cmap='Greens')
    ax_norm.plot(times, L2norms)

    fig_return = plt.figure(2)
    ax_return = fig_return.add_subplot(211)
    ax_u0 = fig_return.add_subplot(212)

    u0s = us[:, 0, 0]
    print(u0s.shape)
    maxs = u0s[1:-1][np.where(np.logical_and(u0s[1:-1] > u0s[:-2], u0s[1:-1] > u0s[2:]))]

    ax_return.plot(maxs[:-1], maxs[1:], 'k.')
    ax_u0.plot(times, u0s)

    plt.show()

    np.savetxt('psi1.dat', us[-1, 0, :])
    np.savetxt('psi2.dat', us[-1, 1, :])
    np.savetxt('P.dat', us[-1, 2, :])
