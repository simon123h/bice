#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import PseudospectralEquation
from bice.continuation import TranslationConstraint, VolumeConstraint


class NikolaevskiyEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Nikolaevskiy Equation
    equation, a nonlinear PDE
    \partial t h &= -\Delta (r - (1+\Delta)^2) h - 1/2 (\nabla h)^2
    """

    def __init__(self, Nx, Ny):
        super().__init__()
        # make sure N is even
        Nx = int(np.floor(Nx/2)*2)
        Ny = int(np.floor(Ny/2)*2)
        self.Nx = Nx
        self.Ny = Ny
        # we have only a single variable h
        self.shape = (Nx*Ny,)
        self.rshape = (Nx, Ny)
        # parameters
        self.r = 0.5  # drive
        self.m = 10  # characteristic system length
        self.ratio = 1  # length ratio Ly/Lx
        # space and fourier space
        L = 1
        self.x = [np.linspace(0, L, Nx), np.linspace(0, L, Ny)]
        kx = np.fft.rfftfreq(Nx, L / (2. * Nx * np.pi))
        ky = np.fft.fftfreq(Ny, L / (2. * Ny * np.pi))
        kx, ky = np.meshgrid(kx, ky)
        self.k = [kx, ky]
        # initial condition
        self.u = 2*(np.random.rand(Nx*Ny)-0.5) * 1e-5
        # create constraints
        self.volume_constraint = VolumeConstraint(self)
        self.translation_constraint_x = TranslationConstraint(
            self, direction=0)
        self.translation_constraint_y = TranslationConstraint(
            self, direction=1)

    # characteristic length scale

    @property
    def L0(self):
        return 2*np.pi / np.sqrt(1+np.sqrt(self.r))

    # definition of the Nikolaevskiy equation (right-hand side)
    def rhs(self, u):
        # calculate the system length
        L = self.L0 * self.m
        # include length scale in the k-vectors
        kx = self.k[0] / L
        ky = self.k[1] / L / self.ratio
        ksq = kx**2 + ky**2
        # fourier transform
        u_k = np.fft.rfft2(u.reshape(self.rshape))
        # calculate linear part (in fourier space)
        lin = ksq * (self.r - (1-ksq)**2) * u_k
        # calculate nonlinear part (in real space)
        nonlin = np.fft.irfft2(1j * kx * u_k)**2
        nonlin += np.fft.irfft2(1j * ky * u_k)**2
        # sum up and return
        return (np.fft.irfft2(lin) - 0.5 * nonlin).ravel()

    # calculate the spatial derivative
    def first_spatial_derivative(self, u, direction=0):
        u2 = u.reshape(self.rshape)
        du_dx = 1j*self.k[direction]*np.fft.rfft2(u2)
        du_dx = np.fft.irfft2(du_dx)
        return du_dx.ravel()

    # plot the solution
    def plot(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        x, y = np.meshgrid(self.x[0], self.x[1])
        h = self.u.reshape(self.rshape)
        Lx = self.L0 * self.m
        Ly = Lx * self.ratio
        pcol = ax.pcolor(x*Lx, y*Ly, h, cmap="coolwarm", rasterized=True)
        pcol.set_edgecolor('face')
        # put velocity labels into plot
        ax.text(0.02, 0.06, "vx = {:.1g}".format(
            self.translation_constraint_x.u[0]), transform=ax.transAxes)
        ax.text(0.02, 0.02, "vy = {:.1g}".format(
            self.translation_constraint_y.u[0]), transform=ax.transAxes)


class NikolaevskiyProblem(Problem):

    def __init__(self, Nx, Ny):
        super().__init__()
        # Add the Nikolaevskiy equation to the problem
        self.ne = NikolaevskiyEquation(Nx, Ny)
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
        # TODO: fix for 2d
        # pass
        u_k = np.fft.rfft(self.ne.u)
        u_k[0] = 0
        self.ne.u = np.fft.irfft(u_k)

    # Norm is the L2-norm of the NE
    def norm(self):
        # TODO: divide by Nx*Nx*Lx*Ly
        return np.linalg.norm(self.ne.u)


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = NikolaevskiyProblem(Nx=12, Ny=12)
problem.ne.r = 0.5
problem.ne.m = 1.25

# create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 10
dudtnorm = 1
T = 50 / problem.ne.r
if not os.path.exists("initial_state.dat"):
    while problem.time < T:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig("out/img/{:05d}.png".format(plotID))
            plotID += 1
            print("step #: {:}".format(n))
            print("time:   {:}".format(problem.time))
            print("dt:     {:}".format(problem.time_stepper.dt))
            print("|dudt|: {:}".format(dudtnorm))
            print("norm:   {:}".format(problem.norm()))
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
problem.continuation_stepper.ds = 1e-1
problem.continuation_stepper.ds_max = 1e0
problem.continuation_stepper.ndesired_newton_steps = 5
problem.continuation_stepper.convergence_tolerance = 1e-8
problem.settings.always_locate_bifurcations = True
problem.settings.neigs = 20
problem.settings.verbose = True

# add constraints
problem.add_equation(problem.ne.volume_constraint)
problem.add_equation(problem.ne.translation_constraint_x)
problem.add_equation(problem.ne.translation_constraint_y)

# create new figure
plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

n = 0
plotevery = 1
bifcount = 0
while problem.ne.m > 0:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1

    if problem.bifurcation_diagram.current_solution().is_bifurcation():
        bifcount += 1
    if bifcount > 0:
        break


# try to switch branches
print("Branch switching started")
problem.switch_branch(amplitude=1e-2)
problem.plot(ax)
fig.savefig("out/img/{:05d}.png".format(plotID))
plotID += 1


while problem.ne.m > 0:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1
