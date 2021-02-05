#!/usr/bin/python3
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from bice import Problem, time_steppers
from bice.pde import PseudospectralEquation
from bice.continuation import TranslationConstraint, VolumeConstraint
from bice import profile, Profiler


class CahnHilliardEquation(PseudospectralEquation):
    r"""
    Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__(shape=(N, N))
        # parameters
        self.a = -0.5
        self.kappa = 1.
        # list of spatial coordinate. list is important,
        # to deal with several dimensions with different discretization/lengths
        self.x = [np.linspace(-L/2, L/2, N), np.linspace(-L/2, L/2, N)]
        self.build_kvectors(real_fft=True)
        # initial condition
        # self.u = (np.random.random((N, N))-0.5)*0.02
        mx, my = np.meshgrid(*self.x)
        self.u = np.cos(np.sqrt(mx**2 + my**2)/(L/4)) - 0.1

    # definition of the CHE (right-hand side)
    @profile
    def rhs(self, u):
        u_k = np.fft.rfft2(u)
        u3_k = np.fft.rfft2(u**3)
        result_k = -self.ksquare * \
            (self.kappa * self.ksquare * u_k + self.a * u_k + u3_k)
        return np.fft.irfft2(result_k)

    @profile
    def du_dx(self, u, direction=0):
        du_dx = 1j*self.k[direction]*np.fft.rfft2(u)
        return np.fft.irfft2(du_dx)


class CahnHilliardProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.che = CahnHilliardEquation(N, L)
        self.add_equation(self.che)
        # initialize time stepper
        # self.time_stepper = time_steppers.RungeKuttaFehlberg45(dt=1e-3)
        # self.time_stepper.error_tolerance = 1e-6
        # self.time_stepper.max_rejections = 100
        self.time_stepper = time_steppers.BDF(self)
        self.time_stepper.dt = 1e-3
        # assign the continuation parameter
        self.continuation_parameter = (self.che, "a")


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = CahnHilliardProblem(N=64, L=64)

# time-stepping
n = 0
plotID = 0
plotevery = 5
dudtnorm = 1
mx, my = np.meshgrid(problem.che.x[0], problem.che.x[1])

Profiler.start()

if not os.path.exists("initial_state2D.npz"):
    while dudtnorm > 1e-6:
        # plot
        if n % plotevery == 0:
            plt.cla()
            plt.pcolormesh(mx, my, problem.che.u, edgecolors='face')
            plt.colorbar()
            plt.savefig("out/img/{:05d}.png".format(plotID))
            plt.close()
            # problem.plot(ax)
            # fig.savefig("out/img/{:05d}.svg".format(plotID))
            plotID += 1
            print("step #: {:}".format(n))
            print("dt:     {:}".format(problem.time_stepper.dt))
            print("|dudt|: {:}".format(dudtnorm))

        n += 1
        # perform timestep
        problem.time_step()
        # calculate the new norm
        dudtnorm = np.linalg.norm(problem.rhs(problem.u))
        # catch divergent solutions
        if np.max(problem.u) > 1e12:
            break

    # save the state, so we can reload it later
    problem.save("initial_state2D.npz")
else:
    # load the initial state
    problem.load("initial_state2D.npz")

Profiler.print_summary()

# start parameter continuation
problem.continuation_stepper.ds = -1e-2
problem.continuation_stepper.ndesired_newton_steps = 3

volume_constraint = VolumeConstraint(problem.che)
problem.add_equation(volume_constraint)
translation_constraint_x = TranslationConstraint(problem.che, direction=0)
problem.add_equation(translation_constraint_x)
translation_constraint_y = TranslationConstraint(problem.che, direction=1)
problem.add_equation(translation_constraint_y)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

n = 0
plotevery = 1
while problem.che.a < 2:
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.png".format(plotID))
        plotID += 1
    # perform continuation step
    problem.continuation_step()
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    n += 1
