#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation, FiniteDifferenceEquation
from bice.time_steppers import RungeKutta4, RungeKuttaFehlberg45, BDF2
from bice.constraints import TranslationConstraint, VolumeConstraint
from bice.profiling import profile, Profiler


class CahnHilliardEquation(Equation):
    r"""
    Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.e = 1
        self.D = 1.2
        # space and fourier space
        self.x = np.array([np.linspace(-L/2, L/2, N),
                           np.linspace(-L/2, L/2, N)])
        self.kx, self.ky = np.meshgrid(np.fft.fftfreq(
            N, L/(N*2*np.pi)), np.fft.fftfreq(N, L/(N*2*np.pi)))
        self.ksquare = self.kx**2 + self.ky**2
        # initial condition
        self.u = (np.random.random((N, N))-0.5)*0.02

    # definition of the CHE (right-hand side)
    @profile
    def rhs(self, u):
        N0 = u.size
        u2 = u.reshape((self.x[0].size, self.x[1].size))
        # u_k = np.fft.fft2(u)
        # u3_k = np.fft.fft2(u**3)
        # linear = -self.D * (self.ksquare * self.ksquare +
        #                     self.e * self.ksquare)

        u_k = np.fft.fft2(u2)  # Fouriertransformation
        u3_k = np.fft.fft2(u2*u2*u2)
        result_k = (-self.D*self.ksquare*self.ksquare +
                    self.ksquare)*u_k - self.ksquare*u3_k
        result = np.fft.ifft2(result_k).real

        return result.reshape(u.size)

        # return np.fft.ifft2(linear*u_k - self.ksquare*u3_k).real.reshape(u.size)


class CahnHilliardProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Swift-Hohenberg equation to the problem
        self.che = CahnHilliardEquation(N, L)
        self.add_equation(self.che)
        # initialize time stepper
        # self.time_stepper = RungeKuttaFehlberg45(dt=1e-3)
        self.time_stepper.error_tolerance = 1e-7
        self.time_stepper = RungeKutta4(dt=5e-3)
        # assign the continuation parameter
        self.continuation_parameter = (self.che, "D")


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = CahnHilliardProblem(N=128, L=128)

# create figure
# fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

# time-stepping
n = 0
plotevery = 1000
dudtnorm = 1
# mx, my = np.meshgrid(problem.che.x[0], problem.che.x[1])

# if not os.path.exists("initial_state2.dat"):

Profiler.start()

for i in range(500):
    problem.time_step()

Profiler.print_summary()

# while dudtnorm > 1e-6:
# # for i in range(500):
#     # plot
#     if n % plotevery == 0:
#         plt.cla()
#         plt.pcolormesh(mx, my, problem.che.u.reshape(
#             (problem.che.x[0].size, problem.che.x[1].size)), edgecolors='face')
#         plt.colorbar()
#         plt.savefig("out/img/{:05d}.svg".format(plotID))
#         plt.close()
#         # problem.plot(ax)
#         # fig.savefig("out/img/{:05d}.svg".format(plotID))
#         plotID += 1
#         print("step #: {:}".format(n))
#         print("dt:     {:}".format(problem.time_stepper.dt))
#         print("|dudt|: {:}".format(dudtnorm))

#     n += 1
#     # perform timestep
#     problem.time_step()
#     # calculate the new norm
#     dudtnorm = np.linalg.norm(problem.rhs(problem.u))
#     # catch divergent solutions
#     if np.max(problem.u) > 1e12:
#         break


#     # save the state, so we can reload it later
#     problem.save("initial_state.dat")
# else:
#     # load the initial state
#     problem.load("initial_state.dat")

# # start parameter continuation
# problem.continuation_stepper.ds = 1e-3
# problem.continuation_stepper.ndesired_newton_steps = 3
# problem.continuation_stepper.always_check_eigenvalues = True

# translation_constraint = TranslationConstraint(problem.che)
# problem.add_equation(translation_constraint)
# volume_constraint = VolumeConstraint(problem.che)
# problem.add_equation(volume_constraint)

# n = 0
# plotevery = 1
# while problem.che.D > 0:
#     # plot
#     if n % plotevery == 0:
#         problem.plot(ax)
#         fig.savefig("out/img/{:05d}.svg".format(plotID))
#         plotID += 1
#     # perform continuation step
#     problem.continuation_step()
#     print("step #:", n, " ds:", problem.continuation_stepper.ds)
#     n += 1
