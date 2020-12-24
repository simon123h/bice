#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.continuation import VolumeConstraint, TranslationConstraint
from bice.core.profiling import Profiler


class AdaptiveSubstrateEquation(FiniteDifferencesEquation):

    def __init__(self, N, L):
        super().__init__(shape=(2, N))
        # parameters
        self.theta = np.sqrt(0.6)  # contact angle
        self.h_p = 1e-2  # precursor film height ratio
        self.sigma = 0.3  # relative grafting density
        self.gamma_bl = 0.1  # surface tension ratio
        self.Nlk = 0.3  # polymer chain length
        self.T = 20  # temperature
        self.chi = 0  # miscibility
        self.D = 1e-4  # brush lateral diffusion constant
        self.M = 1e-3  # absorption constant
        self.U = 0  # substrate velocity
        self.alpha = 0  # substrate inclination
        # spatial coordinate
        self.x = [np.linspace(-L/2, L/2, N)]
        # initial condition
        hmax = 1
        s = self.x[0]
        h = np.maximum(hmax - s**2 / (4 * hmax) * self.theta**2, self.h_p)
        z = 0*s + 0.1
        self.u = np.array([h, z])
        # build finite difference matrices
        self.build_FD_matrices(sparse=True)

    # definition of the equation, using finite difference method
    def rhs(self, u):
        # expand unknowns
        h, z = u
        # dry brush height
        H_dry = self.sigma * self.Nlk
        # polymer volume fraction (polymer concentration)
        c = H_dry / (H_dry + z)
        # disjoining pressure
        djp = 5/3 * (self.theta * self.h_p)**2 * \
            (self.h_p**3 / h**6 - c / h**3)
        # adaptive brush-liquid surface tension
        gamma_bl = self.gamma_bl * c
        # mobilities
        Qhh = h**3
        Qzz = self.D * z
        # brush energy derivative
        dfbrush = self.T * (self.sigma**2 / c + c + np.log(1 - c))
        # include miscibility effects
        dfbrush += self.T * self.chi * c / (z + H_dry)
        # free energy variations
        dFdh = -self.laplace.dot(h+z) - djp
        dFdz = -self.laplace.dot(h+z*(1+gamma_bl)) + dfbrush
        # absorption term
        M_absorb = self.M * (dFdh - dFdz)
        # dynamic equations
        dhdt = self.nabla.dot(Qhh * self.nabla.dot(dFdh)) - M_absorb
        dzdt = self.nabla.dot(Qzz * self.nabla.dot(dFdz)) + M_absorb
        # combine and return
        return np.array([dhdt, dzdt])

    def jacobian(self, u):
        # expand unknowns
        h, z = u
        # dry brush height
        H_dry = self.sigma * self.Nlk
        # polymer volume fraction (polymer concentration)
        c = H_dry / (H_dry + z)
        # disjoining pressure
        djp = 5/3 * (self.theta * self.h_p)**2 * \
            (self.h_p**3 / h**6 - c / h**3)
        # adaptive brush-liquid surface tension
        gamma_bl = self.gamma_bl * c
        # mobilities
        Qhh = h**3
        Qzz = self.D * z
        # brush energy derivative
        dfbrush = self.T * (self.sigma**2 / c + c + np.log(1 - c))
        # include miscibility effects
        dfbrush += self.T * self.chi * c / (z + H_dry)
        # free energy variations
        dFdh = -self.laplace.dot(h+z) - djp
        dFdz = -self.laplace.dot(h+z*(1+gamma_bl)) + dfbrush
        # abbreviations
        nabla = self.nabla
        laplace = self.laplace
        # mobility derivatives
        Qhh = diags(h**3)
        dQhh_dh = diags(3 * h**2)
        Qzz = diags(self.D * z)
        dQzz_dz = self.D
        # brush energy derivatives
        dc_dz = -c / (H_dry + z)
        dgamma_bl_dz = self.gamma_bl * dc_dz
        ddfbrush_dz = self.T * (self.sigma**2 / H_dry +
                                dc_dz + 1/(1-c) * dc_dz)
        ddfbrush_dz += -2 * self.T * self.chi * H_dry / (z + H_dry)**3
        # disjoining pressure derivatives
        djp_pf = 5/3 * (self.theta * self.h_p)**2
        ddjp_dh = diags(djp_pf * (c * 3 / h**4 - self.h_p**3 * 6 / h**7))
        ddjp_dz = diags(-djp_pf * dc_dz / h**3)
        # free energy variation derivatives
        ddFdh_dh = -laplace - ddjp_dh
        ddFdh_dz = -laplace - ddjp_dz
        ddFdz_dh = -laplace
        ddFdz_dz = -laplace - laplace * \
            diags(gamma_bl+z*dgamma_bl_dz) + diags(ddfbrush_dz)
        # absorption term derivative
        dM_absorb_dh = self.M * (ddFdh_dh - ddFdz_dh)
        dM_absorb_dz = self.M * (ddFdh_dz - ddFdz_dz)
        # derivatives of dynamic equations
        ddhdt_dh = nabla.dot(dQhh_dh * nabla.dot(dFdh) +
                             Qhh * nabla.dot(ddFdh_dh)) - dM_absorb_dh
        ddhdt_dz = nabla.dot(Qhh * nabla.dot(ddFdh_dz)) - dM_absorb_dz
        ddzdt_dh = nabla.dot(Qzz * nabla.dot(ddFdz_dh)) + dM_absorb_dh
        ddzdt_dz = nabla.dot(dQzz_dz * nabla.dot(dFdz) +
                             Qzz * nabla.dot(ddFdz_dz)) + dM_absorb_dz
        # combine and return
        return sp.vstack((sp.hstack((ddhdt_dh, ddhdt_dz)),
                          sp.hstack((ddzdt_dh, ddzdt_dz))))

    def du_dx(self, u, direction=0):
        h, z = u
        return np.matmul(self.nabla, h)

    def plot(self, ax):
        ax.set_ylim((0, 1.2))
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        h, xi = self.u
        ax.plot(self.x[0], h+xi, markersize=5, label="liquid")
        ax.plot(self.x[0], xi, markersize=5, label="substrate")
        ax.legend()


class ThinFilm(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        #self.tfe = ThinFilmEquation(N, L)
        self.tfe = AdaptiveSubstrateEquation(N, L)
        self.add_equation(self.tfe)
        # initialize time stepper
        # self.time_stepper = time_steppers.BDF2()
        self.time_stepper = time_steppers.BDF(self)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        self.volume_constraint.fixed_volume = 0
        # Generate the translation constraint
        self.translation_constraint = TranslationConstraint(self.tfe)
        # assign the continuation parameter
        self.continuation_parameter = (self.volume_constraint, "fixed_volume")

    def norm(self):
        h, z = self.tfe.u
        return np.trapz(h, self.tfe.x[0])


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = ThinFilm(N=512, L=10)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

Profiler.start()

# time-stepping
n = 0
plotevery = 100
dudtnorm = 1
while dudtnorm > 1e-8:
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
    # calculate the new norm
    dudtnorm = np.linalg.norm(problem.rhs(problem.u))

Profiler.print_summary()

# save the state, so we can reload it later
problem.save("initial_state.dat")

# # load the initial state
# problem.load("initial_state.dat")

# # start parameter continuation
# problem.continuation_stepper.ds = 1e-2
# problem.continuation_stepper.ndesired_newton_steps = 3

# # Impose the constraints
# problem.volume_constraint.fixed_volume = np.trapz(
#     problem.tfe.u, problem.tfe.x[0])
# problem.add_equation(problem.volume_constraint)
# problem.add_equation(problem.translation_constraint)

# problem.continuation_stepper.convergence_tolerance = 1e-10

# n = 0
# plotevery = 1
# while problem.volume_constraint.fixed_volume < 1000:
#     # perform continuation step
#     problem.continuation_step()
#     # perform dealiasing
#     problem.dealias()
#     n += 1
#     print("step #:", n, " ds:", problem.continuation_stepper.ds)
#     #print('largest EVs: ', problem.eigen_solver.latest_eigenvalues[:3])
#     # plot
#     if n % plotevery == 0:
#         problem.plot(ax)
#         fig.savefig("out/img/{:05d}.svg".format(plotID))
#         plotID += 1
