#!/usr/bin/python3
from volume_constraint import VolumeConstraint
import shutil
import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
import matplotlib.pyplot as plt
sys.path.append("../../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde.finite_differences import FiniteDifferencesEquation, NeumannBC, DirichletBC, NoBoundaryConditions
from bice.core.profiling import Profiler
from bice.continuation import NaturalContinuation


class AdaptiveSubstrateEquation(FiniteDifferencesEquation):

    def __init__(self, N, L):
        super().__init__(shape=(2, N))
        # parameters
        self.theta = np.sqrt(0.6)  # contact angle
        self.h_p = 5e-3  # precursor film height ratio
        self.sigma = 0.3  # relative grafting density
        self.gamma_bl = 0.01  # surface tension ratio
        self.Nlk = 0.2  # polymer chain length
        self.T = 50  # temperature
        self.chi = 0  # miscibility
        self.D = 1e-8  # brush lateral diffusion constant
        self.M = 1e-4  # absorption constant
        self.U = -0.05  # substrate velocity
        self.alpha = 0  # substrate inclination
        # spatial coordinate
        self.x = [np.linspace(0, L/2, N)]
        # initial condition
        hmax = 1
        s = self.x[0]
        h = np.maximum(hmax - s**2 / (4 * hmax) * self.theta**2, self.h_p)
        z = 0*s + 0.1
        self.u = np.array([h, z])
        # build finite difference matrices
        self.build_FD_matrices(approx_order=1)

    # overload building of FD matrices, because this equation has a more complicated set up
    def build_FD_matrices(self, approx_order=1):
        # build finite differences matrices...
        # (i) including the flux boundary conditions for Q * dF/dh
        self.bc = DirichletBC(vals=(1, 0))
        super().build_FD_matrices(approx_order)
        self.nabla_F = self.nabla
        # (ii) including the Neumann boundary conditions for h & z
        self.bc = NeumannBC()
        super().build_FD_matrices(approx_order)
        self.nabla_h = self.nabla
        self.laplace_h = self.laplace
        # (iii) differentiation operators with no specific boundary effects
        self.bc = NoBoundaryConditions()
        super().build_FD_matrices(approx_order)
        self.nabla0 = self.nabla

    # definition of the equation, using finite difference method
    def rhs(self, u):
        # expand unknowns
        h, z = u
        # dry brush height
        H_dry = self.sigma * self.Nlk
        # polymer volume fraction (polymer concentration)
        c = H_dry / (H_dry + z)
        # disjoining pressure
        h3 = h**3
        djp = 5/3 * (self.theta * self.h_p)**2 * \
            (self.h_p**3 / h3**2 - 1 / h3)
        # adaptive brush-liquid surface tension
        gamma_bl = self.gamma_bl * 1
        # mobilities
        Qhh = h3
        Qzz = self.D * z
        # brush energy derivative
        dfbrush = self.T * (self.sigma**2 / c + c + np.log(1 - c))
        # include miscibility effects
        dfbrush += self.T * self.chi * c / (z + H_dry)
        # free energy variations
        laplace_hz = self.laplace_h(h+z)
        dFdh = -laplace_hz - djp
        dFdz = -laplace_hz - gamma_bl * self.laplace_h(z) + dfbrush
        # absorption term
        M_absorb = self.M * (dFdh - dFdz)
        # flux into the liquid film to conserve liquid volume
        q = -(h[-1] - h[0] + z[-1] - z[0]) * self.U
        # dynamic equations
        dhdt = self.nabla_F(Qhh * self.nabla0(dFdh), q) - M_absorb
        dzdt = self.nabla_F(Qzz * self.nabla0(dFdz), 0) + M_absorb
        # advection term
        dhdt -= self.U * self.nabla_h(h)
        dzdt -= self.U * self.nabla_h(z)
        # combine and return
        return np.array([dhdt, dzdt])

    def jacobian2(self, u):
        # expand unknowns
        h, z = u
        # dry brush height
        H_dry = self.sigma * self.Nlk
        # polymer volume fraction (polymer concentration)
        c = H_dry / (H_dry + z)
        # disjoining pressure
        # TODO: add back adaptive wettability
        djp = 5/3 * (self.theta * self.h_p)**2 * \
            (self.h_p**3 / h**6 - 1 / h**3)
        # adaptive brush-liquid surface tension
        gamma_bl = self.gamma_bl * 1
        # mobilities
        Qhh = h**3
        Qzz = self.D * z
        # brush energy derivative
        dfbrush = self.T * (self.sigma**2 / c + c + np.log(1 - c))
        # include miscibility effects
        dfbrush += self.T * self.chi * c / (z + H_dry)
        # free energy variations
        dFdh = -self.laplace_h(h+z) - djp
        dFdz = -self.laplace_h(h+z) - gamma_bl * self.laplace_h(z) + dfbrush
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
        ddFdh_dh = -self.laplace_h() - ddjp_dh
        ddFdh_dz = -self.laplace_h() - ddjp_dz
        ddFdz_dh = -self.laplace_h()
        ddFdz_dz = -self.laplace_h() - self.laplace_h(diags(gamma_bl +
                                                            z*dgamma_bl_dz)) + diags(ddfbrush_dz)
        # absorption term derivative
        dM_absorb_dh = self.M * (ddFdh_dh - ddFdz_dh)
        dM_absorb_dz = self.M * (ddFdh_dz - ddFdz_dz)
        # derivatives of dynamic equations
        ddhdt_dh = self.nabla_F(dQhh_dh * diags(self.nabla0(dFdh)) +
                                Qhh * self.nabla0(ddFdh_dh), q) - dM_absorb_dh
        ddhdt_dz = self.nabla_F(Qhh * self.nabla0(ddFdh_dz), q) - dM_absorb_dz
        ddzdt_dh = self.nabla_F(Qzz * self.nabla0(ddFdz_dh), 0) + dM_absorb_dh
        ddzdt_dz = self.nabla_F(dQzz_dz * diags(self.nabla0(dFdz)) +
                                Qzz * self.nabla0(ddFdz_dz), 0) + dM_absorb_dz
        # combine and return
        an_jac = sp.bmat([[ddhdt_dh, ddhdt_dz],
                          [ddzdt_dh, ddzdt_dz]]).toarray()
        fd_jac = super().jacobian(u).toarray()
        plt.clf()
        plt.title("Analytical Jacobian")
        plt.imshow(np.abs(an_jac), cmap="Reds")
        plt.show()
        plt.title("FD Jacobian")
        plt.imshow(np.abs(fd_jac), cmap="Reds")
        plt.show()
        diff = an_jac - fd_jac
        plt.title("Difference")
        plt.imshow(np.abs(diff), cmap="Reds")
        plt.show()
        exit()
        return sp.bmat([[ddhdt_dh, ddhdt_dz],
                        [ddzdt_dh, ddzdt_dz]])

    def du_dx(self, u, direction=0):
        return self.nabla0(u)

    def plot(self, ax):
        global problem
        ax.set_ylim((0, 1.5))
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        x = self.x[0] - self.U*problem.time
        h, xi = self.u
        ax.plot(x, h+xi, markersize=5, label="liquid")
        ax.plot(x, xi, markersize=5, label="substrate")
        ax.legend()


class AdaptiveSubstrateProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = AdaptiveSubstrateEquation(N, L)
        self.add_equation(self.tfe)
        # initialize time stepper
        # self.time_stepper = time_steppers.BDF2(dt = 1e-5)
        self.time_stepper = time_steppers.BDF(self)
        # Generate the volume constraint
        self.volume_constraint = VolumeConstraint(self.tfe)
        # assign the continuation parameter
        # self.continuation_parameter = (self.tfe, "sigma")
        self.continuation_parameter = (self.tfe, "M")
        # self.continuation_parameter = (self.tfe, "U")

    # def norm(self):
    #     h, z = self.tfe.u
    #     return np.trapz(h, self.tfe.x[0])


# create output folder
shutil.rmtree("out", ignore_errors=True)
os.makedirs("out/img", exist_ok=True)

# create problem
problem = AdaptiveSubstrateProblem(N=512, L=8)

# create figure
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
plotID = 0

Profiler.start()

# time-stepping
n = 0
plotevery = 10
if not os.path.exists("initial_state.npz"):
    while problem.time_stepper.dt < 1e12 and problem.time < 1000:
        # plot
        if n % plotevery == 0:
            problem.plot(ax)
            fig.savefig("out/img/{:05d}.svg".format(plotID))
            plotID += 1
        print("step #: {:}".format(n))
        print("time:   {:}".format(problem.time))
        print("dt:     {:}".format(problem.time_stepper.dt))
        n += 1
        # perform timestep
        problem.time_step()
    Profiler.print_summary()
    # save the state, so we can reload it later
    problem.save("initial_state.npz")
else:
    # load the initial state
    problem.load("initial_state.npz")

# start parameter continuation
problem.continuation_stepper.ds = 1e-2
problem.continuation_stepper.ndesired_newton_steps = 2
# problem.continuation_stepper.convergence_tolerance = 1e-8
# problem.continuation_stepper.max_newton_iterations = 30
problem.settings.eigval_zero_tolerance = 1e-18
problem.settings.neigs = 50

# Impose the constraint
problem.add_equation(problem.volume_constraint)

n = 0
plotevery = 1
while True:
    # perform continuation step
    problem.continuation_step()
    n += 1
    print("step #:", n, " ds:", problem.continuation_stepper.ds)
    # plot
    if n % plotevery == 0:
        problem.plot(ax)
        fig.savefig("out/img/{:05d}.svg".format(plotID))
        plotID += 1
    # save bifurcation points
    if problem.bifurcation_diagram.current_solution().is_bifurcation():
        problem.save("sav/sol{}.npz".format(n))
    # mesh refinement
    # problem.adapt()

problem.save("final_state.npz")
