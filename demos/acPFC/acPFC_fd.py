#!/usr/bin/python3
"""
Finite difference implementation of the 1-dimensional semiactive coupled PFC equation.
This code does nothing with the equation, it only provides the implementation and
is imported by other codes, so we don't have to write the SHE from scratch for every demo.
"""
import sys
import numpy as np
from scipy.sparse import diags
import scipy.sparse as sp
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, time_steppers
from bice.pde.finite_differences import FiniteDifferencesEquation, PeriodicBC
from bice.continuation import ConstraintEquation
from bice import profile

class activePFCEquation(FiniteDifferencesEquation):
    r"""
    Finite difference implementation of the 1-dimensional semiactive coupled PFC equation
    """

    def __init__(self, N, L):
        super().__init__(shape=(3, N))
        # parameters
        self.r = -1.5
        self.phi01 = -0.64
        self.phi02 = -0.64
        self.q1 = 1.0
        self.q2 = 1.0
        self.c = -0.2
        self.v0 = 0.0
        self.C1 = 0.1
        self.Dr = 0.5

        # spatial coordinate
        self.x = [np.linspace(-L/2, L/2, N)]
        # initial condition
        u0 = np.cos(self.x[0]) * np.exp(-0.005 * self.x[0] ** 2)
        u0 = u0 - u0.mean()
        self.u = np.array([u0 + self.phi01, u0 + self.phi02, 0*u0])
        # build finite difference matrices
        self.bc = PeriodicBC() 
        self.build_FD_matrices(approx_order=2)
        laplace = self.laplace()
        self.linear_op1 = 2*self.q1**2 * laplace + laplace.dot(laplace)
        self.linear_op2 = 2*self.q2**2 * laplace + laplace.dot(laplace)


    # definition of the acPFC (right-hand side)
    def rhs(self, u):
        laplace = self.laplace()
        f1 = laplace.dot(self.linear_op1.dot(u[0]) + (self.r + self.q1**4) * u[0] + u[0]**3 + self.c*u[1]) - self.v0*self.nabla.dot(u[2])
        f2 = laplace.dot(self.linear_op2.dot(u[1]) + (self.r + self.q2**4) * u[1] + u[1]**3 + self.c*u[0])
        f3 = self.C1*laplace.dot(u[2]) - self.Dr*self.C1*u[2] - self.v0*self.nabla.dot(u[0])
        return np.array([f1, f2, f3])

    # definition of the Jacobian
    # def jacobian(self, u):
    #     laplace = self.laplace()

    #     f1u1 = laplace.dot(self.linear_op1 + (self.r - self.q1**4) + 3*(u[0] + self.phi01)**2)
    #     f1u2 = laplace*self.c
    #     f1u3 = -self.v0*self.nabla
    #     f2u1 = laplace*self.c
    #     f2u2 = laplace.dot(self.linear_op2 + (self.r - self.q2**4) + 3*(u[1] + self.phi02)**2)
    #     f2u3 = 0*laplace
    #     f3u1 = -self.v0*self.nabla
    #     f3u2 = 0*laplace
    #     f3u3 = self.C1*laplace - self.Dr*self.C1

    #     return sp.bmat([[f1u1, f1u2, f1u3], [f2u1, f2u2, f2u3], [f3u1, f3u2, f3u3]])

    def du_dx(self, u, direction=0):
        return self.nabla[direction](u)

    def plot(self, ax):
        ax.clear()
        ax.set_xlabel("x")
        ax.set_ylabel(r"solution $u_i(x,t)$")
        ax.plot(self.x[0], self.u[0], label=r"$\phi_1$")
        ax.plot(self.x[0], self.u[1], label=r"$\phi_2$")
        ax.plot(self.x[0], self.u[2], label=r"$\mathbf{P}$")
        ax.legend()
    
    def gauss(self, mu, sigma=1.5):
        return np.exp(-(self.x[0]-mu)**2/(2.*sigma**2))/np.sqrt(2.*np.pi*sigma**2)

    def add_gauss_to_sol(self, index):
        cond = True
        try:
            gauss_pos = input(f'phi{index+1:1d}: position for gauss peak\n')
            gauss_fac = input(f'phi{index+1:1d}: height of gauss peak\n')
            gauss_pos = float(gauss_pos)
            gauss_fac = float(gauss_fac)
            u = self.u[index]
            u -= u.mean()
            u += self.gauss(gauss_pos)*gauss_fac
            u -= self.u[index].mean()
            if index == 0:
                u += self.phi01
            if index == 1:
                u += self.phi02
        except ValueError:
            cond = False
        return cond


class acPFC(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.acpfc = activePFCEquation(N, L)
        self.add_equation(self.acpfc)
        # Generate the volume constraint
        # self.volume_constraint = VolumeConstraint(self.tfe, variable=0)
        # Generate the translation constraint
        # self.translation_constraint = TranslationConstraint(self.tfe)
        # initialize time stepper
        # self.time_stepper = time_steppers.BDF2(dt=1e-4)
        # self.time_stepper = time_steppers.ImplicitEuler(dt=1e-3)
        self.time_stepper = time_steppers.BDF(self)
        # assign the continuation parameter
        #self.continuation_parameter = (self.acpfc.phi01, "phi01")
        # self.newton_solver = MyNewtonSolver()
        # self.newton_solver = NewtonSolver()
        #self.newton_solver = NewtonKrylovSolver()
        # self.newton_solver.approximate_jacobian = True
        self.newton_solver.convergence_tolerance = 6e-6
        self.newton_solver.max_newton_iterations = 100
        self.newton_solver.verbosity = 0

    def norm(self):
        return np.linalg.norm(self.u)
