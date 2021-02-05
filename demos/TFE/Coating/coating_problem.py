#!/usr/bin/python3
import numpy as np
from scipy.sparse import diags
from bice import Problem, time_steppers
from bice.pde import FiniteDifferencesEquation
from bice.pde.finite_differences import NeumannBC, DirichletBC, RobinBC, NoBoundaryConditions
from bice import profile


class CoatingEquation(FiniteDifferencesEquation):
    r"""
     Finite differences implementation of the 1-dimensional coating problem
     """

    def __init__(self, N, L):
        super().__init__(shape=(N,))
        # parameters:
        self.U = 0.5  # substrate velocity
        self.q = 0.06  # influx
        self.h_p = 0.04
        self.theta = np.sqrt(0.6)
        print("h_LL =", self.q/self.U)
        # setup the mesh
        self.L = L
        self.x = [np.linspace(0, L, N)]
        # initial condition
        # x = self.x[0]
        # incl = 2
        # hLL = self.q / self.U
        # h1 = 1-incl*(x[1] - x[0])
        # self.u = (h1 - hLL) * (1 + np.tanh(-incl * x)) + hLL
        self.u = np.ones(N)
        # self.u = np.maximum(1 - 0.0*x, self.h_p)
        # build finite differences matrices
        self.build_FD_matrices(approx_order=2)

    # overload building of FD matrices, because this equation has a more complicated set up
    def build_FD_matrices(self, approx_order):
        # build finite differences matrices...
        # (i) including the flux boundary conditions for h^3 * dF/dh
        self.bc = DirichletBC(vals=(1, 0))
        super().build_FD_matrices(approx_order)
        self.nabla_F = self.nabla
        # (ii) including the mixed boundary conditions for h (left Dirichlet, right Neumann)
        self.bc = RobinBC(a=(1, 0), b=(0, 1), c=(1, 0))
        super().build_FD_matrices(approx_order)
        self.nabla_h = self.nabla
        self.laplace_h = self.laplace
        # (iii) differentiation operators with no specific boundary effects
        self.bc = NoBoundaryConditions()
        super().build_FD_matrices(approx_order)
        self.nabla0 = self.nabla

    # definition of the equation
    def rhs(self, h):
        h3 = h**3
        # disjoining pressure
        djp = 5/3*(self.theta*self.h_p)**2 * (self.h_p**3/h3**2 - 1./h3)
        # free energy variation
        dFdh = -self.laplace_h(h) - djp
        # bulk flux
        flux = h3 * self.nabla0.dot(dFdh)
        # boundary flux
        j_in = self.U-self.q
        # dynamics equation, scale boundary condition with j_in
        dhdt = self.nabla_F(flux, j_in)
        # advection term
        dhdt -= self.U * self.nabla_h.dot(h)
        return dhdt

    def jacobian(self, h):
        # disjoining pressure
        h3 = h**3
        djp = 5/3*(self.theta*self.h_p)**2 * (self.h_p**3/h3**2 - 1./h3)
        ddjpdh = 5/3*(self.theta*self.h_p)**2 * \
            diags(3./h**4 - 6.*self.h_p**3/h**7)
        # free energy variation
        dFdh = -self.laplace_h(h) - djp
        ddFdhdh = -self.laplace_h() - ddjpdh
        # d(Qh^3*nabla*dFdh)/dh
        flux = diags(3*h**2 * self.nabla0.dot(dFdh)) + \
            diags(h3) * self.nabla0.dot(ddFdhdh)
        # dynamics equation, boundary condition is a constant --> scale with zero
        jac = self.nabla_F(flux, 0)
        jac -= self.U * self.nabla_h()
        return jac

    def du_dx(self, u, direction=0):
        return self.nabla[direction].dot(u)

    def plot(self, ax):
        global problem
        ax.set_xlabel("x")
        ax.set_ylabel("solution h(x,t)")
        x = self.x[0]
        h = self.u
        # add left ghost point (Dirichlet BC)
        x = np.concatenate(([x[0]-x[1]], x))
        h = np.concatenate(([1], h))
        # x -= self.U*problem.time
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0, 1.1*np.max(h))
        ax.plot(x, h)


class CoatingProblem(Problem):

    def __init__(self, N, L):
        super().__init__()
        # Add the Thin-Film equation to the problem
        self.tfe = CoatingEquation(N, L)
        self.add_equation(self.tfe)
        # initialize time stepper
        self.time_stepper = time_steppers.BDF2(dt=1)
        # self.time_stepper = time_steppers.BDF(self)

    def norm(self):
        return np.trapz(self.tfe.u, self.tfe.x[0])
