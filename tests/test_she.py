#!/usr/bin/python3
import numpy as np
from bice import Problem, time_steppers
from bice.pde.finite_differences import FiniteDifferencesEquation, PeriodicBC
from bice.continuation import TranslationConstraint
from bice import profile
import unittest


class SwiftHohenbergEquation(FiniteDifferencesEquation):
    r"""
    Finite difference implementation of the 1-dimensional Swift-Hohenberg Equation
    equation, a nonlinear PDE
    \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
    """

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.r = -0.013
        self.kc = 0.5
        self.v = 0.41
        self.g = 1
        # spatial coordinate
        self.x = [np.linspace(-L/2, L/2, N)]
        # initial condition
        self.u = np.cos(2 * np.pi * self.x[0] / 10) * \
            np.exp(-0.005 * self.x[0] ** 2)
        # build finite difference matrices
        self.bc = PeriodicBC()
        self.build_FD_matrices(approx_order=2)
        laplace = self.laplace()
        self.linear_op = -2 * self.kc**2 * laplace - laplace.dot(laplace)

    # definition of the SHE (right-hand side)
    @profile
    def rhs(self, u):
        return self.linear_op.dot(u) + (self.r - self.kc**4) * u + self.v * u**2 - self.g * u**3


class TestSwiftHohenbergEquation(unittest.TestCase):
    """
    A simple test using the Swift-Hohenberg equation (SHE) testing if it is possible to:
    - create a FiniteDifferencesEquation
    - create a Problem
    - perform time simulation
    - perform parameter continuation
    """

    def test_SHE(self):
        """
        Run the test for the Swift-Hohenberg equation.
        """
        # create problem
        self.problem = Problem()
        self.she = SwiftHohenbergEquation(N=128, L=240)
        self.problem.add_equation(self.she)
        self.problem.time_stepper = time_steppers.BDF(self.problem)

        # time-stepping
        print("Time stepping...")
        n = 0
        while np.linalg.norm(self.problem.rhs(self.problem.u)) > 1e-5:
            self.problem.time_step()
            n += 1
            if n > 1e4:
                raise Exception(
                    f"Time stepping did not converge after {n:d} steps!")
        print(f"Time stepping finished after {n} steps.")

        # add translation constraint
        constraint = TranslationConstraint(self.she)
        self.problem.add_equation(constraint)

        # parameter contiuation
        self.problem.settings.neigs = 6
        self.problem.continuation_parameter = (self.she, "r")
        print("Parameter continuation...")
        for i in range(10):
            print(i, self.problem.get_continuation_parameter(), self.problem.norm())
            self.problem.continuation_step()
        print("Parameter continuation finished.")


# run the test if called directly
if __name__ == '__main__':
    unittest.main()
