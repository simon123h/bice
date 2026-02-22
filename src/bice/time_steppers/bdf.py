"""Backward Differentiation Formula (BDF) time-stepping schemes."""

from typing import TYPE_CHECKING

import numpy as np
import scipy.integrate

from .time_steppers import TimeStepper

if TYPE_CHECKING:
    from bice.core.problem import Problem


class BDF2(TimeStepper):
    """
    Second-order Backward Differentiation Formula (BDF2) scheme.

    An implicit method for the numerical solution of ordinary differential
    equations. It is particularly well-suited for stiff systems.
    """

    def __init__(self, dt: float = 1e-3) -> None:
        """
        Initialize the BDF2 time-stepper.

        Parameters
        ----------
        dt
            The time step size.
        """
        super().__init__(dt)
        #: the order of the scheme
        self.order = 2

    def step(self, problem: "Problem") -> None:
        """
        Perform a single BDF2 step.

        Uses the problem's history to retrieve previous solutions for the
        multi-step formula.

        Parameters
        ----------
        problem
            The problem instance to step in time.
        """
        # advance in time
        problem.time += self.dt

        # recover the history (impulsive start, if history is missing)
        u_1 = problem.u
        u_2 = problem.history.u(1) if problem.history.length > 1 else u_1
        # obtain the problem's mass matrix
        M = problem.mass_matrix()

        def f(u):
            # assemble the system
            return self.dt * problem.rhs(u) - M.dot(3 * u - 4 * u_1 + u_2)

        def J(u):
            # Jacobian of the system
            return self.dt * problem.jacobian(u) - 3 * M

        # solve it with a Newton solver
        problem.u = problem.newton_solver.solve(f, problem.u, J)


class BDF(TimeStepper):
    """
    Variable-order Backward Differentiation Formula (BDF) scheme.

    A wrapper around `scipy.integrate.BDF` for adaptive time-stepping
    with variable order.
    """

    def __init__(self, problem: "Problem", dt_max: float = np.inf) -> None:
        """
        Initialize the adaptive BDF time-stepper.

        Parameters
        ----------
        problem
            The problem instance to solve.
        dt_max
            Maximum allowed time step size.
        """
        super().__init__()
        #: reference to the problem
        self.problem = problem
        #: relative tolerance for the solver
        self.rtol = 1e-5
        #: absolute tolerance for the solver
        self.atol = 1e-8
        #: maximum allowed time step size
        self.dt_max = dt_max
        # internal storage for the scipy BDF solver instance
        self.bdf = None
        self.factory_reset()

    def step(self, problem: "Problem") -> None:
        """
        Perform a single adaptive BDF step.

        Parameters
        ----------
        problem
            The problem instance to step in time.
        """
        # perform the step
        self.bdf.step()
        # assign the new variables
        self.dt = self.bdf.step_size
        self.problem.time = self.bdf.t
        self.problem.u = self.bdf.y

    def factory_reset(self) -> None:
        """
        Reset the underlying scipy BDF solver instance.

        Useful when the problem state changes significantly.
        """

        # create wrapper for the right-hand side
        def f(t, u):
            self.problem.time = t
            return self.problem.rhs(u)

        # create wrapper for the jacobian
        def jac(t, u):
            self.problem.time = t
            return self.problem.jacobian(u)

        # create instance of scipy.integrate.BDF
        self.bdf = scipy.integrate.BDF(
            f,
            self.problem.time,
            self.problem.u,
            self.problem.time + 1e18,
            jac=jac,
            max_step=self.dt_max,
            rtol=self.rtol,
            atol=self.atol,
            vectorized=False,
        )
