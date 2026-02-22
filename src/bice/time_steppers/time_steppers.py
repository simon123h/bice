"""Base classes for time-stepping algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from bice.core.types import Array, Matrix

if TYPE_CHECKING:
    from bice.core.problem import Problem


class TimeStepper:
    """
    Abstract base class for all time-steppers.

    Specifies attributes and methods that all time-steppers should have.
    """

    def __init__(self, dt: float = 1e-2) -> None:
        """
        Initialize the time-stepper.

        Parameters
        ----------
        dt
            The time step size.
        """
        #: the time step size
        self.dt = dt

    def step(self, problem: Problem) -> None:
        """
        Perform a single time step on a problem.

        Parameters
        ----------
        problem
            The problem instance to step in time.

        Raises
        ------
        NotImplementedError
            This is an abstract base class.
        """
        raise NotImplementedError("'TimeStepper' is an abstract base class - do not use for actual time-stepping!")


class Euler(TimeStepper):
    """
    Explicit Euler (Forward Euler) scheme.

    A first-order numerical procedure for solving ordinary differential equations
    with a given initial value.
    """

    def step(self, problem: Problem) -> None:
        """
        Perform a single explicit Euler step.

        Parameters
        ----------
        problem
            The problem instance to step in time.
        """
        problem.u = problem.u + self.dt * problem.rhs(problem.u)
        problem.time += self.dt


class ImplicitEuler(TimeStepper):
    """
    Implicit Euler (Backward Euler) scheme.

    A first-order implicit method for solving ordinary differential equations,
    offering better stability for stiff systems compared to the explicit Euler method.
    """

    def step(self, problem: Problem) -> None:
        """
        Perform a single implicit Euler step.

        Uses the problem's Newton solver to find the solution at the next time level.

        Parameters
        ----------
        problem
            The problem instance to step in time.
        """
        # advance in time
        problem.time += self.dt
        # obtain the mass matrix
        M = problem.mass_matrix()

        def f(u: Array) -> Array:
            # assemble the system
            return cast(Array, problem.rhs(u) - M.dot(u - problem.u) / self.dt)

        def J(u: Array) -> Matrix:
            # Jacobian of the system
            return cast(Matrix, problem.jacobian(u) - M / self.dt)

        # solve it with a Newton solver
        # TODO: detect if Newton solver failed and reject step
        problem.u = problem.newton_solver.solve(f, problem.u, J)
