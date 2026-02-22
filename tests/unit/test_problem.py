"""Sociable unit tests for the Problem class."""

from typing import Any, cast

import numpy as np

from bice.core.equation import Equation
from bice.core.problem import Problem
from bice.core.solvers import MyNewtonSolver
from bice.core.types import Array, ArrayLike, Shape


class SimpleLinearEquation(Equation):
    """du/dt = a * u + b"""

    def __init__(self, a: float = -1.0, b: float = 1.0, shape: Shape = (1,)) -> None:
        super().__init__(shape=shape)
        self.a = a
        self.b = b

    def rhs(self, u: Array) -> Array:
        return self.a * u + self.b

    def jacobian(self, u: Array) -> Array:
        return np.asarray([[self.a]])


def test_problem_initialization() -> None:
    """Test that a Problem initializes with default components."""
    prob = Problem()
    assert prob.eq is None
    assert prob.time == 0.0
    assert prob.time_stepper is not None
    assert prob.newton_solver is not None
    assert prob.history is not None


def test_problem_add_remove_equation() -> None:
    """Test adding and removing equations to/from a Problem."""
    prob = Problem()
    eq1 = SimpleLinearEquation(a=-1.0, b=1.0, shape=(2,))

    prob.add_equation(eq1)
    assert prob.eq is eq1
    assert prob.ndofs == 2

    eq2 = SimpleLinearEquation(a=-2.0, b=0.0, shape=(3,))
    prob.add_equation(eq2)
    # Adding a second equation should wrap them in an EquationGroup
    from bice.core.equation import EquationGroup

    assert isinstance(prob.eq, EquationGroup)
    assert prob.ndofs == 5

    prob.remove_equation(eq1)
    # After removing one, if only one is left, it might still be a group or just the remaining eq
    # Based on problem.py implementation, it calls eq.remove_equation(eq) if it's a group.
    assert eq1 not in prob.list_equations()
    assert prob.ndofs == 3

    prob.remove_equation(eq2)
    assert prob.eq is None
    assert prob.ndofs == 0


def test_problem_u_property() -> None:
    """Test the u getter and setter."""
    prob = Problem()
    eq = SimpleLinearEquation(shape=(2,))
    prob.add_equation(eq)

    new_u: ArrayLike = np.array([1.5, 2.5])
    prob.u = new_u
    np.testing.assert_allclose(cast(np.ndarray, prob.u), cast(np.ndarray, new_u))
    np.testing.assert_allclose(cast(np.ndarray, eq.u), cast(np.ndarray, new_u))


def test_problem_time_step() -> None:
    """Test performing a time step."""
    prob = Problem()
    eq = SimpleLinearEquation(a=-1.0, b=0.0, shape=(1,))
    prob.add_equation(eq)
    prob.u = np.array([1.0])

    initial_u = prob.u.copy()
    prob.time_step()

    # After one step of RK4 (default) for du/dt = -u, u should decrease
    assert prob.u[0] < initial_u[0]
    assert prob.time > 0


def test_problem_newton_solve() -> None:
    """Test solving for steady state using Newton's method."""
    prob = Problem()
    # du/dt = -u + 2 => steady state at u = 2
    eq = SimpleLinearEquation(a=-1.0, b=2.0, shape=(1,))
    prob.add_equation(eq)
    prob.newton_solver = cast(Any, MyNewtonSolver())
    prob.u = np.array([0.0])

    prob.newton_solve()
    np.testing.assert_allclose(prob.u, [2.0], atol=1e-5)


def test_problem_history() -> None:
    """Test that history is updated during time steps."""
    prob = Problem()
    eq = SimpleLinearEquation(shape=(1,))
    prob.add_equation(eq)
    prob.u = np.array([1.0])

    assert prob.history.length == 0
    prob.time_step()
    assert prob.history.length == 1
    # history.u(0) should return the u *before* the last update call in time_step
    # actually update() is called at the BEGINNING of time_step()
    # So history.u(0) is the state before step() was called.
    np.testing.assert_allclose(prob.history.u(0), [1.0])
