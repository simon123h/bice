"""Unit tests for the solver classes."""

import numpy as np
import pytest

from bice.core.solvers import MyNewtonSolver, NewtonSolver
from bice.core.types import Array, Matrix


def simple_quadratic(u: Array) -> Array:
    """f(u) = u^2 - 4. Root at u=2 and u=-2."""
    return u**2 - 4.0


def simple_quadratic_jacobian(u: Array) -> Matrix:
    """J(u) = 2u."""
    # Return as 2D array/matrix as expected by solvers usually?
    # Solvers expect J(u) * du = -f(u)
    # If u is scalar, J is scalar.
    # But usually solvers work with arrays.
    # Let's assume u is shape (1,)
    return np.diag(2 * u)


def system_2d(u: Array) -> Array:
    """
    Solve a 2D system of equations.

    f1 = x^2 + y^2 - 1 (circle radius 1)
    f2 = x - y (line x=y)
    Solutions: x=y=1/sqrt(2) approx 0.707.
    """
    x, y = u
    return np.array([x**2 + y**2 - 1, x - y])


def system_2d_jacobian(u: Array) -> Matrix:
    """Calculate the Jacobian of the 2D system."""
    x, y = u
    # J = [[2x, 2y], [1, -1]]
    return np.array([[2 * x, 2 * y], [1, -1]])


def test_mynewtonsolver_scalar() -> None:
    """Test MyNewtonSolver with a scalar equation."""
    solver = MyNewtonSolver()
    u0 = np.array([1.0])  # Guess close to 2

    # Adapt functions to handle array inputs
    def f(u: Array) -> Array:
        return simple_quadratic(u)

    def jac(u: Array) -> Matrix:
        return simple_quadratic_jacobian(u)

    sol = solver.solve(f, u0, jac)

    np.testing.assert_allclose(sol, [2.0], atol=1e-5)
    assert solver.niterations is not None and solver.niterations > 0


def test_mynewtonsolver_system() -> None:
    """Test MyNewtonSolver with a 2D system."""
    solver = MyNewtonSolver()
    u0 = np.array([0.5, 0.5])  # Guess inside circle

    sol = solver.solve(system_2d, u0, system_2d_jacobian)

    expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    np.testing.assert_allclose(sol, expected, atol=1e-5)


def test_newtonsolver_scipy_wrapper() -> None:
    """Test the wrapper around scipy.optimize.root."""
    solver = NewtonSolver()
    solver.method = "hybr"  # Default

    u0 = np.array([0.5, 0.5])

    sol = solver.solve(system_2d, u0, system_2d_jacobian)

    expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    np.testing.assert_allclose(sol, expected, atol=1e-5)

    # Check that it sets iterations
    assert solver.niterations is not None and solver.niterations > 0


def test_mynewtonsolver_no_convergence() -> None:
    """Test that it raises error if max iterations reached."""
    solver = MyNewtonSolver()
    solver.max_iterations = 2

    # Bad guess for x^2+1=0 (no real root), or just needs more steps
    # x^2 - 4 = 0 with bad guess or few steps
    u0 = np.array([1000.0])

    def f(u: Array) -> Array:
        return simple_quadratic(u)

    def jac(u: Array) -> Matrix:
        return simple_quadratic_jacobian(u)

    with pytest.raises(np.linalg.LinAlgError):
        solver.solve(f, u0, jac)
