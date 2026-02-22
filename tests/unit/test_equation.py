import numpy as np

from bice.core.equation import Equation


class LinearEquation(Equation):
    """
    A simple linear equation: du/dt = u - 1
    So rhs(u) = u - 1
    Jacobian should be Identity.
    """

    def rhs(self, u):
        return u - 1


def test_equation_jacobian_fd():
    """
    Test that the base Equation class correctly computes the Jacobian
    using finite differences for a simple linear equation.
    """
    N = 5
    eq = LinearEquation(shape=(N,))

    # Set a random state for u
    eq.u = np.random.rand(N)

    # Compute Jacobian using the default FD implementation
    J = eq.jacobian(eq.u)

    # The Jacobian of rhs(u) = u - 1 is the Identity matrix
    expected_J = np.eye(N)

    # Check if J matches expected_J
    # J might be sparse or dense depending on implementation details,
    # but Equation.jacobian returns a numpy array (dense) or sparse matrix.
    # The current implementation in Equation.jacobian returns a numpy array (J is initialized as zeros).

    np.testing.assert_allclose(
        J,
        expected_J,
        atol=1e-8,
        err_msg="Jacobian should be identity for linear equation",
    )


def test_equation_shape():
    """Test shape handling."""
    eq = Equation(shape=(10, 2))
    assert eq.shape == (10, 2)
    assert eq.ndofs == 20

    eq.reshape((5, 4))
    assert eq.shape == (5, 4)
    assert eq.ndofs == 20
