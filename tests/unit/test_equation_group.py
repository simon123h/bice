import numpy as np
import pytest
import scipy.sparse as sp

from bice.core.equation import Equation, EquationGroup


class SimpleEquation(Equation):
    """
    A simple equation: rhs(u) = u * multiplier
    """

    def __init__(self, shape, multiplier=1.0):
        super().__init__(shape)
        self.multiplier = multiplier
        self.u = np.ones(shape)

    def rhs(self, u):
        return u * self.multiplier

    def jacobian(self, u):
        # Jacobian is diagonal with entries = multiplier
        return sp.diags([self.multiplier] * u.size, shape=(u.size, u.size))


class CoupledEquation(Equation):
    """
    An equation that depends on another equation.
    rhs(u_this) = u_this - u_other
    """

    def __init__(self, shape, other_eq):
        super().__init__(shape)
        self.other_eq = other_eq
        self.is_coupled = True
        self.u = np.zeros(shape)

    def rhs(self, u_full):
        # We need to extract our part and the other part from u_full
        # This requires knowledge of the group mapping, which is available in self.group.idx
        if self.group is None:
            raise RuntimeError("Coupled equation must be in a group")

        my_idx = self.group.idx[self]
        other_idx = self.group.idx[self.other_eq]

        u_me = u_full[my_idx]
        u_other = u_full[other_idx]

        # Result must be full size, but only our entries are non-zero?
        # No, EquationGroup.rhs accumulates results.
        # But wait, EquationGroup.rhs logic for coupled equations is:
        # res += eq.rhs(u)
        # So rhs must return a vector of size full_ndofs!

        res = np.zeros_like(u_full)
        res[my_idx] = u_me - u_other
        return res

    def jacobian(self, u_full):
        # Jacobian needs to be full size
        my_idx = self.group.idx[self]
        other_idx = self.group.idx[self.other_eq]

        ndofs = u_full.size
        # Diagonal part for me: I
        # Off-diagonal part for other: -I

        # We can construct this sparsely
        # Indices for me
        rows_me = np.arange(my_idx.start, my_idx.stop)
        cols_me = np.arange(my_idx.start, my_idx.stop)

        # Indices for other
        cols_other = np.arange(other_idx.start, other_idx.stop)

        data = np.concatenate([np.ones(len(rows_me)), -np.ones(len(rows_me))])
        rows = np.concatenate([rows_me, rows_me])
        cols = np.concatenate([cols_me, cols_other])

        return sp.csr_matrix((data, (rows, cols)), shape=(ndofs, ndofs))


def test_equation_group_mapping():
    """Test that EquationGroup maps unknowns correctly."""
    eq1 = SimpleEquation(shape=(2,), multiplier=2.0)
    eq2 = SimpleEquation(shape=(3,), multiplier=3.0)

    group = EquationGroup([eq1, eq2])

    assert group.ndofs == 5
    assert group.u.shape == (5,)

    # Test setting group u updates individual equations
    new_u = np.arange(5, dtype=float)
    group.u = new_u

    np.testing.assert_array_equal(eq1.u, new_u[0:2])
    np.testing.assert_array_equal(eq2.u, new_u[2:5])

    # Test updating individual equations updates group u property
    eq1.u[:] = -1
    np.testing.assert_array_equal(group.u[0:2], [-1, -1])


def test_equation_group_rhs_uncoupled():
    """Test RHS assembly for uncoupled equations."""
    eq1 = SimpleEquation(shape=(2,), multiplier=2.0)
    eq2 = SimpleEquation(shape=(2,), multiplier=3.0)

    group = EquationGroup([eq1, eq2])

    # u = [1, 1, 1, 1]
    # rhs1 = u1 * 2 = [2, 2]
    # rhs2 = u2 * 3 = [3, 3]
    # group rhs = [2, 2, 3, 3]

    rhs = group.rhs(group.u)
    expected = np.array([2.0, 2.0, 3.0, 3.0])

    np.testing.assert_array_equal(rhs, expected)


def test_equation_group_jacobian_uncoupled():
    """Test Jacobian assembly for uncoupled equations."""
    eq1 = SimpleEquation(shape=(2,), multiplier=2.0)
    eq2 = SimpleEquation(shape=(2,), multiplier=3.0)

    group = EquationGroup([eq1, eq2])

    J = group.jacobian(group.u)
    J_dense = J.toarray()

    expected = np.diag([2.0, 2.0, 3.0, 3.0])

    np.testing.assert_array_equal(J_dense, expected)


def test_equation_group_coupled():
    """Test RHS and Jacobian for coupled equations."""
    eq1 = SimpleEquation(shape=(1,), multiplier=1.0)  # u1
    eq2 = CoupledEquation(shape=(1,), other_eq=eq1)  # u2 - u1

    group = EquationGroup([eq1, eq2])

    # Initial u = [1.0, 0.0] (eq1 init ones, eq2 init zeros)

    # RHS:
    # eq1: u1 * 1 = 1.0
    # eq2: u2 - u1 = 0.0 - 1.0 = -1.0
    # total = [1.0, -1.0]

    rhs = group.rhs(group.u)
    np.testing.assert_array_equal(rhs, [1.0, -1.0])

    # Jacobian:
    # eq1 contributes: [1  0]
    #                  [0  0] (in full space)
    #
    # eq2 contributes: [-1 0] (u2 depends on -u1)
    #                  [0  1] (u2 depends on +u2)
    # Wait, CoupledEquation.jacobian implementation returns full matrix.
    # Row 0 (eq1) is handled by eq1 (uncoupled logic in group) -> [1 0]
    # Row 1 (eq2) is handled by eq2 (coupled logic) -> [-1 1]

    # Group logic adds them up.
    # J = [1 0] (from eq1 uncoupled block)
    #     [0 0]
    #   +
    #     [0  0]
    #     [-1 1] (from eq2 coupled)
    #
    # Result:
    # [1  0]
    # [-1 1]

    J = group.jacobian(group.u)
    J_dense = J.toarray()

    expected = np.array([[1.0, 0.0], [-1.0, 1.0]])
    np.testing.assert_array_equal(J_dense, expected)
