"""Deflation operator for detecting disconnected branches."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from bice.core.types import Array, Matrix


class DeflationOperator:
    """
    A deflation operator M for deflated continuation.

    Adds singularities to the equation at given solutions u_i:
    0 = F(u) --> 0 = M(u) * F(u)
    with
    M(u) = product_i <u_i - u, u_i - u>^-p + shift

    The parameters are:
      p: some exponent to the norm <u, v>
      shift: some constant added shift parameter for numerical stability
    """

    def __init__(self) -> None:
        """Initialize the DeflationOperator."""
        #: the order of the norm that will be used for the deflation operator
        self.p = 2
        #: small constant in the deflation operator, for numerical stability
        self.shift = 0.5
        #: list of solutions, that will be suppressed by the deflation operator
        self.solutions: list[Array] = []

    def operator(self, u: Array) -> float:
        """
        Obtain the value of the deflation operator for given u.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        float
            The value of the operator.
        """
        if not self.solutions:
            return 1.0 + self.shift
        return float(np.prod([np.dot(u_i - u, u_i - u) ** -self.p for u_i in self.solutions]) + self.shift)

    def D_operator(self, u: Array) -> Array:
        """
        Calculate the Jacobian of deflation operator for given u.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        Array
            The gradient of the operator.
        """
        if not self.solutions:
            return np.zeros_like(u)
        op = self.operator(u)
        return np.asanyarray(self.p * op * 2 * np.sum([(uk - u) / np.dot(uk - u, uk - u) for uk in self.solutions], axis=0))

    def deflated_rhs(self, rhs: Callable[[Array], Array]) -> Callable[[Array], Array]:
        """
        Deflate the rhs of some equation.

        Returns a new function that represents M(u) * rhs(u).

        Parameters
        ----------
        rhs
            The original right-hand side function.

        Returns
        -------
        callable
            The deflated rhs function.
        """

        def new_rhs(u: Array) -> Array:
            # multiply rhs with deflation operator
            return self.operator(u) * rhs(u)

        # return the function object
        return new_rhs

    def deflated_jacobian(self, rhs: Callable[[Array], Array], jacobian: Callable[[Array], Matrix]) -> Callable[[Array], Matrix]:
        """
        Generate Jacobian of deflated rhs of some equation or problem.

        Parameters
        ----------
        rhs
            The original right-hand side function.
        jacobian
            The original Jacobian function.

        Returns
        -------
        callable
            The deflated Jacobian function.
        """

        def new_jac(u: Array) -> Matrix:
            # obtain operator and operator derivative
            op = self.operator(u)
            D_op = self.D_operator(u)
            # calculate derivative d/du
            return sp.diags(D_op * rhs(u)) + op * jacobian(u)

        # return the function object
        return new_jac

    def add_solution(self, u: Array) -> None:
        """
        Add a solution to the list of solutions used for deflation.

        Parameters
        ----------
        u
            The solution to deflate.
        """
        self.solutions.append(u)

    def remove_solution(self, u: Array) -> None:
        """
        Remove a solution from the list of solutions used for deflation.

        Parameters
        ----------
        u
            The solution to remove.
        """
        self.solutions.remove(u)

    def clear_solutions(self) -> None:
        """Clear the list of solutions used for deflation."""
        self.solutions = []
