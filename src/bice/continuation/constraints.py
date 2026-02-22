"""Predefined constraint equations for continuation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from bice.core.equation import Equation
from bice.core.types import Array, Axes, Shape

if TYPE_CHECKING:
    from bice.pde import PartialDifferentialEquation


class ConstraintEquation(Equation):
    """
    Abstract base class for constraint type equations.

    For simple implementation of PDE constraints and less redundant code.
    """

    def __init__(self, shape: Shape = (1,)) -> None:
        """
        Initialize the ConstraintEquation.

        Parameters
        ----------
        shape
            The shape of the constraint unknowns (Lagrange multipliers).
        """
        # default shape: (1,)
        super().__init__(shape=shape)
        # constraints typically couple to some other equation
        self.is_coupled = True

    def mass_matrix(self) -> float:
        """
        Return the mass matrix contribution.

        Returns
        -------
        float
            Always 0 as constraints usually couple to no time-derivatives.
        """
        # constraint usually couples to no time-derivatives
        return 0.0

    def plot(self, ax: Axes) -> None:
        """
        Plot the constraint state (no-op).

        Parameters
        ----------
        ax
            The matplotlib axes.
        """
        # nothing to plot


class VolumeConstraint(ConstraintEquation):
    """
    Assures the conservation of the integral of the unknowns.

    A volume constraint (or mass constraint) assures the conservation of
    the integral of the unknowns of some given equation when solving the system.
    We may even prescribe the target volume (or mass) with a parameter,
    but we don't have to.
    The constraint equation comes with an additional (unknown) Lagrange
    multiplier that can be interpreted as an influx into the system.
    """

    def __init__(self, reference_equation: Equation, variable: int | None = None) -> None:
        """
        Initialize the VolumeConstraint.

        Parameters
        ----------
        reference_equation
            The equation to constrain.
        variable
            The index of the variable to constrain (if the equation has multiple).
        """
        super().__init__(shape=(1,))
        #: on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        #: on which variable (index) of the equation should the constraint be imposed?
        self.variable = variable
        #: this equation brings a single extra degree of freedom (influx Lagrange
        #: multiplier)
        self.u = np.zeros(1)
        #: This parameter allows for prescribing a fixed volume (unless it is None)
        self.fixed_volume: float | None = None

    def rhs(self, u: Array) -> Array:
        """
        Calculate the residuals of the volume constraint.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        Array
            The residuals vector.
        """
        # generate empty vector of residual contributions
        res = np.zeros(u.size)
        # reference to the indices of the unknowns that we work on
        self_idx = self.group.idx[self]
        eq_idx = self.group.idx[self.ref_eq]
        # optionally split only the part that is referenced by self.variable
        if self.variable is not None:
            eq_shape = self.ref_eq.shape[1:]
            var_ndofs = np.prod(eq_shape)
            start = eq_idx.start + self.variable * var_ndofs
            eq_idx = slice(int(start), int(start + var_ndofs))
        # employ the constraint equation
        if self.fixed_volume is None:
            # calculate the difference in volumes between current
            # and previous unknowns of the reference equation
            # we use the first entry in u_history as the reference state
            u_old = self.ref_eq.u_history[0].ravel() if self.ref_eq.u_history else self.group.u[eq_idx]
            res[self_idx] = np.mean(u[eq_idx] - u_old)
        else:
            # parametric constraint: calculate the difference between current
            # volume and the prescribed fixed_volume parameter
            x = [np.arange(self.ref_eq.shape[-1])]
            if hasattr(self.ref_eq, "x") and getattr(self.ref_eq, "x") is not None:
                x = getattr(self.ref_eq, "x")

            # Use np.trapezoid (NumPy 2.0+) or fallback to np.trapz
            trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
            if trapezoid is None:
                raise AttributeError("Neither np.trapezoid nor np.trapz found.")

            # ensure x is passed correctly to trapezoid (expects 1d array for each axis or just 1d)
            x_vals = x[0] if isinstance(x, list) else x
            res[self_idx] = trapezoid(u[eq_idx], x_vals) - self.fixed_volume
        # Add the constraint to the reference equation: unknown influx is the
        # Langrange multiplier
        res[eq_idx] += u[self_idx]
        return res

    def jacobian(self, u: Array) -> sp.csr_matrix:
        """
        Calculate the analytical Jacobian of the volume constraint.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        sp.csr_matrix
            The Jacobian matrix.
        """
        N = u.size
        # reference to the indices of the unknowns that we work on
        self_idx = self.group.idx[self]
        eq_idx = self.group.idx[self.ref_eq]

        # optionally split only the part that is referenced by self.variable
        if self.variable is not None:
            eq_shape = self.ref_eq.shape[1:]
            var_ndofs = np.prod(eq_shape)
            start = eq_idx.start + self.variable * var_ndofs
            eq_idx = slice(int(start), int(start + var_ndofs))

        # generate empty matrix of residual contributions
        J = sp.lil_matrix((N, N))

        # employ the constraint equation
        if self.fixed_volume is None:
            # d R_vol / d u_i = 1 / nvars
            nvars = eq_idx.stop - eq_idx.start
            J[self_idx, eq_idx] = 1.0 / nvars
        else:
            # d R_vol / d u_i are the weights of the trapezoidal rule
            x = [np.arange(self.ref_eq.shape[-1])]
            if hasattr(self.ref_eq, "x") and getattr(self.ref_eq, "x") is not None:
                x = getattr(self.ref_eq, "x")

            # weights for 1D trapezoidal rule:
            x_vals = x[0]
            dx = np.diff(x_vals)
            weights = np.zeros(len(x_vals))
            weights[0] = 0.5 * dx[0]
            weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
            weights[-1] = 0.5 * dx[-1]
            # TODO: handle multi-dimensional trapz weights if needed

            # if we have multiple variables, and no specific variable is selected,
            # trapz might be applied to each or all.
            # for now, assume single variable or select specific variable
            J[self_idx, eq_idx] = weights

        # Add the contribution to the reference equation:
        # res[eq_idx] = u[self_idx]
        # d R_eq_i / d lambda = 1
        J[eq_idx, self_idx] = 1.0

        return J.tocsr()


class TranslationConstraint(ConstraintEquation):
    """
    Assures that the center of mass does not move.

    A translation constraint assures that the center of mass of some
    reference equation's unknowns does not move when solving the system.
    The additional constraint equations (one per spatial dimension) come
    with Lagrange multipliers, that correspond to the velocities of a comoving
    frame (advection term).
    """

    def __init__(
        self,
        reference_equation: PartialDifferentialEquation,
        variable: int | None = None,
        direction: int = 0,
    ) -> None:
        """
        Initialize the TranslationConstraint.

        Parameters
        ----------
        reference_equation
            The equation to constrain.
        variable
            The variable index to constrain.
        direction
            The spatial direction (index) to apply the constraint to.
        """
        # call parent constructor
        super().__init__(shape=(1,))
        #: on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        #: on which variable (index) of the equation should the constraint be imposed?
        self.variable = variable
        #: which spatial direction (index of [x, y, ...]) should the constraint apply to
        self.direction = direction
        #: the unknowns (velocity vector)
        self.u = np.zeros(1)

    def rhs(self, u: Array) -> Array:
        """
        Calculate the residuals of the translation constraint.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        Array
            The residuals vector.
        """
        # set up the vector of the residual contributions
        res = np.zeros(u.size)
        # reference to the equation, shape and indices of the unknowns that we work on
        eq = self.ref_eq
        eq_shape = eq.shape
        eq_idx = self.group.idx[eq]
        self_idx = self.group.idx[self]
        # optionally split only the part that is referenced by self.variable
        if self.variable is not None:
            eq_shape = self.ref_eq.shape[1:]
            var_ndofs = np.prod(eq_shape)
            start = eq_idx.start + self.variable * var_ndofs
            eq_idx = slice(start, start + var_ndofs)
        # obtain the unknowns
        eq_u = u[eq_idx]
        eq_u_old = self.group.u[eq_idx]
        velocity = u[self_idx]
        # add constraint to residuals of reference equation (velocity is the
        # lagrange multiplier)
        try:  # if method du_dx is implemented, use this
            eq_dudx = eq.du_dx(eq_u.reshape(eq_shape), self.direction).ravel()
        except AttributeError:  # if not, get it from the gradient
            assert eq.x is not None
            eq_dudx = np.gradient(eq_u, eq.x[self.direction])
        res[eq_idx] = velocity * eq_dudx
        # calculate the difference in center of masses between current
        # and previous unknowns of the reference equation
        # res[self_idx] = np.dot(eq.x[self.direction], eq_u-eq_u_old)
        res[self_idx] = np.dot(eq_dudx, (eq_u - eq_u_old))
        return res

    def jacobian(self, u: Array) -> sp.csr_matrix:
        """
        Calculate the analytical Jacobian of the translation constraint.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        sp.csr_matrix
            The Jacobian matrix.
        """
        N = u.size
        # reference to the equation, shape and indices of the unknowns that we work on
        eq = self.ref_eq
        eq_shape = eq.shape
        eq_idx = self.group.idx[eq]
        self_idx = self.group.idx[self]

        # optionally split only the part that is referenced by self.variable
        if self.variable is not None:
            eq_shape = self.ref_eq.shape[1:]
            var_ndofs = np.prod(eq_shape)
            start = eq_idx.start + self.variable * var_ndofs
            eq_idx = slice(start, int(start + var_ndofs))

        # obtain the unknowns
        eq_u = u[eq_idx]
        eq_u_old = self.group.u[eq_idx]
        velocity = u[self_idx]

        # obtain the differentiation matrix D
        # eq_dudx = D(u) = Q * u + G
        if hasattr(eq, "nabla") and eq.nabla is not None:
            # for FiniteDifferencesEquation
            if isinstance(eq.nabla, list):  # multi-dimensional case
                D_aff = eq.nabla[self.direction]
            else:  # 1d case
                D_aff = eq.nabla
            # extraction of the matrix part Q
            from bice.pde.finite_differences import AffineOperator

            if isinstance(D_aff, AffineOperator):
                D = D_aff.Q
                G = D_aff.G
            else:
                D = D_aff
                G = 0
        else:
            # fallback to FD approximation if no analytical matrix exists
            return sp.csr_matrix(super().jacobian(u))

        # create a sparse matrix of the Jacobian contributions
        J = sp.lil_matrix((N, N))

        # d R_eq / d v = D(u) = Q * u + G
        eq_dudx = D.dot(eq_u) + G
        J[eq_idx, self_idx] = eq_dudx.reshape((-1, 1))

        # d R_eq / d u = v * Q
        J[eq_idx, eq_idx] = velocity[0] * D

        # d R_trans / d u = (Q * u + G) + Q^T * (u - u_old)
        row = eq_dudx + D.T.dot(eq_u - eq_u_old)
        J[self_idx, eq_idx] = row

        # d R_trans / d v = 0 (already zero)

        return J.tocsr()
