from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.sparse as sp

from bice.core.equation import Equation
from bice.core.types import Array, Shape

if TYPE_CHECKING:
    from bice.pde import PartialDifferentialEquation


class ConstraintEquation(Equation):
    """
    Abstract base class for constraint type equations.
    For simple implementation of PDE constraints and less redundant code.
    """

    def __init__(self, shape: Shape = (1,)) -> None:
        # default shape: (1,)
        super().__init__(shape=shape)
        # constraints typically couple to some other equation
        self.is_coupled = True

    def mass_matrix(self) -> float:
        # constraint usually couples to no time-derivatives
        return 0

    def plot(self, ax) -> None:
        # nothing to plot
        pass


class VolumeConstraint(ConstraintEquation):
    """
    A volume constraint (or mass constraint) assures the conservation of
    the integral of the unknowns of some given equation when solving the system.
    We may even prescribe the target volume (or mass) with a parameter,
    but we don't have to.
    The constraint equation comes with an additional (unknown) Lagrange
    multiplier that can be interpreted as an influx into the system.
    """

    def __init__(self,
                 reference_equation: Equation,
                 variable: Optional[int] = None) -> None:
        super().__init__(shape=(1,))
        #: on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        #: on which variable (index) of the equation should the constraint be imposed?
        self.variable = variable
        #: this equation brings a single extra degree of freedom (influx Lagrange multiplier)
        self.u = np.zeros(1)
        #: This parameter allows for prescribing a fixed volume (unless it is None)
        self.fixed_volume = None

    def rhs(self, u: Array) -> Array:
        assert self.group is not None
        # generate empty vector of residual contributions
        res = np.zeros((u.size))
        # reference to the indices of the unknowns that we work on
        self_idx = self.group.idx[self]
        eq_idx = self.group.idx[self.ref_eq]
        # optionally split only the part that is referenced by self.variable
        if self.variable is not None:
            eq_shape = self.ref_eq.shape[1:]
            var_ndofs = np.prod(eq_shape)
            start = eq_idx.start + self.variable * var_ndofs
            eq_idx = slice(start, start + var_ndofs)
        # employ the constraint equation
        if self.fixed_volume is None:
            # calculate the difference in volumes between current
            # and previous unknowns of the reference equation
            res[self_idx] = np.mean(u[eq_idx] - self.group.u[eq_idx])
        else:
            # parametric constraint: calculate the difference between current
            # volume and the prescribed fixed_volume parameter
            x = [np.arange(self.ref_eq.shape[-1])]
            if hasattr(self.ref_eq, "x") and getattr(self.ref_eq, "x") is not None:
                x = getattr(self.ref_eq, "x")
            res[self_idx] = np.trapz(u[eq_idx], x) - self.fixed_volume
        # Add the constraint to the reference equation: unknown influx is the Langrange multiplier
        res[eq_idx] = u[self_idx]
        return res

    def jacobian(self, u: Array) -> sp.csr_matrix:
        # TODO: implement analytical / semi-analytical Jacobian
        # convert FD Jacobian to sparse matrix
        return sp.csr_matrix(super().jacobian(u))


class TranslationConstraint(ConstraintEquation):
    """
    A translation constraint assures that the center of mass of some
    reference equation's unknowns does not move when solving the system.
    The additional constraint equations (one per spatial dimension) come
    with Lagrange multipliers, that correspond to the velocities of a comoving
    frame (advection term).
    """

    def __init__(self,
                 reference_equation: 'PartialDifferentialEquation',
                 variable: Optional[int] = None,
                 direction: int = 0) -> None:
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
        assert self.group is not None
        # set up the vector of the residual contributions
        res = np.zeros((u.size))
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
        # add constraint to residuals of reference equation (velocity is the lagrange multiplier)
        try:  # if method du_dx is implemented, use this
            eq_dudx = eq.du_dx(
                eq_u.reshape(eq_shape), self.direction).ravel()
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
        # contributions:
        # - d constraint eq. / du
        # - d bulk eq. / d u
        # - d bulk eq. / d lagrange mul
        # TODO: implement analytical / semi-analytical Jacobian
        # convert FD Jacobian to sparse matrix
        return sp.csr_matrix(super().jacobian(u))
