"""Base classes for partial differential equations (PDEs)."""

import numpy as np

from bice.core.equation import Equation
from bice.core.types import Array, Shape


class PartialDifferentialEquation(Equation):
    """
    Abstract base class for spatially discretized equations.

    Serves as a parent class for more specific implementations such as
    pseudospectral or finite difference schemes.
    """

    def __init__(self, shape: Shape | None = None) -> None:
        """
        Initialize the PDE.

        Parameters
        ----------
        shape
            The shape of the unknowns.
        """
        super().__init__(shape)
        #: the spatial coordinates
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1])]
        else:
            self.x = None

    @property
    def spatial_dimension(self) -> int:
        """
        Return the spatial dimension of the domain.

        Returns
        -------
        int
            The spatial dimension.
        """
        assert self.x is not None
        if isinstance(self.x, np.ndarray):
            return self.x.ndim
        return len(self.x)

    def du_dt(self, u: Array | None = None) -> Array:
        """
        Calculate the time derivative du/dt of the unknowns.

        Parameters
        ----------
        u
            The vector of unknowns. If None, the current state `self.u` is used.

        Returns
        -------
        Array
            The time derivative vector.
        """
        # if u is not given, use self.u
        if u is None:
            u = self.u
        # typically, the mass matrix determines which part of rhs(u) go into du/dt
        return self.mass_matrix().dot(self.rhs(u))

    def du_dx(self, u: Array | None = None, direction: int = 0) -> Array:
        """
        Calculate the spatial derivative du/dx in a given direction.

        Parameters
        ----------
        u
            The vector of unknowns.
        direction
            The index of the spatial direction (e.g., 0 for x, 1 for y).

        Returns
        -------
        Array
            The spatial derivative vector.

        Raises
        ------
        NotImplementedError
            This is an abstract method.
        """
        raise NotImplementedError("No spatial derivative (du_dx) implemented for this equation!")
