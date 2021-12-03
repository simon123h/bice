import numpy as np
from bice.core.equation import Equation


class PartialDifferentialEquation(Equation):
    """
    The PseudospectralEquation is a subclass of the general Equation
    and serves as an abstract parent class to more specific classes that offer spatially discretized
    implementations for partial differential equations, e.g. the PseudospectralEquation or the
    FiniteDifferencesEquation.
    """

    def __init__(self, shape=None):
        super().__init__(shape)
        # the spatial coordinates
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1])]
        else:
            self.x = None

    @property
    def spatial_dimension(self):
        """Returns the spatial dimension of the domain self.x"""
        if isinstance(self.x, np.ndarray):
            return self.x.ndim
        return len(self.x)

    def du_dt(self, u=None):
        """calculate the time derivative du/dt of the unknowns"""
        # if u is not given, use self.u
        if u is None:
            u = self.u
        # typically, the mass matrix determines which part of rhs(u) go into du/dt
        return self.mass_matrix().dot(self.rhs(u))

    def du_dx(self, u=None, direction=0):
        """"
        Calculate the spatial derivative du/dx in a given spatial direction.
        (abstract, needs to be specified for child classes)
        """
        raise NotImplementedError(
            "No spatial derivative (du_dx) implemented for this equation!")
