import numpy as np
from .pde import PartialDifferentialEquation


class CollocationEquation(PartialDifferentialEquation):
    """
    The CollocationEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs using the collocation method, where the solution u(x) is treated in a polynomial base:
    u(x) = \sum_{n=0}^N U_n x^n. In self.u, we store the coefficients U_n.
    see: https://math.stackexchange.com/questions/3195545/using-collocation-method-to-solve-a-nonlinear-boundary-value-ode
    """

    def __init__(self, shape=None):
        super().__init__(shape)
        # list of derivatives matrices up to desired order
        self.ddx = []
        # the spatial coordinates
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1], endpoint=False)]
        else:
            self.x = None

    def build_ddx_matrices(self, max_order=4):
        N = self.shape[-1]
        # build nabla operator: d/dx
        nabla = np.zeros((N, N))
        for n in range(N-1):
            # coefficient of order n is prior order n+1 coefficient multiplied by exponent n+1
            nabla[n, n+1] = n+1

        # build derivative operators of arbitrary order i by calculating nabla^i
        self.ddx = [nabla]
        for i in range(1, max_order):
            # build derivative operator d^i / dx^i = d^(i-1) / dx^(i-1) * nabla = nabla^i
            self.ddx.append(self.ddx[i-1].dot(nabla))

    # first order derivative operator
    @property
    def nabla(self):
        return self.ddx[0]

    # second order derivative operator
    @property
    def laplace(self):
        return self.ddx[1]

    # calculate the spatial derivative du/dx in a given spatial direction
    def du_dx(self, u=None, direction=0):
        # if u is not given, use self.u
        if u is None:
            u = self.u
        # multiply with nabla matrix
        return self.ddx[0].dot(u)

    # convert from spatial basis u(x) to polynomial basis U_n
    def u2poly(self, u=None):
        if u is None:
            u = self.u
        N = self.shape[-1]
        # convert using least squares fit to polynomial of order N
        return np.polyfit(self.x[0], u, N)[::-1]

    # convert from polynomial basis U_n to spatial basis u(x)
    def poly2u(self, u=None):
        if u is None:
            u = self.u
        # evaluate polynomial with given coefficients u at positions self.x
        return np.polyval(u[::-1], self.x[0])
