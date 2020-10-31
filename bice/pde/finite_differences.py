import numpy as np
from bice.core.equation import Equation


class FiniteDifferenceEquation(Equation):
    """
    The FiniteDifferenceEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs with a finite difference scheme.
    """

    def __init__(self, shape=None):
        super().__init__(shape)
        # first order derivative
        self.nabla = None
        # second order derivative
        self.laplace = None
        # the spatial coordinates
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1], endpoint=False)]
        else:
            self.x = None


    def build_FD_matrices(self):
        N = self.shape[-1]
        # identity matrix
        I = np.eye(N)
        # spatial increment
        dx = self.x[0][1] - self.x[0][0]

        # nabla operator: d/dx
        self.nabla = np.zeros((N, N))
        self.nabla += -3*np.roll(I, -4, axis=1)
        self.nabla += 32*np.roll(I, -3, axis=1)
        self.nabla += -168*np.roll(I, -2, axis=1)
        self.nabla += 672*np.roll(I, -1, axis=1)
        self.nabla -= 672*np.roll(I, 1, axis=1)
        self.nabla -= -168*np.roll(I, 2, axis=1)
        self.nabla -= 32*np.roll(I, 3, axis=1)
        self.nabla -= -3*np.roll(I, 4, axis=1)
        self.nabla /= dx * 840

        # nabla operator: d^2/dx^2
        self.laplace = np.zeros((N, N))
        self.laplace += -9*np.roll(I, -4, axis=1)
        self.laplace += 128*np.roll(I, -3, axis=1)
        self.laplace += -1008*np.roll(I, -2, axis=1)
        self.laplace += 8064*np.roll(I, -1, axis=1)
        self.laplace += -14350*np.roll(I, 0, axis=1)
        self.laplace += 8064*np.roll(I, 1, axis=1)
        self.laplace += -1008*np.roll(I, 2, axis=1)
        self.laplace += 128*np.roll(I, 3, axis=1)
        self.laplace += -9*np.roll(I, 4, axis=1)
        self.laplace /= dx**2 * 5040
