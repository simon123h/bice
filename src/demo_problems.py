from .problem import Problem
import numpy as np

# The Lotka-Volterra equations (predator prey model)
class LotkaVolterra(Problem):

    def __init__(self):
        super().__init__()
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1
        self.u = np.array([1., 0.7])

    def rhs(self, u):
        x = u[0]
        y = u[1]
        return np.array([
            (self.a - self.b * y) * x,
            (self.d * x - self.c) * y
        ])


# Pseudospectral implementation of the 1-dimensional Swift-Hohenberg Equation
# equation, a nonlinear PDE
# \partial t h &= (r - (kc^2 + \Delta)^2)h + v * h^2 - g * h^3
class SwiftHohenberg(Problem):

    def __init__(self, N, L):
        super().__init__()
        # parameters
        self.r = -0.013
        self.kc = 0.5
        self.v = 0.41
        self.g = 1
        # space and fourier space
        self.x = np.linspace(0, L, N)
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initialize unknowns
        self.u = np.cos(self.x) + np.random.rand(N) * 0.02

    def rhs(self, u):
        u_k = np.fft.rfft(u)
        return np.fft.irfft((self.r - (self.kc**2 - self.k**2)**2) * u_k) + self.v * u**2 - self.g * u**3
