from .problem import Problem
import numpy as np


# Pseudospectral implementation of the 1-dimensional Advection Equation
class Advection(Problem):

    def __init__(self, N, L):
        super().__init__()
        # velocity parameter
        self.u = 1
        # space and fourier space
        self.x = np.linspace(0, L, N)
        self.k = np.fft.rfftfreq(N, L / (2. * N * np.pi))
        # initialize unknowns
        self.u = np.cos(self.x)

    def rhs(self, u):
        u_k = np.fft.rfft(u)
        return np.fft.irfft(1j * u * self.k * u_k)
