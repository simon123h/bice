import numpy as np
from bice.core.equation import Equation


class PseudospectralEquation(Equation):
    """
    The PseudospectralEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs with a pseudospectral scheme.
    """

    def __init__(self, shape=(1,)):
        super().__init__(shape)
        # the spatial coordinates
        self.x = [np.linspace(0, 1, self.shape[-1])]
        self.k = None
        self.ksquare = None

    # build the k-vectors for the Fourier space
    # set real=True, if real input to the FFT can be assumed (rfft)
    # (the k-vectors will be smaller and rfft is more performant than fft)
    def build_kvectors(self, real_fft=False):
        if len(self.x) == 1:
            Lx = self.x[0][-1] - self.x[0][0]
            Nx = self.x[0].size
            # the fourier space
            if real_fft:
                self.k = [np.fft.rfftfreq(Nx, Lx / (2. * Nx * np.pi))]
            else:
                self.k = [np.fft.fftfreq(Nx, Lx / (2. * Nx * np.pi))]
            self.ksquare = self.k[0]**2
        elif len(self.x) == 2:
            Lx = self.x[0][-1] - self.x[0][0]
            Nx = self.x[0].size
            Ly = self.x[1][-1] - self.x[1][0]
            Ny = self.x[1].size
            # the fourier space
            if real_fft:
                kx = np.fft.rfftfreq(Nx, Lx / (2. * Nx * np.pi))
            else:
                kx = np.fft.fftfreq(Nx, Lx / (2. * Nx * np.pi))
            ky = np.fft.fftfreq(Ny, Ly / (2. * Ny * np.pi))
            kx, ky = np.meshgrid(kx, ky)
            self.k = [kx, ky]
            self.ksquare = kx**2 + ky**2
        elif len(self.x) == 3:
            Lx = self.x[0][-1] - self.x[0][0]
            Nx = self.x[0].size
            Ly = self.x[1][-1] - self.x[1][0]
            Ny = self.x[1].size
            Lz = self.x[2][-1] - self.x[2][0]
            Nz = self.x[2].size
            # the fourier space
            if real_fft:
                kx = np.fft.rfftfreq(Nx, Lx / (2. * Nx * np.pi))
            else:
                kx = np.fft.fftfreq(Nx, Lx / (2. * Nx * np.pi))
            ky = np.fft.fftfreq(Ny, Ly / (2. * Ny * np.pi))
            kz = np.fft.fftfreq(Nz, Lz / (2. * Nz * np.pi))
            kx, ky, kz = np.meshgrid(kx, ky, kz)
            self.k = [kx, ky, kz]
            self.ksquare = kx**2 + ky**2 + kz**2
