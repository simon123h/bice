import numpy as np
import scipy.sparse as sp
import numdifftools.fornberg as fornberg
from .pde import PartialDifferentialEquation


class FiniteDifferencesEquation(PartialDifferentialEquation):
    """
    The FiniteDifferencesEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs with a finite difference scheme.
    Uses finite difference matrixes from the python package 'findiff'.
    """

    def __init__(self, shape=None):
        super().__init__(shape)
        # List of differential matrices: ddx[order][i] for d^order / dx_i^order operator
        self.ddx = []
        # first order derivative
        self.nabla = None
        # second order derivative
        self.laplace = None
        # approximation order of the finite differences
        self.approx_order = 2
        # the spatial coordinates
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1], endpoint=False)]
        else:
            self.x = None
        # the boundary conditions, defaults to cheap homogeneous Dirichlet
        self.bc = None

    # build finite difference differentiation matrices using Fornberg (1988) algorithm
    def build_FD_matrices(self, approx_order=2):
        # number of grid points
        N = self.shape[-1]
        # spatial increment
        # TODO: support for higher dimensions than 1d
        dx = self.x[0][1] - self.x[0][0]
        # accuracy / order of the FD scheme
        self.approx_order = approx_order
        order = approx_order

        # stencil grid points
        stencil_x = np.arange(-order, order+1) * dx

        # nabla operator: d/dx
        # get weights for stencil from Fornberg algorithm
        stencil_weights = fornberg.fd_weights(x=stencil_x, x0=0, n=1)
        self.nabla = 0
        for k, w in enumerate(stencil_weights):
            self.nabla += w * sp.eye(N, N+2*order, k=k)

        # laplace operator: d^2/dx^2
        # get weights for stencil from Fornberg algorithm
        stencil_weights = fornberg.fd_weights(x=stencil_x, x0=0, n=2)
        self.laplace = 0
        for k, w in enumerate(stencil_weights):
            self.laplace += w * sp.eye(N, N+2*order, k=k)

    # Create Robin boundary conditions for the equation
    def robin_BC(self, a=(0, 0), b=(1, 1), c=(0, 0)):
        # number of grid points
        N = self.shape[-1]
        # spatial increment
        dx = self.x[0][1] - self.x[0][0]
        self.bc = RobinBC(N, dx, a=a, b=b, c=c, approx_order=self.approx_order)

    # Create Dirichlet boundary conditions for the equation
    def dirichlet_BC(self, vals=(0, 0)):
        # number of grid points
        N = self.shape[-1]
        # spatial increment
        dx = self.x[0][1] - self.x[0][0]
        self.bc = RobinBC(N, dx, a=(1, 1), b=(0, 0), c=vals,
                          approx_order=self.approx_order)

    # Create Neumann boundary conditions for the equation
    def neumann_BC(self, vals=(0, 0)):
        # number of grid points
        N = self.shape[-1]
        # spatial increment
        dx = self.x[0][1] - self.x[0][0]
        self.bc = RobinBC(N, dx, b=(1, 1), c=vals,
                          approx_order=self.approx_order)

    # Create Neumann boundary conditions for the equation
    def periodic_BC(self, premultiply_matrices=True):
        # number of grid points
        N = self.shape[-1]
        self.bc = PeriodicBC(N, approx_order=self.approx_order)
        # premultiply differentiation matrices with boundary conditions, only possible if bc.G = 0
        # will break any other boundary conditions until FD matrices are re-generated
        if premultiply_matrices:
            self.nabla = self.nabla.dot(self.bc.Q)
            self.laplace = self.laplace.dot(self.bc.Q)

    def build_periodic_FD_matrices(self):
        # TODO: merge with above method and PeriodicBC
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

        # convert to sparse matrices
        self.nabla = sp.csr_matrix(self.nabla)
        self.laplace = sp.csr_matrix(self.laplace)


class FDBoundaryConditions:
    """
    Boundary conditions for FD are actually an affine transformation and consist of the matrix Q and
    the vector G. Suppose we have a central FD scheme of order o and the unknowns u are discretized
    to N grid points. Then, the FD matrix D_x is a (N x (N+2*o))-matrix, that works on the unknowns
    vector padded with ghost points:
    u_padded = (ghost, ghost, u_0, u_1, ..., u_{N-1}, ghost, ghost)
    We define a linear operator that maps the unknowns to the padded unknowns: u_padded = Q * u
    The differentiation with the FD matrix is then: d_x u = D_x * Q * u.
    For inhomogeneous boundaries, one needs to add a constant part to the differentiation:
    d_x u = D_x * (Q * u + G)
    The boundary matrix is a wrapper class for the matrix Q and vector G
    """
    # NOTE: the boundary conditions currently only support 1d grids!

    def __init__(self, N, approx_order=2):
        # approximation order of the FD scheme
        self.approx_order = approx_order
        # linear part (matrix)
        # maps u to padded boundary u (gp, gp, u0, u1, ...., u(N-1), gp, gp)
        self.Q = sp.eye(N+2*approx_order, N, k=-approx_order)
        # constant (affine) part
        self.G = 0

    # transform a vector u to the boundary padded vector
    def pad(self, u):
        return self.Q.dot(u) + self.G


class RobinBC(FDBoundaryConditions):
    """
    Robin boundary conditions: a*u(x_b) + b*u'(x_b) = c at the boundaries x_b
    a, b, c are tuples with values for (left, right) boundaries.
    """

    def __init__(self, N, dx=1, a=(0, 0), b=(1, 1), c=(0, 0), approx_order=2):
        super().__init__(N, approx_order=approx_order)
        order = approx_order
        # expand coefficients for left and right
        al, ar = a
        bl, br = b
        cl, cr = c
        # obtain FD stencil from Fornberg (1988) algorithm
        self.stencil = fornberg.fd_weights(x=np.arange(1, order+2), x0=1, n=1)
        # generate linear part Q and constant part G
        # cf. RobinBC in https://github.com/SciML/DiffEqOperators.jl
        # linear part (Q)
        s = self.stencil
        top = np.zeros((order, N))
        top[-1, :order] = -s[1:] / (al*dx/bl + s[0]) if bl != 0 else 0*s[1:]
        I = sp.eye(N)
        bot = np.zeros((order, N))
        bot[0, -order:] = s[:0:-1] / (ar*dx/br - s[0]) if br != 0 else 0*s[1:]
        self.Q = sp.vstack((top, I, bot))
        # constant part (G)
        self.G = np.zeros(N+2*order)
        self.G[order-1] = cl/(al+bl*s[0]/dx) if cl != 0 else 0
        self.G[-order] = cr/(ar-br*s[0]/dx) if cr != 0 else 0


def DirichletBC(N, dx=1, vals=(0, 0), approx_order=2):
    """
    Dirichlet boundary conditions: u(left) = vals[0], u(right) = vals[1]
    """
    return RobinBC(N, dx, a=(1, 1), b=(0, 0), c=vals, approx_order=approx_order)


def NeumannBC(N, dx=1, vals=(0, 0), approx_order=2):
    """
    Neumann boundary conditions: u'(left) = vals[0], u'(right) = vals[1]
    """
    return RobinBC(N, dx, a=(0, 0), b=(1, 1), c=vals, approx_order=approx_order)


class PeriodicBC(FDBoundaryConditions):
    """
    periodic boundary condition
    """

    def __init__(self, N, approx_order=2):
        super().__init__(N, approx_order=approx_order)
        # generate matrix that maps u_i --> u_{i%N} for 1d periodic ghost points
        top = sp.eye(approx_order, N, k=N-approx_order)
        bot = sp.eye(approx_order, N, k=0)
        self.Q = sp.vstack((top, sp.eye(N), bot))
        # constant part is zero
        self.G = 0
