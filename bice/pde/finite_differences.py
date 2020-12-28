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
        # List of differential matrices: ddx[order] for d^order / dx^order operator
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
        # the boundary conditions, defaults to periodic BC
        self.bc = PeriodicBC()

    # build finite difference differentiation matrices using Fornberg (1988) algorithm
    def build_FD_matrices(self, boundary_conditions=None, approx_order=None, max_order=2, premultiply_bc=True):
        # number of grid points
        N = self.shape[-1]
        # spatial increment
        # TODO: support for higher dimensions than 1d
        dx = self.x[0][1] - self.x[0][0]
        # accuracy / approximation order of the FD scheme (size of stencil = 2*ao + 1)
        if approx_order is not None:
            self.approx_order = approx_order
        ao = self.approx_order
        # stencil grid points scheme
        # TODO: support for non-uniform grids
        stencil_x = np.arange(-ao, ao+1) * dx
        # list of differentiation operators up to desired order d^n / dx^n
        # NOTE: ddx matrices are of shape N x (N+2ao), because they work on boundary padded vectors
        # trivial 0th order d^0 / dx^0 = 1
        zeroth_order = sp.eye(N, N+2*ao, k=ao)
        self.ddx = [zeroth_order]
        # higher order operators
        for order in range(1, max_order+1):
            # get weights for stencil from Fornberg algorithm
            stencil_weights = fornberg.fd_weights(x=stencil_x, x0=0, n=order)
            # put weights on matrix diagonals
            op = sum(w*sp.eye(N, N+2*ao, k=k)
                     for k, w in enumerate(stencil_weights))
            self.ddx.append(op)

        # if given, update boundary conditions
        if boundary_conditions is not None:
            self.bc = boundary_conditions
        # build boundary condition matrices:
        # affine operator (Q*u + G) maps u to boundary padded vector
        self.bc.update(N, dx, approx_order=ao)

        # premultiply operators with boundary matrices: ddx --> ddx * Q
        # NOTE: this neglects the constant (G) of the affine transformation Q*u + G
        if premultiply_bc:
            for order in range(0, max_order+1):
                self.ddx[order] = self.ddx[order].dot(self.bc.Q)
            # check if G is not 0 (inhomogeneous BC)
            if np.count_nonzero(self.bc.G) > 0:
                # if yes, premultiplying the BC is a bad idea, because G is discarded, print warning
                print("WARNING: inhomogeneous boundary conditions should not be pre-multiplied with"
                      " the differentiation operators, because this discards the inhomogeneity G, "
                      "i.e., we end up with homogeneous boundary conditions.\n"
                      "Use build_FD_matrices(premultiply_bc=False) to prevent this. However, you "
                      "then have to do the transformation from u to 'boundary padded u' yourself "
                      "before using the differentiation operators, e.g.:\n"
                      "  du_dx = nabla.dot(bc.Q.dot(u)+bc.G)\n"
                      "or:\n"
                      "  du_dx = nabla.dot(bc.pad(u))\n"
                      "where bc is the boundary conditions object.\n")

        # special names for some operators:
        # nabla operator: d^1/dx^1
        self.nabla = self.ddx[1]
        # laplace operator: d^2/dx^2
        self.laplace = self.ddx[2]


class FDBoundaryConditions:
    # TODO: update description
    # maps u to boundary padded u: u --> (gp, gp, u_0, u_1, ...., u_{N-1}, gp, gp)
    # where gp are the ghost points
    """
    Boundary conditions for FD are applied using an affine transformation Q*u + G that maps the
    unknowns u to the 'boundary padded u', i.e., the u-vector padded with ghost points that assure
    the desired boundary conditions.
    u_pad = (ghost_pt, ghost_pt, u_0, u_1, ..., u_{N-1}, ghost_pt, ghost_pt)
    The differentiation operators then work with the padded unknowns:
    Suppose we have a central FD scheme of approximation order ao and the unknowns u are discretized
    to N grid points. Then, the FD matrix D_x is a (N x (N+2*ao))-matrix, that works on the padded
    unknowns.
    du_dx = D_x * u_pad = D_x * (Q * u + G)
    The matrix Q and the constant G are generated by calling the update(N, dx, ...) method of the
    boundary conditions. This will called automatically during build_FD_matrices of a
    FiniteDifferenceEquation and needs to be called again only if the boundary conditions change.

    For both periodic and homogeneous boundary conditions the transformation is linear (G=0), i.e.,
    we can pre-multiply the operators with Q:
    du_dx = D_x * Q * u = D_x' * u
    This may give a more convenient (and performant) behaviour and is the default in
    FiniteDifferenceEquations. If G is nonzero, pre-multiplication will prompt a warning.
    NOTE: the boundary conditions currently only support 1d grids!
    """

    def __init__(self):
        # linear part: ((N+2*ao) x N)-matrix)
        # (needs to be generated with update(...) before using)
        self.Q = None
        # constant (affine) part
        self.G = 0

    # build the matrix and constant part for the affine transformation u_padded = Q*u + G
    def update(self, N, dx, approx_order):
        # default case: identity matrix (equals homogeneous Dirichlet conditions)
        ao = approx_order
        self.Q = sp.eye(N+2*ao, N, k=-ao)
        self.G = 0

    # transform a vector u to the boundary padded vector
    def pad(self, u):
        return self.Q.dot(u) + self.G


class PeriodicBC(FDBoundaryConditions):
    """
    periodic boundary conditions
    """

    # build the matrix and constant part for the affine transformation u_padded = Q*u + G
    def update(self, N, dx, approx_order):
        # generate matrix that maps u_i --> u_{i%N} for 1d periodic ghost points
        top = sp.eye(approx_order, N, k=N-approx_order)
        bot = sp.eye(approx_order, N, k=0)
        self.Q = sp.vstack((top, sp.eye(N), bot))
        # constant part is zero
        self.G = 0


class RobinBC(FDBoundaryConditions):
    """
    Robin boundary conditions: a*u(x_b) + b*u'(x_b) = c at the boundaries x_b
    a, b, c are tuples with values for (left, right) boundaries.
    """

    def __init__(self, a=(0, 0), b=(1, 1), c=(0, 0)):
        super().__init__()
        # store coefficients
        self.a = a
        self.b = b
        self.c = c

    # build the matrix and constant part for the affine transformation u_padded = Q*u + G
    # cf. RobinBC in https://github.com/SciML/DiffEqOperators.jl
    def update(self, N, dx, approx_order):
        ao = approx_order
        # expand coefficients for left and right
        al, ar = self.a
        bl, br = self.b
        cl, cr = self.c
        # obtain FD stencil from Fornberg (1988) algorithm
        s = fornberg.fd_weights(x=np.arange(1, ao+2), x0=1, n=1)
        # linear part (Q)
        top = np.zeros((ao, N))
        top[-1, :ao] = -s[1:] / (al*dx/bl + s[0]) if bl != 0 else 0*s[1:]
        bot = np.zeros((ao, N))
        bot[0, -ao:] = s[:0:-1] / (ar*dx/br - s[0]) if br != 0 else 0*s[1:]
        self.Q = sp.vstack((top, sp.eye(N), bot))
        # constant part (G)
        self.G = np.zeros(N+2*ao)
        self.G[ao-1] = cl/(al+bl*s[0]/dx) if cl != 0 else 0
        self.G[-ao] = cr/(ar-br*s[0]/dx) if cr != 0 else 0
        # return the matrices
        return self.Q, self.G


def DirichletBC(vals=(0, 0)):
    """
    Dirichlet boundary conditions: u(left) = vals[0], u(right) = vals[1]
    """
    return RobinBC(a=(1, 1), b=(0, 0), c=vals)


def NeumannBC(vals=(0, 0)):
    """
    Neumann boundary conditions: u'(left) = vals[0], u'(right) = vals[1]
    """
    return RobinBC(a=(0, 0), b=(1, 1), c=vals)
