import numpy as np
import findiff
import scipy.sparse as sp
from scipy.interpolate import interp1d
import numdifftools.fornberg as fornberg
from .pde import PartialDifferentialEquation
from bice.core import profile


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
        # mesh adaption settings
        self.max_refinement_error = 1e-1
        self.min_refinement_error = 1e-3
        self.min_dx = 1e-3
        self.max_dx = 2

    # build finite difference differentiation matrices using Fornberg (1988) algorithm
    @profile
    def build_FD_matrices(self, boundary_conditions=None, approx_order=None, max_order=2, premultiply_bc=True):
        # number of grid points
        N = self.shape[-1]
        # accuracy / approximation order of the FD scheme (size of stencil = 2*ao + 1)
        if approx_order is not None:
            self.approx_order = approx_order
        ao = self.approx_order
        # pad x vector with x-values of ghost nodes
        # TODO: support for higher dimensions than 1d
        x = self.x[0]
        x_pad = self.bc.pad_x(x)
        # list of differentiation operators up to desired order d^n / dx^n
        # NOTE: ddx matrices are of shape N x (N+2), because they work on boundary padded vectors
        # trivial 0th order d^0 / dx^0 = 1
        zeroth_order = sp.eye(N, N+2, k=1)
        self.ddx = [zeroth_order]
        # higher order operators
        for order in range(1, max_order+1):
            # obtain differentiation matrix from findiff package
            op = findiff.FinDiff(0, x_pad, order, acc=ao).matrix(x_pad.shape)
            # slice rows corresponding to ghost nodes
            op = op[1:-1, :]
            self.ddx.append(op)

        # if given, update boundary conditions
        if boundary_conditions is not None:
            self.bc = boundary_conditions
        # build boundary condition matrices:
        # affine operator (Q*u + G) maps u to boundary padded vector
        self.bc.update(x, approx_order=ao)

        # premultiply operators with boundary matrices: ddx --> ddx * Q
        # NOTE: this neglects the constant (G) of the affine transformation Q*u + G
        if premultiply_bc:
            self.bc.premultiplied = True
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
        else:
            self.bc.premultiplied = False

        # special names for some operators:
        # nabla operator: d^1/dx^1
        self.nabla = self.ddx[1]
        # laplace operator: d^2/dx^2
        self.laplace = self.ddx[2]

    # Jacobian of the equation
    def jacobian(self, u):
        # FD Jacobians are typically sparse, so we convert to a sparse matrix
        return sp.csr_matrix(super().jacobian(u))
    

    # perform adaption of the grid to the solution
    # TODO: support higher dimensions than 1d
    @profile
    def adapt(self):
        # calculate error estimate
        error_estimate = self.refinement_error_estimate()
        # adapt the mesh
        x_old = self.x[0]
        x_new = []
        i = 0
        while i < len(x_old):
            x = x_old[i]
            # exclude boundaries
            if not (0 < i < len(x_old) - 1):
                x_new.append(x)
                i += 1
                continue
            # unrefinement
            err = error_estimate[i]
            dx = x_old[i+1]-x_old[i-1]
            if err < self.min_refinement_error and dx < self.max_dx:
                x_new.append(x_old[i+1])
                i += 2
                continue
            # refinement
            dx = (x - x_old[i-1])/2
            if err > self.max_refinement_error and dx > self.min_dx:
                x_new.append((x + x_old[i-1])/2)
            x_new.append(x)
            i += 1
        x_new = np.array(x_new)
        # interpolate unknowns to new grid points
        nvars = self.shape[0] if len(self.shape) > 1 else 1
        if nvars > 1:
            u_new = np.array([np.interp(x_new, x_old, self.u[n])
                              for n in range(nvars)])
        else:
            u_new = np.interp(x_new, x_old, self.u)
        # update shape, u and x
        self.reshape(u_new.shape)
        self.u = u_new
        self.x = [x_new]
        # interpolate history to new grid points
        for t, u in enumerate(self.u_history):
            if nvars > 1:
                self.u_history[t] = np.array([np.interp(x_new, x_old, u[n])
                                              for n in range(nvars)])
            else:
                self.u_history[t] = np.interp(x_new, x_old, u)
        # re-build the FEM matrices
        self.build_FD_matrices(premultiply_bc=self.bc.premultiplied)

    # estimate the error made in each grid point
    @profile
    def refinement_error_estimate(self):
        # calculate integral of curvature:
        # error = | \int d^2 u / dx^2 * test(x) dx |
        # NOTE: overwrite this method, if different weights of the curvatures are needed
        err = 0
        dx = np.diff(self.x[0])
        dx = [max(dx[i], dx[i+1]) for i in range(len(dx)-1)]
        dx = np.concatenate(([0], dx, [0]))
        nvars = self.shape[0] if len(self.shape) > 1 else 1
        for n in range(nvars):
            u = self.u[n] if len(self.shape) > 1 else self.u
            curv = self.laplace.dot(self.bc.pad(u))
            err += np.abs(curv*dx)
        return err


class FDBoundaryConditions:
    # TODO: update description
    # maps u to boundary padded u: u --> (gp, gp, u_0, u_1, ...., u_{N-1}, gp, gp)
    # where gp are the ghost points
    """
    Boundary conditions for FD are applied using an affine transformation Q*u + G that maps the
    unknowns u to the 'boundary padded u', i.e., the u-vector padded with ghost points that assure
    the desired boundary conditions.
    u_pad = (ghost_pt_l, u_0, u_1, ..., u_{N-1}, ghost_pt_r)
    The differentiation operators then work with the padded unknowns:
    Suppose we have a central FD scheme of approximation order ao and the unknowns u are discretized
    to N grid points. Then, the FD matrix D_x is a (N x (N+2))-matrix, that works on the padded
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
        # linear part: ((N+2) x N)-matrix)
        # (needs to be generated with update(...) before using)
        self.Q = None
        # constant (affine) part
        self.G = 0
        # are the boundary conditions Q already multiplied with the differentiation operators?
        self.premultiplied = False

    # build the matrix and constant part for the affine transformation u_padded = Q*u + G
    def update(self, x, approx_order):
        # default case: identity matrix (equals homogeneous Dirichlet conditions)
        N = len(x)
        ao = approx_order
        self.Q = sp.eye(N+2, N, k=-ao)
        self.G = 0

    # transform a vector u to the boundary padded vector
    def pad(self, u):
        # if Q is already multiplied to operators, mapping is not required
        if self.premultiplied:
            return u
        # else, do the mapping
        return self.Q.dot(u) + self.G

    # pad vector of node values with the x-values of the ghost nodes
    def pad_x(self, x):
        dxl = x[1] - x[0]
        dxr = x[-1] - x[-2]
        return np.concatenate(([x[0]-dxl], x, [x[-1]+dxr]))


class PeriodicBC(FDBoundaryConditions):
    """
    periodic boundary conditions
    """

    # build the matrix and constant part for the affine transformation u_padded = Q*u + G
    def update(self, x, approx_order):
        # generate matrix that maps u_i --> u_{i%N} for 1d periodic ghost points
        N = len(x)
        top = sp.eye(1, N, k=N-1)
        bot = sp.eye(1, N, k=0)
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
    @profile
    def update(self, x, approx_order):
        N = len(x)
        ao = approx_order
        # expand coefficients for left and right
        al, ar = self.a
        bl, br = self.b
        cl, cr = self.c
        # pad x with the ghost point values
        x = self.pad_x(x)
        # obtain FD stencils from Fornberg (1988) algorithm
        sl = fornberg.fd_weights(x=x[:ao+1], x0=x[0], n=1)
        sr = fornberg.fd_weights(x=x[-ao-1:], x0=x[-1], n=1)
        # linear part (Q)
        top = np.zeros((1, N))
        top[-1, :ao] = -sl[1:] / (al/bl + sl[0]) if bl != 0 else 0*sl[1:]
        bot = np.zeros((1, N))
        bot[0, -ao:] = -sr[:-1] / (ar/br + sr[-1]) if br != 0 else 0*sr[1:]
        self.Q = sp.vstack((top, sp.eye(N), bot))
        # constant part (G)
        self.G = np.zeros(N+2)
        self.G[0] = cl/(al+bl*sl[0]) if cl != 0 else 0
        self.G[-1] = cr/(ar+br*sr[-1]) if cr != 0 else 0
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
