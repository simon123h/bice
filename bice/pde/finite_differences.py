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
        # the spatial coordinates
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1], endpoint=False)]
        else:
            self.x = None
        # the boundary conditions, if None, defaults to periodic BCs
        self.bc = None
        # mesh adaption settings
        self.max_refinement_error = 1e-0
        self.min_refinement_error = 1e-2
        self.min_dx = 1e-3
        self.max_dx = 2

    # build finite difference differentiation matrices using Fornberg (1988) algorithm
    @profile
    def build_FD_matrices(self, approx_order=2, max_order=2):
        # TODO: support for higher dimensions than 1d
        # accuracy / approximation order of the FD scheme (size of stencil = 2*ao + 1)
        ao = approx_order
        # the spatial grid
        x = self.x[0]
        N = len(x)
        # check if boundary conditions are set, otherwise set default
        if self.bc is None:
            self.bc = PeriodicBC()
            print("WARNING: No boundary conditions set for:", self, "\n"
                  "Defaulting to periodic boundaries. To change this "
                  "(and to supress this warning), use, e.g.:\n"
                  "  self.bc = DirichletBC()\n"
                  "or any other boundary conditions type from pde.finite_differences, "
                  "before building the FD matrices in the Equation object.")
        # build boundary condition operators given the grid points and approximation order
        self.bc.update(x, approx_order=ao)
        # boundary conditions implement an affine operator (Q*u + G) with a matrix Q and
        # a constant G, that maps the unknowns u to a 'boundary padded vector' with ghost points
        # at the boundaries.
        # pad x vector with x-values of ghost nodes
        x_pad = self.bc.pad_x(x)
        # the number of ghost points
        Ngp = len(x_pad) - N

        # list of differentiation operators up to desired order d^n / dx^n
        # NOTE: ddx matrices are of shape N x (N+Ngp), because they work on boundary padded vectors
        self.ddx = []
        # higher order operators
        for order in range(0, max_order+1):
            if order == 0:
                # trivial 0th order d^0 / dx^0 = 1
                op = sp.eye(N, N+Ngp, k=Ngp//2)
            else:
                # obtain differentiation matrix from findiff package
                op = findiff.FinDiff(
                    0, x_pad, order, acc=2*ao).matrix(x_pad.shape)
                # slice rows corresponding to ghost nodes
                op = op[Ngp//2:N+Ngp//2, :]
            # include the boundary conditions into the differentiation operator by pre-multiplying
            # it with the (affine) boundary operator (matrix Q and constant G)
            opQ = op.dot(self.bc.Q)
            opG = op.dot(np.zeros(N+Ngp)+self.bc.G)
            # store as new affine operator Op(u) = ddx*(Q*u + G) = opQ*u + opG
            self.ddx.append(AffineOperator(opQ, opG))

        # special names for some operators:
        # nabla operator: d / dx
        self.nabla = self.ddx[1]
        # laplace operator: d^2 / dx^2
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
        self.build_FD_matrices()

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
            curv = self.laplace(u)
            err += np.abs(curv*dx)
        return err


# wrapper object for an affine operator:
# Op: u --> Q*u + G, where Q is a matrix and G is some constant
class AffineOperator:

    def __init__(self, Q, G=0):
        # linear part
        self.Q = Q
        # constant (affine) part
        self.G = G

    # Apply the operator to some vector/tensor u and scale the constant part with g:
    # operator(u) = Q*u + g*G
    # if called without arguments, only the linear part is returned, for an intuitive
    # implementation of operator derivatives, e.g.:
    # f(u) = operator(u) = Q*u + g*G
    # f'(u) = operator() = Q
    # this also allows for simple operator algebra, e.g.:
    # op2 = operator()*operator() = Q*Q (unfortunately discarding the constant part)
    def __call__(self, u=None, g=1):
        if u is None:
            return self.Q
        return self.Q.dot(u) + g*self.G

    # overload the dot method, so we can do operator.dot(u) as with numpy/scipy matrices
    def dot(self, u):
        return self.__call__(u)


class FDBoundaryConditions:
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
    FiniteDifferencesEquation and needs to be called again only if the boundary conditions change.

    TODO: update this part, since we have AffineOperators now!
    For both periodic and homogeneous boundary conditions the transformation is linear (G=0), i.e.,
    we can pre-multiply the operators with Q:
    du_dx = D_x * Q * u = D_x' * u
    This may give a more convenient (and performant) behaviour and is the default in
    FiniteDifferencesEquations. If G is nonzero, pre-multiplication will prompt a warning.
    NOTE: the boundary conditions currently only support 1d grids!
    """

    def __init__(self):
        # linear part: ((N+2) x N)-matrix)
        # (needs to be generated with update(...) before using)
        self.Q = None
        # constant (affine) part
        self.G = 0

    # build the matrix and constant part for the affine transformation u_padded = Q*u + G
    def update(self, x, approx_order):
        # default case: identity matrix (equals homogeneous Dirichlet conditions)
        N = len(x)
        ao = approx_order
        self.Q = sp.eye(N+2, N, k=-ao)
        self.G = 0

    # transform a vector u to the boundary padded vector: u_pad = Q*u + G
    def pad(self, u):
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
    # TODO: PeriodicBCs could support 2*approx_order ghost points!

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


class GenericBC(FDBoundaryConditions):
    # TODO: rename?
    # TODO: equal to hom. Dirichlet or Neumann conditions??
    """
    These boundaries have no ghost points at all!
    Therefore, the derivatives on the boundaries are determined by
    forward / backward differences of suitable order.
    The resulting differentiation matrices are very helpful for performing
    derivatives fields that should do not imply any boundary conditons, e.g.:
    Delta u = nabla_bc * (nabla_free * u),
    where the inner nabla_free should not conflict with the boundary conditions
    imposed by the outer nabla_bc. Hence, nabla_free can be built using
    GenericBC.
    """

    # build the matrix and constant part for the affine transformation u_padded = Q*u + G
    def update(self, x, approx_order):
        # default case: identity matrix (equals homogeneous Dirichlet conditions)
        # no ghost points, mapping is identity: boundary padded u = u = 1 * u + 0
        N = len(x)
        self.Q = sp.eye(N)
        self.G = 0

    # pad vector of node values with the x-values of the ghost nodes
    def pad_x(self, x):
        # here: no ghost nodes! x_pad = x
        return x
