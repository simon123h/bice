from __future__ import annotations

from typing import Optional

import findiff
import numdifftools.fornberg as fornberg
import numpy as np
import scipy.sparse as sp

from bice.core import profile
from bice.core.types import Array, Matrix, Shape

from .pde import PartialDifferentialEquation


class FiniteDifferencesEquation(PartialDifferentialEquation):
    """
    The FiniteDifferencesEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs with a finite difference scheme.
    Uses finite difference matrixes from the python package 'findiff'.
    """

    def __init__(self, shape: Optional[Shape] = None) -> None:
        super().__init__(shape)
        #: List of differential matrices: ddx[order] for d^order / dx^order operator
        self.ddx = []
        #: first order derivative
        self.nabla = None
        #: second order derivative
        self.laplace = None
        #: the spatial coordinates
        self.x = None
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1], endpoint=False)]
        #: the boundary conditions, if None, defaults to periodic BCs
        self.bc = None
        # mesh adaption settings
        #: mesh adaption: maximum error tolerance
        self.max_refinement_error = 1e-0
        #: mesh adaption: minimum error tolerance
        self.min_refinement_error = 1e-2
        #: minimum grid size for mesh adaption
        self.min_dx = 1e-3
        #: maximum grid size for mesh adaption
        self.max_dx = 2

    @profile
    def build_FD_matrices(self, approx_order: int = 2):
        """Build finite difference differentiation matrices using 1d FD matrices"""
        assert self.x is not None
        # check for spatial dimension:
        if self.spatial_dimension == 1:
            # 1d case: proceed with x-vector
            x = self.x[0] if isinstance(self.x, list) else self.x
            return self.build_FD_matrices_1d(x=x, approx_order=approx_order)
        # else, higher-than-1d case:
        # construct FD matrices from 1d FD matrices for each spatial dimension
        # TODO: support higher than 2 dimensions
        if self.spatial_dimension > 2:
            raise NotImplementedError("Finite difference operators for spatial dimensions"
                                      "higher than 1d are not yet supported.")
        # 2d case:
        ops1d = []
        for x in self.x:
            # generate 1d operators
            op1d = self.build_FD_matrices_1d(
                x=x, approx_order=approx_order)[:3]
            # check if we have inhomogeneous boundary conditions:
            for op in op1d:
                if not op.is_linear():
                    raise NotImplementedError("Inhomogeneous boundary conditions are unfortunately"
                                              "not supported spatial dimensions higher than 1d.")
            # drop inhomogeneous part of affine operators (is zero for homogeneous BCs)
            ops1d.append([op.Q for op in op1d])
        # 2D FD matrices from 1D matrices using Kronecker product
        Ix, Dx_1d, D2x_1d = ops1d[0]
        Iy, Dy_1d, D2y_1d = ops1d[1]
        Dx_2d = sp.kron(Iy, Dx_1d)
        Dy_2d = sp.kron(Dy_1d, Ix)
        D2x_2d = sp.kron(Iy, D2x_1d)
        D2y_2d = sp.kron(D2y_1d, Ix)
        # store operators in class member variables
        self.ddx = [[Ix, Iy], [Dx_2d, Dy_2d], [D2x_2d, D2y_2d]]
        self.nabla = [Dx_2d, Dy_2d]  # nabla operator
        self.laplace = D2x_2d + D2y_2d  # laplace operator
        return self.ddx

    @profile
    def build_FD_matrices_1d(self, approx_order=2, x=None) -> list['AffineOperator']:
        """Build 1d finite difference differentiation matrices using Fornberg (1988) algorithm"""
        # accuracy / approximation order of the FD scheme (size of stencil = 2*ao + 1)
        ao = 2 * (approx_order // 2)  # has to be an even number
        # maximum derivative order of operators to build
        max_order = 2
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
                    0, x_pad, order, acc=ao).matrix(x_pad.shape)
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
        # return the resulting list of FD matrices
        return self.ddx

    def jacobian(self, u) -> Matrix:
        """Jacobian of the equation"""
        # FD Jacobians are typically sparse, so we convert to a sparse matrix
        return sp.csr_matrix(super().jacobian(u))

    # TODO: support higher dimensions than 1d
    @profile
    def adapt(self) -> None:
        """Perform adaption of the grid to the solution"""
        # mesh adaption is only supported for 1d
        if self.spatial_dimension > 1:
            return
        # calculate error estimate
        error_estimate = self.refinement_error_estimate()
        # adapt the mesh
        assert self.x is not None
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
        # re-build the finite difference matrices
        self.build_FD_matrices()

    @profile
    def refinement_error_estimate(self) -> np.ndarray:
        """Estimate the error made in each grid point"""
        # calculate integral of curvature:
        # error = | \int d^2 u / dx^2 * test(x) dx |
        # NOTE: overwrite this method, if different weights of the curvatures are needed
        err = 0
        assert self.x is not None
        dx = np.diff(self.x[0])
        dx = [max(dx[i], dx[i+1]) for i in range(len(dx)-1)]
        dx = np.concatenate(([0], dx, [0]))
        nvars = self.shape[0] if len(self.shape) > 1 else 1
        for n in range(nvars):
            u = self.u[n] if len(self.shape) > 1 else self.u
            assert self.laplace is not None
            curv = self.laplace(u)
            err += np.abs(curv*dx)
        return err

    def du_dx(self, u: Array, direction: int = 0) -> Array:
        """Default implementation for spatial derivative"""
        assert self.nabla is not None
        if self.spatial_dimension == 1:  # 1d case
            assert isinstance(self.nabla, AffineOperator)
            return self.nabla(u)
        return self.nabla[direction].dot(u)

    def save(self) -> dict:
        """
        Save the state of the equation, including the x-values.
        Override this method, if your equation needs to store more stuff.
        """
        data = super().save()
        data.update({'x': self.x})
        return data

    def load(self, data) -> None:
        """
        Load the state of the equation, including the x-values.
        Override this method, if your equation needs to recover more stuff.
        """
        self.x = data['x']
        super().load(data)


class AffineOperator:
    """
    Wrapper object for an affine operator:
    Op: u --> Q*u + G, where Q is a matrix and G is some constant
    Needed for including boundary conditions into differentiation operators
    """

    def __init__(self, Q, G=0):
        #: linear part
        self.Q = Q
        #: constant (affine) part
        self.G = G

    def __call__(self, u=None, g=1):
        """
        Apply the operator to some vector/tensor u and scale the constant part with g:
        operator(u) = Q*u + g*G
        if called without arguments, only the linear part is returned, for an intuitive
        implementation of operator derivatives, e.g.:
        f(u) = operator(u) = Q*u + g*G
        f'(u) = operator() = Q
        this also allows for simple operator algebra, e.g.:
        op2 = operator()*operator() = Q*Q (unfortunately discarding the constant part)
        """
        # if u is not given, return the matrix Q alone
        if u is None:
            return self.Q
        # check if u is not a vector (but a matrix or tensor)
        # make sure that a sparse matrix is returned
        if u.ndim > 1:
            return self.Q.dot(u) + sp.coo_matrix(g*np.resize(self.G, u.shape))
        # else, u is a vector, simply perform the Q*u + G
        return self.Q.dot(u) + g*self.G

    def dot(self, u):
        """Overloaded dot method, so we can do operator.dot(u) as with numpy/scipy matrices"""
        return self.__call__(u)

    def is_linear(self):
        """
        Is the affine operator a linear operator, i.e., is the constant part G=0?
        """
        return not np.any(self.G)  # checks if all entries of G are zero


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

    When the finite differences operators, e.g. an FD operator 'D', are built by the Equation class,
    the FD operators and the boundary operator are composited:

    D_bc(u) = D(u_pad) = D(Q * u + G) = (D * Q)*u + D * G = Q' * u + G',

    i.e., the differentiation operators become themselves affine operators (with Q' and G') and are
    stored using AffineOperator(Q', G') objects (see code above).
    For both periodic and homogeneous boundary conditions the boundary transformation is
    linear (G=0), i.e., the affine operators are simply classic matrices.
    NOTE: inhomogeneous boundary conditions are currently only supported by 1d grids!
    """

    def __init__(self):
        # linear part: ((N+2) x N)-matrix)
        # (needs to be generated with update(...) before using)
        self.Q = None
        # constant (affine) part
        self.G = 0

    def update(self, x, approx_order):
        """build the matrix and constant part for the affine transformation u_padded = Q*u + G"""
        # default case: identity matrix (equals homogeneous Dirichlet conditions)
        N = len(x)
        ao = approx_order
        self.Q = sp.eye(N+2, N, k=-ao)
        self.G = 0

    def pad(self, u):
        """Transform a vector u to the boundary padded vector: u_pad = Q*u + G"""
        assert self.Q is not None
        return self.Q.dot(u) + self.G

    def pad_x(self, x):
        """Pad vector of node values with the x-values of the ghost nodes"""
        dxl = x[1] - x[0]
        dxr = x[-1] - x[-2]
        return np.concatenate(([x[0]-dxl], x, [x[-1]+dxr]))


class PeriodicBC(FDBoundaryConditions):
    """
    Periodic boundary conditions
    """

    def __init__(self):
        super().__init__()
        #: how many ghost nodes at each boundary?
        self.order = 1
        #: the virtual distance between the left and right boundary node
        self.boundary_dx = None

    def update(self, x, approx_order):
        """Build the matrix and constant part for the affine transformation u_padded = Q*u + G"""
        # generate matrix that maps u_i --> u_{i%N} for 1d periodic ghost points
        self.order = approx_order
        N = len(x)
        top = sp.eye(self.order, N, k=N-self.order)
        bot = sp.eye(self.order, N, k=0)
        self.Q = sp.vstack((top, sp.eye(N), bot))
        # constant part is zero
        self.G = 0

    def pad_x(self, x):
        """Pad vector of node values with the x-values of the ghost nodes"""
        # obtain the (constant!) virtual distance between the left and right boundary nodes
        if self.boundary_dx is None:
            self.boundary_dx = x[1] - x[0]
        dx_lr = np.array([self.boundary_dx])
        # build the full list of dx's for the periodic domain
        dx = np.concatenate((dx_lr, np.diff(x), dx_lr))
        # construct the left and right ghost nodes (number of ghost points = self.order)
        x_l = [x[0]-sum(dx[-n-1:]) for n in range(self.order)][::-1]
        x_r = [x[-1]+sum(dx[:n+1]) for n in range(self.order)]
        # concatenate for full padded x vector
        return np.concatenate((x_l, x, x_r))


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

    @profile
    def update(self, x, approx_order):
        """
        Build the matrix and constant part for the affine transformation u_padded = Q*u + G
        cf. RobinBC in https://github.com/SciML/DiffEqOperators.jl
        """
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


class NoBoundaryConditions(FDBoundaryConditions):
    """
    These boundaries have no ghost points at all!
    Therefore, the derivatives on the boundaries are determined by
    forward / backward differences of suitable order.
    The resulting differentiation matrices are very helpful for performing
    derivatives fields that should do not imply any boundary conditons, e.g.:
    Delta u = nabla_bc * (nabla_free * u),
    where the inner nabla_free should not conflict with the boundary conditions
    imposed by the outer nabla_bc. Hence, nabla_free can be built using
    NoBoundaryConditions.
    """

    def update(self, x, approx_order):
        """Build the matrix and constant part for the affine transformation u_padded = Q*u + G"""
        # default case: identity matrix (equals homogeneous Dirichlet conditions)
        # no ghost points, mapping is identity: boundary padded u = u = 1 * u + 0
        N = len(x)
        self.Q = sp.eye(N)
        self.G = 0

    def pad_x(self, x):
        """Pad vector of node values with the x-values of the ghost nodes"""
        # here: no ghost nodes! x_pad = x
        return x
