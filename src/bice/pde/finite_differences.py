"""Finite difference discretization schemes and boundary conditions."""

from __future__ import annotations

import findiff
import numdifftools.fornberg as fornberg
import numpy as np
import scipy.sparse as sp

from bice.core import profile
from bice.core.types import Array, DataDict, Matrix, RealArray, Shape

from .pde import PartialDifferentialEquation


class FiniteDifferencesEquation(PartialDifferentialEquation):
    """
    Spatially discretized equation using a finite difference scheme.

    Provides routines for building differentiation matrices and managing
    boundary conditions. Uses the 'findiff' package for matrix generation.
    """

    def __init__(self, shape: Shape | None = None) -> None:
        """
        Initialize the FiniteDifferencesEquation.

        Parameters
        ----------
        shape
            The shape of the unknowns.
        """
        super().__init__(shape)
        #: List of differential matrices: ddx[order] for d^order / dx^order operator
        #  Outer list is derivative order, inner list is spatial dimension
        self.ddx: list[list[AffineOperator | Matrix]] = []
        #: first order derivative operator
        self.nabla: AffineOperator | list[Matrix] | None = None
        #: second order derivative operator (Laplacian)
        self.laplace: AffineOperator | Matrix | None = None
        #: the spatial coordinates
        self.x: list[RealArray] | None = None
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1], endpoint=False)]
        #: the boundary conditions, if None, defaults to periodic BCs
        self.bc: FDBoundaryConditions | None = None
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
    def build_FD_matrices(self, approx_order: int = 2) -> list[list[AffineOperator | Matrix]]:
        """
        Build finite difference differentiation matrices.

        Supports 1D and 2D domains. 2D matrices are constructed from 1D
        operators using Kronecker products.

        Parameters
        ----------
        approx_order
            The desired approximation order of the finite difference scheme.

        Returns
        -------
        list
            The list of built differentiation matrices.

        Raises
        ------
        NotImplementedError
            If spatial dimension is higher than 2.
        """
        assert self.x is not None
        # check for spatial dimension:
        if self.spatial_dimension == 1:
            # 1d case: proceed with x-vector
            grid_x = self.x[0] if isinstance(self.x, list) else (self.x if self.x is not None else np.array([]))
            ops_1d = self.build_FD_matrices_1d(x=grid_x, approx_order=approx_order)
            self.ddx = [[op] for op in ops_1d]
            return self.ddx
        # else, higher-than-1d case:
        # construct FD matrices from 1d FD matrices for each spatial dimension
        # TODO: support higher than 2 dimensions
        if self.spatial_dimension > 2:
            raise NotImplementedError("Finite difference operators for spatial dimensions higher than 1d are not yet supported.")
        # 2d case:
        ops1d = []
        for x in self.x:
            # generate 1d operators
            op1d = self.build_FD_matrices_1d(x=x, approx_order=approx_order)[:3]
            # check if we have inhomogeneous boundary conditions:
            for op in op1d:
                if not op.is_linear():
                    raise NotImplementedError(
                        "Inhomogeneous boundary conditions are unfortunately not supported spatial dimensions higher than 1d."
                    )
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
        self.ddx = [[sp.csr_matrix(Ix), sp.csr_matrix(Iy)], [sp.csr_matrix(Dx_2d), sp.csr_matrix(Dy_2d)], [sp.csr_matrix(D2x_2d + D2y_2d)]]
        self.nabla = [sp.csr_matrix(Dx_2d), sp.csr_matrix(Dy_2d)]  # nabla operator
        self.laplace = sp.csr_matrix(D2x_2d + D2y_2d)  # laplace operator
        return self.ddx

    @profile
    def build_FD_matrices_1d(self, approx_order: int = 2, x: Array | None = None) -> list[AffineOperator]:
        """
        Build 1D finite difference differentiation matrices.

        Uses the Fornberg (1988) algorithm for generating weights.

        Parameters
        ----------
        approx_order
            The desired approximation order of the finite difference scheme.
        x
            The spatial grid points. If None, uses `self.x[0]`.

        Returns
        -------
        list of AffineOperator
            The list of 1D differentiation operators.
        """
        # accuracy / approximation order of the FD scheme (size of stencil = 2*ao + 1)
        ao = 2 * (approx_order // 2)  # has to be an even number
        # maximum derivative order of operators to build
        max_order = 2
        # Use provided x or default to self.x[0]
        grid_x = x if x is not None else (self.x[0] if self.x is not None else np.array([]))
        N = len(grid_x)
        # check if boundary conditions are set, otherwise set default
        if self.bc is None:
            self.bc = PeriodicBC()
            print(
                "WARNING: No boundary conditions set for:",
                self,
                "\n"
                "Defaulting to periodic boundaries. To change this "
                "(and to supress this warning), use, e.g.:\n"
                "  self.bc = DirichletBC()\n"
                "or any other boundary conditions type from pde.finite_differences, "
                "before building the FD matrices in the Equation object.",
            )
        # build boundary condition operators given the grid points and approximation order
        self.bc.update(grid_x, approx_order=ao)
        # boundary conditions implement an affine operator (Q*u + G) with a matrix Q and
        # a constant G, that maps the unknowns u to a 'boundary padded vector' with ghost points
        # at the boundaries.
        # pad x vector with x-values of ghost nodes
        x_pad = self.bc.pad_x(grid_x)
        # the number of ghost points
        Ngp = len(x_pad) - N

        # list of differentiation operators up to desired order d^n / dx^n
        # NOTE: ddx matrices are of shape N x (N+Ngp), because they work on boundary padded vectors
        self.ddx_1d: list[AffineOperator] = []
        # higher order operators
        for order in range(0, max_order + 1):
            if order == 0:
                # trivial 0th order d^0 / dx^0 = 1
                op = sp.eye(N, N + Ngp, k=Ngp // 2)
            else:
                # obtain differentiation matrix from findiff package
                op = (findiff.Diff(0, x_pad, acc=ao) ** order).matrix(x_pad.shape)
                # slice rows corresponding to ghost nodes
                op = op[Ngp // 2 : N + Ngp // 2, :]
            # include the boundary conditions into the differentiation operator by pre-multiplying
            # it with the (affine) boundary operator (matrix Q and constant G)
            opQ = op.dot(self.bc.Q)
            opG = op.dot(np.zeros(N + Ngp) + self.bc.G)
            # store as new affine operator Op(u) = ddx*(Q*u + G) = opQ*u + opG
            self.ddx_1d.append(AffineOperator(opQ, opG))

        # special names for some operators:
        # nabla operator: d / dx
        self.nabla = self.ddx_1d[1]
        # laplace operator: d^2 / dx^2
        self.laplace = self.ddx_1d[2]
        # return the resulting list of FD matrices
        return self.ddx_1d

    def jacobian(self, u: Array) -> Matrix:
        """
        Calculate the Jacobian of the equation.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        Matrix
            The Jacobian matrix, typically as a CSR sparse matrix.
        """
        # FD Jacobians are typically sparse, so we convert to a sparse matrix
        return sp.csr_matrix(super().jacobian(u))

    @profile
    def adapt(self) -> tuple[float, float] | None:
        """
        Perform adaption of the grid to the solution.

        Currently only supported for 1D spatial domains. Interpolates
        unknowns and history to the new grid points.
        """
        # mesh adaption is only supported for 1d
        if self.spatial_dimension > 1:
            return None
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
            dx = x_old[i + 1] - x_old[i - 1]
            if err < self.min_refinement_error and dx < self.max_dx:
                x_new.append(x_old[i + 1])
                i += 2
                continue
            # refinement
            dx = (x - x_old[i - 1]) / 2
            if err > self.max_refinement_error and dx > self.min_dx:
                x_new.append((x + x_old[i - 1]) / 2)
            x_new.append(x)
            i += 1
        x_new_arr: RealArray = np.asarray(x_new, dtype=np.float64)

        def _interpolate(data: Array, old_grid: RealArray, new_grid: RealArray) -> Array:
            """Interpolate real or complex data."""
            if np.iscomplexobj(data):
                res_re = np.interp(new_grid, old_grid, data.real)
                res_im = np.interp(new_grid, old_grid, data.imag)
                return np.asanyarray(res_re + 1j * res_im, dtype=data.dtype)
            return np.asanyarray(np.interp(new_grid, old_grid, data), dtype=data.dtype)

        # interpolate unknowns to new grid points
        nvars = self.shape[0] if len(self.shape) > 1 else 1
        u_new: Array
        if nvars > 1:
            u_new = np.array([_interpolate(self.u[n], x_old, x_new_arr) for n in range(nvars)])
        else:
            u_new = np.atleast_1d(_interpolate(self.u, x_old, x_new_arr))
        # update shape, u and x
        self.reshape(u_new.shape)
        self.u = u_new
        self.x = [x_new_arr]
        # interpolate history to new grid points
        for t, u_hist in enumerate(self.u_history):
            if nvars > 1:
                self.u_history[t] = np.array([_interpolate(u_hist[n], x_old, x_new_arr) for n in range(nvars)])
            else:
                self.u_history[t] = np.atleast_1d(_interpolate(u_hist, x_old, x_new_arr))
        # re-build the finite difference matrices
        self.build_FD_matrices()
        return (float(min(error_estimate)), float(max(error_estimate)))

    @profile
    def refinement_error_estimate(self) -> Array:
        """
        Estimate the error made in each grid point.

        Calculates the integral of curvature as an error estimate.

        Returns
        -------
        Array
            The error estimate at each grid point.
        """
        # calculate integral of curvature:
        # error = | \int d^2 u / dx^2 * test(x) dx |
        # NOTE: overwrite this method, if different weights of the curvatures are needed
        assert self.x is not None
        dx = np.diff(self.x[0])
        dx_vals = [max(dx[i], dx[i + 1]) for i in range(len(dx) - 1)]
        dx_padded = np.concatenate(([0.0], dx_vals, [0.0]))
        err: Array = np.zeros_like(dx_padded)
        nvars = self.shape[0] if len(self.shape) > 1 else 1
        for n in range(nvars):
            u = self.u[n] if len(self.shape) > 1 else self.u
            assert self.laplace is not None
            # laplace can be an AffineOperator or a list of matrices
            if isinstance(self.laplace, AffineOperator):
                curv = self.laplace(u)
            else:
                curv = self.laplace.dot(u)
            err = err + np.abs(curv * dx_padded)
        return np.asarray(err)

    def du_dx(self, u: Array | None = None, direction: int = 0) -> Array:
        """
        Calculate the spatial derivative in a given direction.

        Parameters
        ----------
        u
            The vector of unknowns.
        direction
            The spatial direction index.

        Returns
        -------
        Array
            The spatial derivative vector.
        """
        if u is None:
            u = self.u
        assert self.nabla is not None
        if isinstance(self.nabla, AffineOperator):
            return np.asanyarray(self.nabla(u))
        if isinstance(self.nabla, list):
            return np.asanyarray(self.nabla[direction].dot(u))
        raise TypeError(f"nabla has unexpected type: {type(self.nabla)}")

    def save(self) -> DataDict:
        """
        Save the state of the equation, including the x-values.

        Returns
        -------
        dict
            The state dictionary.
        """
        data = super().save()
        data.update({"x": self.x})
        return data

    def load(self, data: DataDict) -> None:
        """
        Load the state of the equation, including the x-values.

        Parameters
        ----------
        data
            The state dictionary to load from.
        """
        self.x = data["x"]
        super().load(data)


class AffineOperator:
    """
    Wrapper object for an affine operator of the form Op: u --> Q*u + G.

    Used for including boundary conditions into differentiation operators.
    """

    def __init__(self, Q: Matrix, G: float | Array = 0):
        """
        Initialize the AffineOperator.

        Parameters
        ----------
        Q
            The linear part (matrix).
        G
            The constant (affine) part.
        """
        #: linear part
        self.Q: Matrix = Q
        #: constant (affine) part
        self.G: float | Array = G

    def __call__(self, u: Array | None = None, g: float = 1.0) -> Matrix | Array:
        """
        Apply the operator to a vector/tensor u.

        Parameters
        ----------
        u
            The vector/tensor to apply the operator to. If None, returns the
            linear matrix part Q.
        g
            Scaling factor for the constant part G.

        Returns
        -------
        Union[Matrix, Array]
            The result of Q*u + g*G, or the matrix Q if u is None.
        """
        # if u is not given, return the matrix Q alone
        if u is None:
            return self.Q
        # check if u is not a vector (but a matrix or tensor)
        # make sure that a sparse matrix is returned
        if u.ndim > 1:
            return self.Q.dot(u) + sp.coo_matrix(g * np.resize(self.G, u.shape))
        # else, u is a vector, simply perform the Q*u + G
        return np.asanyarray(self.Q.dot(u) + g * self.G)

    def dot(self, u: Array) -> Array:
        """
        Apply the operator to a vector u.

        Equivalent to `self.__call__(u)`.

        Parameters
        ----------
        u
            The vector to apply the operator to.

        Returns
        -------
        Array
            The result of the operator application.
        """
        return np.asanyarray(self.__call__(u))

    def is_linear(self) -> bool:
        """
        Check if the operator is linear (i.e., G=0).

        Returns
        -------
        bool
            True if linear, False otherwise.
        """
        return not np.any(self.G)


class FDBoundaryConditions:
    """
    Base class for finite difference boundary conditions.

    Applied using an affine transformation u_pad = Q*u + G.
    """

    def __init__(self) -> None:
        """Initialize the boundary conditions."""
        # linear part: ((N+Ngp) x N)-matrix)
        #: linear part of the boundary operator
        self.Q: Matrix | None = None
        #: constant part of the boundary operator
        self.G: float | Array = 0.0

    def update(self, x: Array, approx_order: int) -> None:
        """
        Build the matrix and constant part for the boundary transformation.

        Parameters
        ----------
        x
            The spatial grid points.
        approx_order
            The approximation order of the FD scheme.
        """
        # default case: identity matrix (equals homogeneous Dirichlet conditions)
        N = len(x)
        ao = approx_order
        self.Q = sp.eye(N + 2, N, k=-ao)
        self.G = 0.0

    def pad(self, u: Array) -> Array:
        """
        Transform a vector u to the boundary padded vector u_pad = Q*u + G.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        Array
            The padded vector.
        """
        assert self.Q is not None
        return np.asanyarray(self.Q.dot(u) + self.G)

    def pad_x(self, x: Array) -> Array:
        """
        Pad the vector of node values with the x-values of the ghost nodes.

        Parameters
        ----------
        x
            The spatial grid points.

        Returns
        -------
        Array
            The padded grid points.
        """
        dxl = x[1] - x[0]
        dxr = x[-1] - x[-2]
        return np.concatenate(([x[0] - dxl], x, [x[-1] + dxr]))


class PeriodicBC(FDBoundaryConditions):
    """Periodic boundary conditions for finite difference schemes."""

    def __init__(self) -> None:
        """Initialize the periodic boundary conditions."""
        super().__init__()
        #: how many ghost nodes at each boundary?
        self.order = 1
        #: the virtual distance between the left and right boundary node
        self.boundary_dx: float | None = None

    def update(self, x: Array, approx_order: int) -> None:
        """
        Build the transformation for periodic boundaries.

        Parameters
        ----------
        x
            The spatial grid points.
        approx_order
            The approximation order.
        """
        # generate matrix that maps u_i --> u_{i%N} for 1d periodic ghost points
        self.order = approx_order
        N = len(x)
        top = sp.eye(self.order, N, k=N - self.order)
        bot = sp.eye(self.order, N, k=0)
        self.Q = sp.vstack((top, sp.eye(N), bot))
        # constant part is zero
        self.G = 0

    def pad_x(self, x: Array) -> Array:
        """
        Pad grid points with periodic ghost points.

        Parameters
        ----------
        x
            The grid points.

        Returns
        -------
        Array
            The padded grid points.
        """
        # obtain the (constant!) virtual distance between the left and right boundary nodes
        if self.boundary_dx is None:
            self.boundary_dx = x[1] - x[0]
        dx_lr = np.array([self.boundary_dx])
        # build the full list of dx's for the periodic domain
        dx = np.concatenate((dx_lr, np.diff(x), dx_lr))
        # construct the left and right ghost nodes (number of ghost points = self.order)
        x_l = [x[0] - sum(dx[-n - 1 :]) for n in range(self.order)][::-1]
        x_r = [x[-1] + sum(dx[: n + 1]) for n in range(self.order)]
        # concatenate for full padded x vector
        return np.concatenate((x_l, x, x_r))


class RobinBC(FDBoundaryConditions):
    """Robin boundary conditions of the form a*u(x_b) + b*u'(x_b) = c."""

    def __init__(
        self,
        a: tuple[float, float] = (0.0, 0.0),
        b: tuple[float, float] = (1.0, 1.0),
        c: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """
        Initialize Robin boundary conditions.

        Parameters
        ----------
        a
            Tuple (left, right) of coefficients for u.
        b
            Tuple (left, right) of coefficients for u'.
        c
            Tuple (left, right) of values for the boundary condition.
        """
        super().__init__()
        # store coefficients
        self.a = a
        self.b = b
        self.c = c

    @profile
    def update(self, x: Array, approx_order: int) -> None:
        """
        Build the transformation for Robin boundaries.

        Parameters
        ----------
        x
            The grid points.
        approx_order
            The approximation order.
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
        sl = fornberg.fd_weights(x=x[: ao + 1], x0=x[0], n=1)
        sr = fornberg.fd_weights(x=x[-ao - 1 :], x0=x[-1], n=1)
        # linear part (Q)
        top = np.zeros((1, N))
        top[-1, :ao] = -sl[1:] / (al / bl + sl[0]) if bl != 0 else 0 * sl[1:]
        bot = np.zeros((1, N))
        bot[0, -ao:] = -sr[:-1] / (ar / br + sr[-1]) if br != 0 else 0 * sr[1:]
        self.Q = sp.vstack((top, sp.eye(N), bot))
        # constant part (G)
        self.G = np.zeros(N + 2)
        self.G[0] = cl / (al + bl * sl[0]) if cl != 0 else 0
        self.G[-1] = cr / (ar + br * sr[-1]) if cr != 0 else 0


def DirichletBC(vals: tuple[float, float] = (0.0, 0.0)) -> RobinBC:
    """
    Create Dirichlet boundary conditions: u(left) = vals[0], u(right) = vals[1].

    Parameters
    ----------
    vals
        Tuple of (left, right) values.

    Returns
    -------
    RobinBC
        The configured boundary conditions.
    """
    return RobinBC(a=(1.0, 1.0), b=(0.0, 0.0), c=vals)


def NeumannBC(vals: tuple[float, float] = (0.0, 0.0)) -> RobinBC:
    """
    Create Neumann boundary conditions: u'(left) = vals[0], u'(right) = vals[1].

    Parameters
    ----------
    vals
        Tuple of (left, right) derivative values.

    Returns
    -------
    RobinBC
        The configured boundary conditions.
    """
    return RobinBC(a=(0.0, 0.0), b=(1.0, 1.0), c=vals)


class NoBoundaryConditions(FDBoundaryConditions):
    """
    Boundary conditions with no ghost points.

    Useful for building differentiation matrices that do not imply any
    specific boundary conditions.
    """

    def update(self, x: Array, approx_order: int) -> None:
        """
        Build an identity transformation.

        Parameters
        ----------
        x
            The grid points.
        approx_order
            The approximation order.
        """
        # default case: identity matrix
        # no ghost points, mapping is identity: boundary padded u = u = 1 * u + 0
        N = len(x)
        self.Q = sp.eye(N)
        self.G = 0

    def pad_x(self, x: Array) -> Array:
        """
        No padding for grid points.

        Parameters
        ----------
        x
            The grid points.

        Returns
        -------
        Array
            The original grid points.
        """
        return x
