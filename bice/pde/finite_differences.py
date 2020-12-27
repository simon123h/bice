import numpy as np
import findiff as fd
import fd_boundary_conditions as fdbc
import scipy.sparse
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
        # the spatial coordinates
        if len(self.shape) > 0:
            self.x = [np.linspace(0, 1, self.shape[-1], endpoint=False)]
        else:
            self.x = None

    # 4th order differentiation matrices for 1d, but with periodic boundaries
    # TODO: find a way to do this with findiff
    def build_periodic_FD_matrices(self):
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
        self.nabla = scipy.sparse.csr_matrix(self.nabla)
        self.laplace = scipy.sparse.csr_matrix(self.laplace)

    # generate differentiation matrices using finite difference scheme.
    # Arguments:
    #  - acc: stencil accuracy (integer)
    #  - max_order: maximum derivative order (integer)
    #  - uniform: is the spatial grid uniform or non-uniform? (boolean)

    def build_FD_matrices(self, acc=4, max_order=2, uniform=True):
        # shape of x-array
        xshape = tuple([len(x) for x in self.x])
        # 0th order derivative d^0 / dx^0 = 1
        self.ddx = [[1 for _ in self.x]]
        # generate differentiation matrices for orders 1 ... max_order
        for order in range(1, max_order+1):
            self.ddx.append([])
            # for each spatial dimension
            for dim, x in enumerate(self.x):
                x = self.x[dim]
                # distinguish between uniform and non-uniform grids
                dx = x[1] - x[0] if uniform else x
                # generate differentiation matrix for desired order and dimension
                diff_mat = fd.FinDiff(dim, dx, order, acc=acc).matrix(xshape)
                self.ddx[order].append(diff_mat)

        # generate nabla operator (gradient)
        self.nabla = self.ddx[1]
        if len(self.nabla) == 1:
            self.nabla = self.nabla[0]

        # generate Laplace operator
        self.laplace = sum(self.ddx[2])

    # calculate the spatial derivative du/dx in a given spatial direction
    def du_dx(self, u=None, direction=0):
        # if u is not given, use self.u
        if u is None:
            u = self.u
        # multiply with FD nabla matrix
        return self.nabla.dot(u)

    # generate a FinDiff boundary conditions object
    # needs to be converted to matrices manually before using in rhs
    def boundary_conditions(self):
        # shape of x-array
        xshape = tuple([len(x) for x in self.x])
        # generate boundary conditions object
        return fd.BoundaryConditions(xshape)

    # build boundary matrices to impose Neumann/Dirichlet boundary conditions
    # in the rhs of a finite differences equation
    # Arguments:
    #  - dirichlet_map: list of boolean tuples (left, right) that determine
    #                   whether to use Dirichlet (true) or Neumann (false)
    #                   boundary conditions on the left and right boundaries
    #                   of the respective dimension
    # - boundary_value: the value of the Dirichlet / Neumann condition at the
    #                    boundary, e.g., u(x=bound) = value or u'(x=bound) = value
    # - uniform: whether the spatial grid is uniform or non-uniform
    # TODO: untested, might possibly be wrong! See FinDiff documentation on boundary conditions:
    #       https://github.com/maroba/findiff
    def build_boundary_matrices(self, dirichlet_map, boundary_value, uniform=True):
        # generate boundary conditions object
        bc = self.boundary_conditions()
        # tuple of slices for each dimension
        full_idx = tuple([slice(len(x)) for x in self.x])
        # for each spatial dimension
        for dim, x in enumerate(self.x):
            x = self.x[dim]
            # distinguish between uniform and non-uniform grids
            dx = x[1] - x[0] if uniform else x
            # should left and right boundaries have dirichlet or neumann conditions?
            dirichlet_left, dirichlet_right = dirichlet_map[dim]
            value_left, value_right = boundary_value[dim]
            # differentiation operator for neumann conditions
            ddx = fd.FinDiff(dim, dx, 1)
            # left side
            idx = list(full_idx)
            idx[dim] = 0
            idx = tuple(idx)
            if dirichlet_left:  # dirichlet
                bc[idx] = value_left
            else:  # neumann
                bc[idx] = ddx, value_left
            # right side side
            idx = list(full_idx)
            idx[dim] = 0
            idx = tuple(idx)
            if dirichlet_right:  # dirichlet
                bc[idx] = value_right
            else:  # neumann
                bc[idx] = ddx, value_right
        # convert boundary matrix to sparse csr format
        Q = bc.lhs.tocsr()
        # convert boundary weight array to 1d array
        G = bc.rhs.toarray().ravel()
        # return boundary matrix and weights
        return Q, G
