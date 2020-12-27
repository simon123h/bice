import numpy as np
import scipy.sparse as sp


class FiniteDifferenceOperator:
    ...


class BoundaryConditions:
    """
    A boundary matrix is actually an affine transformation and consists of the matrix Q and the vector G
    Suppose we have a central FD scheme of order 'acc' and the unknowns u are discretized to N grid points.
    Then, the FD matrix D_x is a ((N+2*acc) x N)-matrix, that works on the unknowns vector padded with ghost points:
    u_padded = (ghost, ghost, u_0, u_1, ..., u_{N-1}, ghost, ghost)
    We define a linear operator that maps the unknowns to the padded unknowns: u_padded = Q * u
    The differentiation with the FD matrix is then: d_x u = D_x * Q * u.
    For inhomogeneous boundaries, one needs to add a constant part to the differentiation:
    d_x u = D_x * Q * u + G     (TODO: or maybe D_x * (Q* u + G) ?)
    The boundary matrix is a wrapper class for the matrix Q and vector G
    """
    # NOTE: the boundary conditions currently only support 1d grids!

    def __init__(self, N, acc=1):
        # accuracy of the FD scheme
        self.acc = acc
        # linear part (matrix)
        # maps u to padded boundary u (gp, gp, u0, u1, ...., u(N-1), gp, gp)
        self.Q = sp.eye(N+2*acc, N, k=acc)
        # constant part (vector)
        self.G = 0


# Dirichlet boundary conditions: u(left) = g_left, u(right) = g_right
class DirichletBC(BoundaryConditions):
    def __init__(self, N, acc=1, vals=(0, 0)):
        super().__init__(N, acc=acc)
        # linear part
        self.Q = sp.eye(N+2*acc, N, k=acc)
        # constant part
        self.G = np.zeros(N)
        # set the boundary values to the constant part
        self.set_value(*vals)

    # set the boundary values
    def set_value(self, left=0, right=0):
        # TODO: support acc > 1
        # TODO: prefactor? like -2/dx or so
        self.G[0] = left
        self.G[-1] = right


# Neumann boundary conditions: u'(left) = g_left, u'(right) = g_right
class NeumannBC(BoundaryConditions):
    def __init__(self, N, acc=1, vals=(0, 0)):
        super().__init__(N, acc=acc)
        # linear part
        I = sp.eye(N)
        antiI = sp.eye(acc)  # TODO: fix
        zero = sp.csr_matrix((acc, N-acc))
        top = sp.hstack((antiI, zero))
        bot = sp.hstack((zero, antiI))
        self.Q = sp.vstack((top, I, bot))
        # constant part
        self.G = np.zeros(N)
        # set the boundary values to the constant part
        self.set_value(*vals)

    # set the boundary values
    def set_value(self, left=0, right=0):
        # TODO: support acc > 1
        # TODO: prefactor? like -2/dx or so
        self.G[0] = left
        self.G[-1] = right

# periodic boundary condition
class PeriodicBC(BoundaryConditions):
    def __init__(self, N, acc=1):
        super().__init__(N, acc=acc)
        # constant part (vector)
        self.G = 0
        # generate matrix that maps u_i --> u_{i%N} for 1d periodic ghost points
        top = sp.eye(acc, N, k=N-acc)
        I = sp.eye(N)
        bot = sp.eye(acc, N, k=0)
        self.Q = sp.vstack((top, I, bot))
