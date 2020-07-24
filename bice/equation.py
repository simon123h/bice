import numpy as np


class Equation:
    """
    TODO: add docstring
    """

    def __init__(self):
        # Does the equation couple to any other unknowns?
        # If it is coupled, the all unknowns and methods of this equation will have the
        # full dimension of the problem and need to be mapped to the equation's
        # variables accordingly. Otherwise, they only have the dimension of this equation.
        self.is_coupled = False
        # Indices for the mapping from Problem.u to Equation.u: eq.u = problem.u[eq.idx]
        self.idx = None
        # the problem that the equation belongs to
        self.problem = None
        # the equation's storage for the unknowns if it is not currently part of a problem
        self.__u = None

    # Getter for the vector of unknowns
    @property
    def u(self):
        if self.problem is None:
            # return the unknowns that are stored in the equation itself
            return self.__u
        # fetch the unknowns from the problem with the equation mapping
        return self.problem.u[self.idx]

    # Getter for the vector of unknowns
    @u.setter
    def u(self, v):
        if self.problem is None:
            # update the unknowns stored in the equation itself
            self.__u = v
        else:
            # set the unknowns in the problem with the equation mapping
            self.problem.u[self.idx] = v

    # The number of unknowns / degrees of freedom of the equation
    @property
    def dim(self):
        return self.u.size

    # Calculate the right-hand side of the equation 0 = rhs(u)
    def rhs(self, u):
        raise NotImplementedError(
            "No right-hand side (rhs) implemented for this equation!")

    # Calculate the Jacobian of the system J = d rhs(u) / du for the unknowns u
    def jacobian(self, u):
        # default implementation: calculate Jacobian with finite differences
        J = np.zeros([self.dim, self.dim], dtype=np.float)
        for i in range(self.dim):
            eps = 1e-10
            u1 = u.copy()
            u2 = u.copy()
            u1[i] += eps
            u2[i] -= eps
            f1 = self.rhs(u1)
            f2 = self.rhs(u2)
            J[:, i] = (f1 - f2) / (2 * eps)
        return J

    # The mass matrix M determines the linear relation of the rhs to the temporal derivatives:
    # M * du/dt = rhs(u)
    def mass_matrix(self):
        # default case: assume the identity matrix I (--> du/dt = rhs(u))
        return np.eye(self.dim)


class FiniteDifferenceEquation:
    """
    TODO: add docstring
    """

    def __init__(self):
        # first order derivative
        self.nabla = None
        # second order derivative
        self.laplace = None
        # the spatial coordinates
        self.x = np.linspace(0, 1, 100, endpoint=False)

    def build_FD_matrices(self, N):
        # identity matrix
        I = np.eye(N)
        # spatial increment
        dx = self.x[1] - self.x[0]

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
