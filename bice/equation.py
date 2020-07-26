import numpy as np
import scipy.optimize
from .profiling import profile

class Equation:
    """
    The Equation class holds algebraic (Cauchy) equations of the form
    M du/dt = rhs(u, t, r)
    where M is the mass matrix, u is the vector of unknowns, t is the time
    and r is a parameter vector. This may include ODEs and PDEs.
    All custom equations must inherit from this class and implement the rhs(u) method.
    Time and parameters are implemented as member attributes.
    The general Equation class gives the general interface, takes care of some bookkeeping,
    i.e., mapping the equation's unknowns and variables to the ones of the Problem that
    the equation belongs to, and provides some general functionality that every equation
    should have.
    This is a very fundamental class. Specializations of the Equation class exist for covering
    more intricate types of equations, i.e., particular discretizations for spatial fields, e.g.,
    finite difference schemes or pseudospectral methods.
    """

    def __init__(self):
        # Does the equation couple to any other unknowns?
        # If it is coupled, then all unknowns and methods of this equation will have the
        # full dimension of the problem and need to be mapped to the equation's
        # variables accordingly. Otherwise, they only have the dimension of this equation.
        self.is_coupled = False
        # the problem that the equation belongs to
        self.problem = None
        # Slice for the mapping from Problem.u to Equation.u: eq.u = problem.u[eq.idx]
        self.idx = None
        # The equation's storage for the unknowns if it is not currently part of a problem
        self.__u = None
        # The list of parameter names (list of strings, must match the name of the attributes)
        self.parameter_names = []

    # Getter for the vector of unknowns
    @property
    def u(self):
        if self.idx is None:
            # return the unknowns that are stored in the equation itself
            return self.__u
        # fetch the unknowns from the problem with the equation mapping
        return self.problem.u[self.idx]

    # Setter for the vector of unknowns
    @u.setter
    def u(self, v):
        if self.idx is None:
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
    @profile
    def jacobian(self, u):
        # default implementation: calculate Jacobian with finite differences
        # TODO: is there some way to make this more performant?
        N = self.problem.dim if self.is_coupled else self.dim
        J = np.zeros([N, N], dtype=np.float)
        f0 = self.rhs(u)
        for i in np.arange(N):
            eps = 1e-10
            u1 = u.copy()
            u1[i] += eps
            f1 = self.rhs(u1)
            J[:, i] = (f1 - f0) / eps
            # alternatively, central differences:
            # u2 = u.copy()
            # u2[i] -= eps
            # f2 = self.rhs(u2)
            # J[:, i] = (f1 - f2) / (2 * eps)
        return J

    # The mass matrix M determines the linear relation of the rhs to the temporal derivatives:
    # M * du/dt = rhs(u)
    def mass_matrix(self):
        # default case: assume the identity matrix I (--> du/dt = rhs(u))
        return np.eye(self.dim)

    # plot the solution into a matplotlib axes object
    def plot(self, ax):
        if self.x.ndim == 1:
            ax.set_xlabel("x")
            ax.set_ylabel("solution u(x,t)")
            ax.plot(self.x, self.u)
        if self.x.ndim == 2:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            mx, my = np.meshgrid(self.x[0], self.x[1])
            # u = self.u.reshape((self.x[0].size, self.x[1].size))
            # ax.imshow(u)
        # ax.set_xlabel("k")
        # ax.set_ylabel("fourier spectrum u(k,t)")
        # ax.plot(self.she.k, np.abs(np.fft.rfft(self.she.u)))


class FiniteDifferenceEquation(Equation):
    """
    The FiniteDifferenceEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs with a finite difference scheme.
    """

    def __init__(self):
        super().__init__()
        # first order derivative
        self.nabla = None
        # second order derivative
        self.laplace = None
        # the spatial coordinates
        self.x = np.linspace(0, 1, 100, endpoint=False)

    def build_FD_matrices(self):
        N = self.dim
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
