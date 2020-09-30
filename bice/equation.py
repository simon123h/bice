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

    def __init__(self, shape=(1,)):
        # the shape of the equation's unknowns: self.u.shape = self.shape
        # the shape may either be (N) for a single-variable equation with N values or
        # (nvariables, N) for a multi-variable (e.g. vector-valued) equation with N values
        # per independent variable
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        # Does the equation couple to any other unknowns?
        # If it is coupled, then all unknowns and methods of this equation will have the
        # full dimension of the problem and need to be mapped to the equation's
        # variables accordingly. Otherwise, they only have the dimension of this equation.
        self.is_coupled = False
        # the problem that the equation belongs to
        self.problem = None
        # Slice for the mapping from Problem.u to Equation.u: eq.u = problem.u[eq.idx]
        self.idx = None
        # List of slices for the mapping from Problem.u to unknowns assiociated with each variable in the equation
        self.var_idx = None
        # The equation's storage for the unknowns if it is not currently part of a problem
        self.__u = np.zeros(self.shape)

    # Getter for the vector of unknowns
    @property
    def u(self):
        if self.idx is None:
            # return the unknowns that are stored in the equation itself
            return self.__u.reshape(self.shape)
        # fetch the unknowns from the problem with the equation mapping
        return self.problem.u[self.idx].reshape(self.shape)

    # Setter for the vector of unknowns
    @u.setter
    def u(self, v):
        if self.idx is None:
            # update the unknowns stored in the equation itself
            self.__u = v.reshape(self.shape)
        else:
            # set the unknowns in the problem with the equation mapping
            self.problem.u[self.idx] = v.ravel()

    # the number of unknowns per independent variable in the equation
    @property
    def dim(self):
        if len(self.shape) == 1:
            return self.shape[0]
        return self.shape[1]

    @dim.setter
    def dim(self, d):
        if len(self.shape) == 1:
            self.shape = (d,)
        else:
            self.shape = (self.shape[0], d)

    # the number of independent variables in the equation (e.g. for vector-valued equations)
    @property
    def nvariables(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[0]

    @nvariables.setter
    def nvariables(self, n):
        if len(self.shape) == 1 and n == 1:
            pass
        else:
            self.shape[0] = (n, self.dim)

    # The total number of unknowns / degrees of freedom of the equation
    @property
    def ndofs(self):
        return self.nvariables * self.dim

    # Calculate the right-hand side of the equation 0 = rhs(u)
    def rhs(self, u):
        raise NotImplementedError(
            "No right-hand side (rhs) implemented for this equation!")

    # Calculate the Jacobian of the system J = d rhs(u) / du for the unknowns u
    @profile
    def jacobian(self, u):
        # default implementation: calculate Jacobian with finite differences
        eps = 1e-10
        use_central_differences = False
        N = u.size
        J = np.zeros((N, N))
        # uncoupled equations require u to be reshaped the self.shape before calling rhs(u)
        shape = u.shape if self.is_coupled else self.shape
        # reference rhs without central differences
        if not use_central_differences:
            f0 = self.rhs(u.reshape(shape)).ravel()
        u1 = u.copy().ravel()
        # perturb every degree of freedom and calculate Jacobian using FD
        for i in np.arange(N):
            k = u1[i]
            u1[i] = k + eps
            f1 = self.rhs(u1.reshape(shape)).ravel()
            if use_central_differences:
                # central difference
                u1[i] = k - eps
                f2 = self.rhs(u1.reshape(shape)).ravel()
                J[i] = (f1 - f2) / (2*eps)
            else:
                # forward difference
                J[i] = (f1 - f0) / eps
            u1[i] = k
        return J.T

    # The mass matrix M determines the linear relation of the rhs to the temporal derivatives:
    # M * du/dt = rhs(u)
    def mass_matrix(self):
        # default case: assume the identity matrix I (--> du/dt = rhs(u))
        # TODO: should be a sparse matrix
        return np.eye(self.ndofs)

    # This method is called before each evaluation of the rhs/Jacobian and may be
    # overwritten to do anything specific to the equation
    def actions_before_evaluation(self, u):
        pass

    # This method is called after each newton solve and may be
    # overwritten to do anything specific to the equation
    def actions_after_newton_solve(self):
        pass

    # plot the solution into a matplotlib axes object
    def plot(self, ax):
        if len(self.x) == 1:
            ax.set_xlabel("x")
            ax.set_ylabel("solution u(x,t)")
            # deal with the shape of u (1d vs 2d)
            if len(self.shape) == 1:
                ax.plot(self.x[0], self.u)
            else:
                for n in range(self.nvariables):
                    ax.plot(self.x[0], self.u[n])
        if len(self.x) == 2:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            mx, my = np.meshgrid(self.x[0], self.x[1])
            # deal with the shape of u (1d vs 2d)
            if len(self.shape) == 1:
                u = self.u
            else:
                # plot only the first variable
                u = self.u[0]
            u = u.reshape((self.x[0].size, self.x[1].size))
            ax.pcolormesh(mx, my, u)


class FiniteDifferenceEquation(Equation):
    """
    The FiniteDifferenceEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs with a finite difference scheme.
    """

    def __init__(self, shape=(1,)):
        super().__init__(shape)
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


class PseudospectralEquation(Equation):
    """
    The PseudospectralEquation is a subclass of the general Equation
    and provides some useful routines that are needed for implementing
    ODEs/PDEs with a pseudospectral scheme.
    """

    def __init__(self, shape=(1,)):
        super().__init__(shape)
        # the spatial coordinates
        self.x = None
        self.k = None
        self.ksquare = None

    def build_kvectors(self):
        if len(self.x) == 1:
            Lx = self.x[0][-1] - self.x[0][0]
            Nx = self.x[0].size
            # the fourier space
            self.k = [np.fft.fftfreq(Nx, Lx / (2. * Nx * np.pi))]
            self.ksquare = self.k[0]**2
        elif len(self.x) == 2:
            Lx = self.x[0][-1] - self.x[0][0]
            Nx = self.x[0].size
            Ly = self.x[1][-1] - self.x[1][0]
            Ny = self.x[1].size
            # the fourier space
            kx = np.fft.fftfreq(Nx, Lx / (2. * Nx * np.pi))
            ky = np.fft.fftfreq(Ny, Ly / (2. * Ny * np.pi))
            kx, ky = np.meshgrid(kx, ky)
            self.k = [kx, ky]
            self.ksquare = kx**2 + ky**2
        elif len(self.x) == 3:
            Lx = self.x[0][-1] - self.x[0][0]
            Nx = self.x[0].size
            Ly = self.x[1][-1] - self.x[1][0]
            Ny = self.x[1].size
            Lz = self.x[2][-1] - self.x[2][0]
            Nz = self.x[2].size
            # the fourier space
            kx = np.fft.fftfreq(Nx, Lx / (2. * Nx * np.pi))
            ky = np.fft.fftfreq(Ny, Ly / (2. * Ny * np.pi))
            kz = np.fft.fftfreq(Nz, Lz / (2. * Nz * np.pi))
            kx, ky, kz = np.meshgrid(kx, ky, kz)
            self.k = [kx, ky, kz]
            self.ksquare = kx**2 + ky**2 + kz**2
