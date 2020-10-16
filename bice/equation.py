import numpy as np
from .profiling import profile


class Equation:
    """
    The Equation class holds algebraic (Cauchy) equations of the form
    M du/dt = rhs(u, t, r)
    where M is the mass matrix, u is the vector of unknowns, t is the time
    and r is a parameter vector. This may include ODEs and PDEs.
    All custom equations must inherit from this class and implement the rhs(u) method.
    Time and parameters are implemented as member attributes.
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
        # The equation's storage for the unknowns
        self.u = np.zeros(self.shape)
        # Does the equation couple to any other unknowns?
        # If it is coupled, then all unknowns and methods of this equation will have the
        # full dimension of the problem and need to be mapped to the equation's
        # variables accordingly. Otherwise, they only have the dimension of this equation.
        self.is_coupled = False
        # optional reference to group of equations that this equation belongs to
        self.group = None

    # the number of unknowns per independent variable in the equation
    @property
    def dim(self):
        return self.shape[-1]

    @dim.setter
    def dim(self, d):
        self.shape = self.shape[:-1] + (d,)

    # the number of independent variables in the equation (e.g. for vector-valued equations)
    @property
    def nvariables(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[0]

    # The total number of unknowns / degrees of freedom of the equation
    @property
    def ndofs(self):
        return self.nvariables * self.dim

    # Calculate the right-hand side of the equation 0 = rhs(u)
    def rhs(self, u):
        raise NotImplementedError(
            "No right-hand side (rhs) implemented for this equation!")

    # Calculate the Jacobian J = d rhs(u) / du for the unknowns u
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
        # NOTE: could be a scipy.sparse matrix
        return np.eye(self.ndofs)

    # plot the solution into a matplotlib axes object
    def plot(self, ax):
        # check if there is spatial coordinates, otherwise generate fake coordinates
        if hasattr(self, 'x'):
            x = self.x
        else:
            x = [np.arange(self.dim)]
        if len(x) == 1:
            ax.set_xlabel("x")
            ax.set_ylabel("solution u(x,t)")
            # deal with the shape of u
            if len(self.shape) == 1:
                ax.plot(x[0], self.u)
            else:
                for n in range(self.nvariables):
                    ax.plot(x[0], self.u[n])
        if len(x) == 2:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            mx, my = np.meshgrid(x[0], x[1])
            # deal with the shape of u
            if len(self.shape) == 1:
                u = self.u
            else:
                # plot only the first variable
                u = self.u[0]
            u = u.reshape((x[0].size, x[1].size))
            ax.pcolormesh(mx, my, u)


class EquationGroup:
    """
    An EquationGroup or 'system of equations', groups multiple equations into a single large equation.
    All properties and functions are assembled from the subequations.
    EquationGroups may even form hierarchical trees, where one group of equations serves as a
    subequation to another one.
    """

    def __init__(self, equations=None):
        # the list of sub-equations (or even sub-groups-of-equations)
        self.equations = []
        # The indices of the equation's unknowns to the group's unknowns and vice versa
        self.idx = {}
        # optional reference to a parent EquationGroup
        self.group = None
        # optionally add the given equations
        if equations is not None:
            for eq in equations:
                self.add_equation(eq)

    # the number of unknowns per independent variable in the equation
    @property
    def dim(self):
        return self.ndofs

    # the number of independent variables (always = 1 for group)
    @property
    def nvariables(self):
        return 1

    # The number of unknowns / degrees of freedom of the group
    @property
    def ndofs(self):
        return sum([eq.ndofs for eq in self.equations])

    # The shape of the unknowns
    @property
    def shape(self):
        return (self.ndofs,)

    # A group of equations should never couple to other groups
    @property
    def is_coupled(self):
        return False

    # The unknowns of the system: combined unknowns of the sub-equations
    @property
    def u(self):
        return np.concatenate([eq.u.ravel() for eq in self.equations])

    # set the unknowns
    @u.setter
    def u(self, u):
        for eq in self.equations:
            # extract the equation's unknowns using the mapping and reshape to the equation's shape
            eq.u = u[self.idx[eq]].reshape(eq.shape)

    # add an equation to the group
    def add_equation(self, eq):
        # check if eq already in self.equations
        if eq in self.equations:
            print("Error: Equation is already part of this group!")
            return
        # check if eq already in other group
        if eq.group is not None:
            print("Error: Equation is already part of another group of equations!")
            return
        # append to list of equations
        self.equations.append(eq)
        # assign this group as the equation's group
        eq.group = self
        # redo the mapping from equation's to group's unknowns
        self.map_unknowns()

    # remove an equation from the group
    def remove_equation(self, eq):
        # check if eq in self.equations
        if eq not in self.equations:
            print("Error: Equation is not part of this group!")
            return
        # remove from the list of equations
        self.equations.remove(eq)
        # remove the equations association with the group
        eq.group = None
        # redo the mapping from equation's to group's unknowns
        self.map_unknowns()

    # create the mapping from equation unknowns to group unknowns, in the sense
    # that group.u[idx[eq]] = eq.u.ravel() where idx is the mapping
    def map_unknowns(self):
        # counter for the current position in group.u
        i = 0
        # assign index range for each equation according to their dimension
        for eq in self.equations:
            # unknowns / equations indexing
            # NOTE: It is very important for performance that this is a slice,
            #       not a range or anything else. Slices extract coherent parts
            #       of an array, which goes much much faster than extracting values
            #       from positions given by integer indices.
            # indices of the equation's unknowns in EquationGroup.u
            self.idx[eq] = slice(i, i+eq.ndofs)
            # increment counter by the equation's number of degrees of freedom
            i += eq.ndofs
        # if there is a parent group, update its mapping as well
        if self.group:
            self.group.map_unknowns()

    # Calculate the right-hand side of the group 0 = rhs(u)
    @profile
    def rhs(self, u):
        # if there is only one equation, we can return the rhs directly
        if len(self.equations) == 1:
            eq = self.equations[0]
            shape = u.shape if eq.is_coupled else eq.shape
            return eq.rhs(u.reshape(shape)).ravel()
        # otherwise, we need to assemble the result vector
        res = np.zeros(self.ndofs)
        # add the contributions of each equation
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                res += eq.rhs(u)
            else:
                # uncoupled equations simply work on their own variables, so we do the mapping
                idx = self.idx[eq]
                res[idx] += eq.rhs(u[idx].reshape(eq.shape)).ravel()
        # everything assembled, return result
        return res

    # Calculate the Jacobian J = d rhs(u) / du for the unknowns u
    @profile
    def jacobian(self, u):
        # if there is only one equation, we can return the matrix directly
        if len(self.equations) == 1:
            eq = self.equations[0]
            shape = u.shape if eq.is_coupled else eq.shape
            return eq.jacobian(u.reshape(shape))
        # otherwise, we need to assemble the matrix
        J = np.zeros((self.ndofs, self.ndofs))
        # add the Jacobian of each equation
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                J += eq.jacobian(u)
            else:
                # uncoupled equations simply work on their own variables, so we do a mapping
                idx = self.idx[eq]
                J[idx, idx] += eq.jacobian(u[idx].reshape(eq.shape))
        # all entries assembled, return
        return J

    # The mass matrix determines the linear relation of the rhs to the temporal derivatives:
    # M * du/dt = rhs(u)
    def mass_matrix(self):
        # if there is only one equation, we can return the matrix directly
        if len(self.equations) == 1:
            return self.equations[0].mass_matrix()
        # otherwise, we need to assemble the matrix
        M = np.zeros((self.ndofs, self.ndofs))
        # add the entries of each equation
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                M += eq.mass_matrix()
            else:
                # uncoupled equations simply work on their own variables, so we do a mapping
                idx = self.idx[eq]
                M[idx, idx] += eq.mass_matrix()
        # all entries assembled, return
        return M

    # return a flattened list of all equations in the group and sub-groups
    def list_equations(self):
        res = []
        for eq in self.equations:
            if isinstance(eq, EquationGroup):
                # if it is a group of equations, traverse it
                res += eq.list_equations()
            elif isinstance(eq, Equation):
                # if it is an actual equation, add to the result list
                res.append(eq)
        return res


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
        self.x = [np.linspace(0, 1, self.dim, endpoint=False)]

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
        self.x = [np.linspace(0, 1, self.dim)]
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
