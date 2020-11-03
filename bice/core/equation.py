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
    An equation has a 'shape' that should at all times be equal to the shape of the unknowns 'u'.
    """

    def __init__(self, shape=None):
        # The equation's storage for the unknowns
        self.u = np.zeros(shape)
        # we keep our own __shape variable, so that the shape is not unintentionally lost
        # when the user changes u. If stored shape is undefined, we'll simply fallback to u.shape
        self.__shape = self.u.shape
        # a history of the unknowns, needed e.g. for implicit schemes
        self.u_history = []
        # optional reference to group of equations that this equation belongs to
        self.group = None
        # Does the equation couple to any other unknowns?
        # If it is coupled, then all unknowns and methods of this equation will have the
        # full dimension of the problem and need to be mapped to the equation's
        # variables accordingly. Otherwise, they only have the dimension of this equation.
        self.is_coupled = False

    # The total number of unknowns / degrees of freedom of the equation
    @property
    def ndofs(self):
        return np.prod(self.shape)

    # Returns the shape of the equation's unknowns: self.u.shape
    @property
    def shape(self):
        # if no shape is explicitly assigned, just return u.shape
        if self.__shape is None or len(self.__shape) == 0:
            return self.u.shape
        # else, return the assigned shape
        return self.__shape

    # Change the shape of the equation / the equation's unknowns
    def reshape(self, shape):
        # resize the unknowns
        self.u = np.resize(self.u, shape)
        # update the internal shape variable
        self.__shape = self.u.shape
        # if the equation belongs to a group of equations, redo it's mapping of the unknowns
        # since the number of unknowns changed
        if self.group is not None:
            self.group.map_unknowns()

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
        J = np.zeros((N, N), dtype=u.dtype)
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

    # adapt the equation to the solution (mesh refinement or similar).
    # May be overridden for specific types of equations,
    # do not forget to adapt Equation.u_history as well!
    def adapt(self):
        pass

    # plot the solution into a matplotlib axes object
    def plot(self, ax):
        # check if there is spatial coordinates, otherwise generate fake coordinates
        try:
            x = self.x
        except AttributeError:
            x = [np.arange(self.shape[-1])]
        # for 1d
        if len(x) == 1:
            ax.set_xlabel("x")
            ax.set_ylabel("solution u(x,t)")
            # deal with the shape of u
            if len(self.shape) == 1:
                ax.plot(x[0], self.u)
            else:
                for n in range(self.shape[0]):
                    ax.plot(x[0], self.u[n])
        # for 2d
        if len(x) == 2:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            mx, my = np.meshgrid(x[0], x[1])
            u = self.u.reshape((x[0].size, x[1].size))
            ax.pcolormesh(mx, my, u)


class EquationGroup:
    """
    An EquationGroup groups multiple equations into a single new equation (a system of equations).
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
        res = np.zeros(self.ndofs, dtype=u.dtype)
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
        J = np.zeros((self.ndofs, self.ndofs), dtype=u.dtype)
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

    # pretty-print EquationGroups in the terminal
    def __repr__(self):
        res = super().__repr__()
        # prints tree structure of nested equations
        for i, eq in enumerate(self.equations):
            eq_repr = eq.__repr__()
            if i < len(self.equations)-1:
                res += "\n ├─" + eq_repr.replace("\n", "\n │ ")
            else:
                res += "\n └─" + eq_repr.replace("\n", "\n   ")
        return res
