from __future__ import annotations

from typing import Optional, Union, List

import numpy as np
import scipy.sparse as sp

from .profiling import profile
from .types import Array, Matrix, Shape


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

    def __init__(self, shape: Optional[Shape] = None) -> None:
        #: The equation's storage for the unknowns
        self.u: Array = np.zeros(() if shape is None else shape)
        # we keep our own __shape variable, so that the shape is not unintentionally lost
        # when the user changes u. If stored shape is undefined, we'll simply fallback to u.shape
        self.__shape = self.u.shape
        #: a history of the unknowns, needed e.g. for implicit schemes
        self.u_history: list[Array] = []
        #: optional reference to group of equations that this equation belongs to
        self.group: Optional[EquationGroup] = None
        #: Does the equation couple to any other unknowns?
        #: If it is coupled, then all unknowns and methods of this equation will have the
        #: full dimension of the problem and need to be mapped to the equation's
        #: variables accordingly. Otherwise, they only have the dimension of this equation.
        self.is_coupled = False

    @property
    def ndofs(self) -> int:
        """The total number of unknowns / degrees of freedom of the equation"""
        return np.prod(self.shape)

    @property
    def shape(self) -> tuple:
        """Returns the shape of the equation's unknowns: self.u.shape"""
        # if no shape is explicitly assigned, just return u.shape
        if self.__shape is None or len(self.__shape) == 0:
            return self.u.shape
        # else, return the assigned shape
        return self.__shape

    def reshape(self, shape: Shape) -> None:
        """Change the shape of the equation / the equation's unknowns"""
        # resize the unknowns
        self.u = np.resize(self.u, shape)
        # update the internal shape variable
        self.__shape = self.u.shape
        # if the equation belongs to a group of equations, redo it's mapping of the unknowns
        # since the number of unknowns changed
        if self.group is not None:
            self.group.map_unknowns()

    def rhs(self, u: Array) -> Array:
        """Calculate the right-hand side of the equation 0 = rhs(u)"""
        raise NotImplementedError(
            "No right-hand side (rhs) implemented for this equation!")

    @profile
    def jacobian(self, u: Array) -> Matrix:
        """
        Calculate the Jacobian J = d rhs(u) / du for the unknowns u.
        Defaults to automatic calculation of the Jacobian using finite differences.
        """
        eps = 1e-10  # the finite perturbation size
        use_central_differences = False  # use central or forward differences?
        N = u.size
        J = np.zeros((N, N), dtype=u.dtype)
        # uncoupled equations require u to be reshaped the self.shape before calling rhs(u)
        shape = u.shape if self.is_coupled else self.shape
        # make a copy of the unknowns
        u1 = u.copy().ravel()
        if use_central_differences:
            # perturb every degree of freedom and calculate Jacobian using central FD
            for i in np.arange(N):
                k = u1[i]
                u1[i] = k + eps
                f1 = self.rhs(u1.reshape(shape)).ravel()
                u1[i] = k - eps
                f2 = self.rhs(u1.reshape(shape)).ravel()
                J[i] = (f1 - f2) / (2*eps)
                u1[i] = k
        else:  # use forward differences
            # reference rhs for unperturbed u
            f0 = self.rhs(u.reshape(shape)).ravel()
            # perturb every degree of freedom and calculate Jacobian using FD
            for i in np.arange(N):
                k = u1[i]
                u1[i] = k + eps
                f1 = self.rhs(u1.reshape(shape)).ravel()
                J[i] = (f1 - f0) / eps
                u1[i] = k
        return J.T

    def mass_matrix(self) -> Matrix:
        """
        The mass matrix M determines the linear relation of the rhs to the temporal derivatives:
        M * du/dt = rhs(u)
        """
        # default case: assume the identity matrix I (--> du/dt = rhs(u))
        return sp.eye(self.ndofs)

    def adapt(self) -> None:
        """
        Adapt the equation to the solution (mesh refinement or similar).
        May be overridden for specific types of equations,
        do not forget to adapt Equation.u_history as well!
        """
        pass

    def save(self) -> dict:
        """
        Save everything that is relevant for this equation to a dict. The Problem class
        will call this and save the dict to the disk.
        May be overridden for saving more stuff for specific types of equations.
        """
        return {'u': self.u}

    def load(self, data) -> None:
        """
        Restore unknowns / parameters / etc. from the given dictionary, that was created by
        Equation.save(). Equation.load() is the inverse of Equation.save().
        May be overridden for loading more stuff for specific types of equations.
        """
        self.u = data['u']

    def plot(self, ax) -> None:
        """plot the solution into a matplotlib axes object"""
        # check if there is spatial coordinates, otherwise generate fake coordinates
        if hasattr(self, "x"):
            x = getattr(self, "x")
        else:
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

    def __init__(self, equations: Optional[List[EquationLike]] = None):
        #: the list of sub-equations (or even sub-groups-of-equations)
        self.equations = []
        #: The indices of the equation's unknowns to the group's unknowns and vice versa
        self.idx = {}
        #: optional reference to a parent EquationGroup
        self.group: Optional["EquationGroup"] = None
        # optionally add the given equations
        if equations is not None:
            for eq in equations:
                self.add_equation(eq)

    @property
    def ndofs(self) -> int:
        """The number of unknowns / degrees of freedom of the group"""
        return sum([eq.ndofs for eq in self.equations])

    @property
    def shape(self) -> Shape:
        """The shape of the unknowns"""
        return (self.ndofs,)

    @property
    def is_coupled(self) -> bool:
        """A group of equations should never couple to other groups"""
        return False

    @property
    def u(self) -> Array:
        """The unknowns of the system: combined unknowns of the sub-equations"""
        return np.concatenate([eq.u.ravel() for eq in self.equations])

    @u.setter
    def u(self, u) -> None:
        """set the unknowns"""
        for eq in self.equations:
            # extract the equation's unknowns using the mapping and reshape to the equation's shape
            eq.u = u[self.idx[eq]].reshape(eq.shape)

    def add_equation(self, eq: Union[Equation, 'EquationGroup']) -> None:
        """add an equation to the group"""
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

    def remove_equation(self, eq: EquationLike) -> None:
        """remove an equation from the group"""
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

    def map_unknowns(self) -> None:
        """
        Create the mapping from equation unknowns to group unknowns, in the sense
        that group.u[idx[eq]] = eq.u.ravel() where idx is the mapping
        """
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

    @profile
    def rhs(self, u: Array) -> Array:
        """Calculate the right-hand side of the group 0 = rhs(u)"""
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

    @profile
    def jacobian(self, u: Array) -> Matrix:
        """Calculate the Jacobian J = d rhs(u) / du for the unknowns u"""
        # if there is only one equation, we can return the matrix directly
        if len(self.equations) == 1:
            eq = self.equations[0]
            shape = u.shape if eq.is_coupled else eq.shape
            return eq.jacobian(u.reshape(shape))
        # otherwise, we need to assemble the matrix
        J = sp.csr_matrix((self.ndofs, self.ndofs), dtype=u.dtype)
        # add the Jacobian of each equation
        J_uncoupled = []
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                eq_jac = eq.jacobian(u)
                if not sp.issparse(eq_jac):
                    eq_jac = sp.csr_matrix(eq_jac)
                # simply add to the global Jacobian
                J += eq_jac
                # add dummy matrix to J_uncoupled
                J_uncoupled.append(sp.csr_matrix((eq.ndofs, eq.ndofs)))
            else:
                # uncoupled equations work on their own variables, so we do a mapping
                idx = self.idx[eq]
                eq_jac = eq.jacobian(u[idx].reshape(eq.shape))
                if not sp.issparse(eq_jac):
                    eq_jac = sp.csr_matrix(eq_jac)
                # add to list to later construct the global Jacobian
                J_uncoupled.append(eq_jac)
        # add contributions of uncoupled equations
        J += sp.block_diag(J_uncoupled, format="csr")
        # all entries assembled, return
        return J

    def mass_matrix(self) -> Matrix:
        """
        The mass matrix determines the linear relation of the rhs to the temporal derivatives:
        M * du/dt = rhs(u)
        """
        # if there is only one equation, we can return the matrix directly
        if len(self.equations) == 1:
            return self.equations[0].mass_matrix()
        # otherwise, we need to assemble the matrix
        M = sp.csr_matrix((self.ndofs, self.ndofs))
        # add the entries of each equation
        M_uncoupled = []
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                M += eq.mass_matrix()
                # add dummy matrix to M_uncoupled
                M_uncoupled.append(sp.csr_matrix((eq.ndofs, eq.ndofs)))
            else:
                # uncoupled equations simply work on their own variables, so we do a mapping
                M_uncoupled.append(eq.mass_matrix())
        # add contributions of uncoupled equations
        M += sp.block_diag(M_uncoupled, format="csr")
        # all entries assembled, return
        return M

    def list_equations(self) -> List[Equation]:
        """return a flattened list of all equations in the group and sub-groups"""
        res = []
        for eq in self.equations:
            if isinstance(eq, EquationGroup):
                # if it is a group of equations, traverse it
                res += eq.list_equations()
            elif isinstance(eq, Equation):
                # if it is an actual equation, add to the result list
                res.append(eq)
        return res

    def __repr__(self):
        """pretty-print EquationGroups in the terminal"""
        res = super().__repr__()
        # prints tree structure of nested equations
        for i, eq in enumerate(self.equations):
            eq_repr = eq.__repr__()
            if i < len(self.equations)-1:
                res += "\n ├─" + eq_repr.replace("\n", "\n │ ")
            else:
                res += "\n └─" + eq_repr.replace("\n", "\n   ")
        return res


# common type for Equations/EquationGroups
EquationLike = Union[Equation, EquationGroup]
