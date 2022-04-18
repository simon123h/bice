import numpy as np
import scipy.sparse as sp
from bice.core.equation import Equation
from typing import Union, Any

from bice.core.types import Matrix


class BifurcationConstraint(Equation):
    # TODO: add docstring
    def __init__(self, phi: np.ndarray, free_parameter: tuple[Any, str]):
        super().__init__()
        # the constraint equation couples to some other equations of the problem
        self.is_coupled = True
        # copy and normalize the null-eigenvector phi
        phi = phi.copy() / np.linalg.norm(phi)
        #: reference to the free parameter
        self.free_parameter = free_parameter
        # get the value of the free parameter
        param_obj, param_name = tuple(self.free_parameter)
        parameter_value = getattr(param_obj, param_name)
        # the unknowns are the null-eigenvector and the value of the free parameter,
        # so we have N + 1 degrees of freedom, where N is the #dofs of the problem
        self.u = np.concatenate((phi, np.array([parameter_value])))

        # the constraint can disable itself with this attribute,
        # so that only the original (unextended) system is obtained,
        # preventing redundant, recursive or duplicate calculations
        self.__disabled = False

    def rhs(self, u: np.ndarray) -> Union[np.ndarray, float]:
        assert self.group is not None
        # if the constraint is disabled, no residuals will be calculated
        if self.__disabled:
            return 0
        # reference to the indices of the own unknowns
        self_idx = self.group.idx[self]
        # get the value of the current and the previous null-eigenvector phi
        phi = u[self_idx][:-1]
        phi_old = self.u[:-1]
        # calculate the original Jacobian of the problem
        Gu = self.original_jacobian(u)
        # check for mismatch in dimension
        if Gu.shape != (phi.size, phi.size):
            raise Exception("It seems that the dimension of the problem does not "
                            "match the dimension of the null-eigenvector phi in "
                            "the BifurcationConstraint. Did your problem change "
                            "since you imposed the constraint?")
        # calculate the residuals
        res = np.zeros((u.size))
        res1 = np.matmul(Gu, phi)
        res2 = np.array([np.dot(phi, phi_old) - 1])
        res[self_idx] = np.concatenate((res1, res2))
        return res

    def jacobian(self, u: np.ndarray) -> Union[Matrix, float]:
        # if the constraint is disabled, no Jacobian will be calculated
        if self.__disabled:
            return 0
        # pass Jacobian calculation to the FD method of the parent Equation class
        # NOTE: this is what takes so long, since it evaluates the rhs 2N+1 times
        #       and the rhs evaluates the original Jacobian... :-/
        return Equation.jacobian(self, u)

        # Alternative: quasi-analytical calculation...

        # empty result matrix
        J = np.zeros((u.size, u.size))
        r"""
            /   d(rhs_orig)/d(u_orig) [NxN]  |  d(rhs_orig)/d(phi) = 0  [NxN]  |                 \
            |   -----------------------------|---------------------------------|                 |
        J = |   d(res1)/d(u_orig)     [NxN]  |  d(res1)/d(phi) = Gu     [NxN]  | d(rhs)/d(param) |
            |   -----------------------------|---------------------------------|   [(2N+1)x1]    |
            \   d(res2)/d(u_orig) = 0 [1xN]  | d(res2)/d(phi) = phi_old [1xN]  |                 /
        """

        # upper left block: d(rhs_orig)/d(u_orig) will be assembled by their respective equations

        # reference to the indices of the own unknowns
        self_idx = self.group.idx[self]
        # get phi and phi_old
        phi = u[self_idx][:-1]
        phi_old = self.u[:-1]
        # calculate the original Jacobian of the problem
        Gu = self.original_jacobian(u)

        # lower left sub-block: d(res1)/d(u_orig), calculate with FD
        eps = 1e-10
        u1 = u.copy()
        for i in range(phi.size):
            k = u1[i]
            u1[i] = k + eps
            # calculate the original Jacobian of the problem
            # NOTE: okay, here we'd also need N times evaluation of the Jacobian, that's also slow
            Gu1 = self.original_jacobian(u1)
            # NOTE: the following line is not general, wrt. self_idx
            J[:, i] = np.matmul(Gu1-Gu, phi) / eps
            # .... I'm giving up at this point!

        # lower right sub-block: d(res1)/d(phi) = Gu
        res1_slice = slice(self_idx.start, self_idx.stop-1)
        J[res1_slice, res1_slice] = Gu

        # last column: d(rhs)/d(param), calculate with FD
        f0 = self.group.rhs(u)
        # deviate the free parameter value
        param_val = u[self_idx][-1]
        u[self_idx][-1] += eps
        # calculate new residuals
        f1 = self.group.rhs(u)
        # reset the value of the free parameter
        u[self_idx][-1] = param_val
        # add FD parameter derivative to Jacobian
        J[:, self_idx.stop-1] = (f1 - f0) / eps

        # last row: d(res2)/du = ([0]*N, phi_old, 0)
        J[self_idx.stop-1] = np.concatenate(
            (np.zeros(phi_old.size), phi_old, np.array([0])))

        return J

    def mass_matrix(self) -> float:
        # couples to no time-derivatives
        return 0

    def original_jacobian(self, u: np.ndarray) -> np.ndarray:
        """Calculate the original (unextended) Jacobian of the problem"""
        assert self.group is not None
        # disable the null-space equations
        self.__disabled = True
        Gu = self.group.jacobian(u)
        if isinstance(Gu, sp.spmatrix):
            Gu = Gu.toarray()
        self.__disabled = False
        # reference to the indices of the own unknowns
        self_idx = self.group.idx[self]
        # remove those columns/rows of the Jacobian that belong to self,
        # so we are left with the original (unextended) Jacobian
        return np.delete(np.delete(Gu, self_idx, axis=0), self_idx, axis=1)

    def actions_before_evaluation(self, u: np.ndarray) -> None:
        # TODO: these methods are currently not called from anywhere in the code!
        # write the free parameter back from the given unknowns
        param_obj, param_name = tuple(self.free_parameter)
        setattr(param_obj, param_name, u[-1])

    def actions_after_newton_solve(self) -> None:
        # TODO: these methods are currently not called from anywhere in the code!
        # write the free parameter back from the unknowns
        param_obj, param_name = tuple(self.free_parameter)
        setattr(param_obj, param_name, self.u[-1])

    def plot(self, ax) -> None:
        # nothing to plot
        pass
