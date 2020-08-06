import numpy as np
from .equation import Equation


class BifurcationConstraint(Equation):
    def __init__(self, phi, free_parameter):
        super().__init__()
        # the constraint equation couples to some other equations of the problem
        self.is_coupled = True
        # copy and normalize the null-eigenvector phi
        phi = phi.copy() / np.linalg.norm(phi)
        # store the reference to the free parameter
        # TODO: rather use the problem's continuation parameter?
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

    def rhs(self, u):
        # if the constraint is disabled, no residuals will be calculated
        if self.__disabled:
            return 0
        # get the value of the current and the previous null-eigenvector phi
        phi = u[self.idx][:-1]
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
        res[self.idx] = np.concatenate((res1, res2))
        return res

    def jacobian(self, u):
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

        # get phi and phi_old
        phi = u[self.idx][:-1]
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
            # NOTE: oookay... here we'd also need N times evaluation of the Jacobian... that's slow as well
            Gu1 = self.original_jacobian(u1)
            # NOTE: the following line is not general, wrt. self.idx
            J[:, i] = np.matmul(Gu1-Gu, phi) / eps
            # .... I'm giving up at this point!

        # lower right sub-block: d(res1)/d(phi) = Gu
        res1_slice = slice(self.idx.start, self.idx.stop-1)
        J[res1_slice, res1_slice] = Gu

        # last column: d(rhs)/d(param), calculate with FD
        f0 = self.problem.rhs(u)
        # deviate the free parameter value
        param_val = u[self.idx][-1]
        u[self.idx][-1] += eps
        # calculate new residuals
        f1 = self.problem.rhs(u)
        # reset the value of the free parameter
        u[self.idx][-1] = param_val
        # add FD parameter derivative to Jacobian
        J[:, self.idx.stop-1] = (f1 - f0) / eps

        # last row: d(res2)/du = ([0]*N, phi_old, 0)
        J[self.idx.stop -
            1] = np.concatenate((np.zeros(phi_old.size), phi_old, np.array([0])))

        return J

    def mass_matrix(self):
        # couples to no time-derivatives
        return 0

    # calculate the original (unextended) Jacobian of the problem
    def original_jacobian(self, u):
        # disable the null-space equations
        self.__disabled = True
        Gu = self.problem.jacobian(u)
        self.__disabled = False
        # remove those columns/rows of the Jacobian that belong to self,
        # so we are left with the original (unextended) Jacobian
        return np.delete(np.delete(Gu, self.idx, axis=0), self.idx, axis=1)

    def actions_before_evaluation(self, u):
        # write the free parameter back from the given unknowns
        param_obj, param_name = tuple(self.free_parameter)
        setattr(param_obj, param_name, u[-1])

    def actions_after_newton_solve(self):
        # write the free parameter back from the unknowns
        param_obj, param_name = tuple(self.free_parameter)
        setattr(param_obj, param_name, self.u[-1])

    def plot(self, ax):
        # nothing to plot
        # TODO: should we maybe plot the eigenvector from here in case of fold point continuation?
        pass
