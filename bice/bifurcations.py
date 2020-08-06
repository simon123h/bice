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
        phi_old = self.u[self.idx][:-1]
        # apply the given value of the free parameter
        param_obj, param_name = tuple(self.free_parameter)
        setattr(param_obj, param_name, u[self.idx][-1])
        # calculate the original Jacobian of the problem by disabling self
        self.__disabled = True
        Gu = self.problem.jacobian(u)
        self.__disabled = False
        # remove those columns/rows of the Jacobian that belong to self,
        # so we are left with the original (unextended) Jacobian
        Gu = np.delete(np.delete(Gu, self.idx, axis=0), self.idx, axis=1)
        # check for mismatch in dimension
        if Gu.shape != (phi.size, phi.size):
            raise Exception("It seems that the dimension of the problem does not "
                            "match the dimension of the null-eigenvector phi in "
                            "the BifurcationConstraint. Did your problem change "
                            "since you imposed the constraint?")
        # reset the value of the free parameter
        setattr(param_obj, param_name, self.u[self.idx][-1])
        # calculate the residuals
        res = np.zeros((u.size))
        res1 = np.matmul(Gu, phi)
        res2 = np.dot(phi, phi_old) - 1
        res[self.idx] = np.concatenate((res1, res2))
        return res

    def jacobian(self, u):
        # if the constraint is disabled, no Jacobian will be calculated
        if self.__disabled:
            return 0
        # apply the given value of the free parameter
        param_obj, param_name = tuple(self.free_parameter)
        setattr(param_obj, param_name, u[self.idx][-1])
        # pass Jacobian calculation to the FD method of the parent Equation class
        J = Equation.jacobian(self, u)
        # as the default implementation does not respect that the free parameter is
        # an unknown, add the corresponding parameter derivative to the Jacobian.
        # the constraint's own rhs does respect it, though. --> disable it
        self.__disabled = True
        f0 = self.problem.rhs(u)
        # deviate the free parameter value
        eps = 1e-10
        setattr(param_obj, param_name, u[self.idx][-1] + eps)
        # calculate new residuals
        f1 = self.problem.rhs(u)
        # add FD parameter derivative to Jacobian
        J = J.T
        J[self.idx.stop-1] += (f1 - f0) / eps
        # re-enable the constraint
        self.__disabled = False
        # reset the value of the free parameter and return the Jacobian
        setattr(param_obj, param_name, self.u[self.idx][-1])
        return J.T

    def mass_matrix(self):
        # couples to no time-derivatives
        return 0

    def plot(self, ax):
        # nothing to plot
        # TODO: should we maybe plot the eigenvector from here in case of fold point continuation?
        pass
