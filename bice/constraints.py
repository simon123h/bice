from .equation import Equation
import numpy as np


class VolumeConstraint(Equation):
    """
    TODO: add docstring
    """
    # TODO: is this constraint implemented correctly?

    def __init__(self, reference_equation):
        super().__init__()
        # on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        # the constraint equation couples to some other equation of the problem
        self.is_coupled = True
        # this equation brings zero degrees of freedom!
        self.dim = 0
        self.u = np.zeros(self.dim)

    def rhs(self, u):
        # calculate the difference in volumes between current
        # and previous unknowns of the reference equation
        return np.trapz(u[self.ref_eq.idx]-self.ref_eq.u, self.ref_eq.x)

    def mass_matrix(self):
        # couples to no time-derivatives
        return np.zeros(self.problem.dim)


class TranslationConstraint(Equation):
    """
    TODO: add docstring
    """

    def __init__(self, reference_equation):
        # call parent constructor
        super().__init__()
        # on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        # the dimension of this equation is equal to the spatial dimension of the reference eq
        # TODO: fix for higher than 1 dimensions
        dim = 1
        # initialize unknowns (velocity vector) to zero
        self.u = np.zeros(dim)
        # the constraint equation couples to some other equation of the problem
        self.is_coupled = True

    def rhs(self, u):
        # TODO: fix for higher than 1 dimensions
        # set up the vector of the residual contributions
        res = np.zeros(self.problem.dim)
        # define some variables
        eq = self.ref_eq
        eq_u = u[eq.idx]
        eq_u_old = eq.u
        velocity = u[self.idx]
        # add constraint to residuals of reference equation (velocity is the langrange multiplier)
        eq_dudx = np.gradient(eq_u, eq.x)
        res[eq.idx] = velocity * eq_dudx
        # calculate the difference in center of masses between current
        # and previous unknowns of the reference equation
        res[self.idx] = np.dot(eq.x, eq_u-eq_u_old)
        return res

    def mass_matrix(self):
        # couples to no time-derivatives
        return np.zeros(self.problem.dim)
