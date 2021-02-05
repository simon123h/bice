import numpy as np
import scipy.sparse as sp
from bice.continuation import ConstraintEquation
from bice import profile


class VolumeConstraint(ConstraintEquation):
    """
    A volume constraint (or mass constraint) assures the conservation of
    the integral of the unknowns of some given equation when solving the system.
    We may even prescribe the target volume (or mass) with a parameter,
    but we don't have to.
    The constraint equation comes with an additional (unknown) Lagrange
    multiplier that can be interpreted as an influx into the system.
    """

    def __init__(self, reference_equation):
        super().__init__(shape=(1,))
        # on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        # this equation brings a single extra degrees of freedom (influx Lagrange multiplier)
        self.u = np.zeros(1)
        # This parameter allows for prescribing a fixed volume (unless it is None)
        self.fixed_volume = None

    def rhs(self, u):
        # generate empty vector of residual contributions
        res = np.zeros((u.size))
        # reference to the indices of the unknowns that we work on
        self_idx = self.group.idx[self]
        eq_idx = self.group.idx[self.ref_eq]
        # split it into the parts that are referenced by the first two variables
        var_ndofs = np.prod(self.ref_eq.shape[1:])
        eq_idx1 = slice(eq_idx.start + 0*var_ndofs,
                        eq_idx.start + 1*var_ndofs)
        eq_idx2 = slice(eq_idx.start + 1*var_ndofs,
                        eq_idx.start + 2*var_ndofs)
        # employ the constraint equation
        # calculate the difference in volumes between current
        # and previous unknowns of the reference equation
        x = self.ref_eq.x[0]
        res[self_idx] = np.array([
            np.trapz(u[eq_idx1] - self.group.u[eq_idx1], x),
        ])
        # res[self_idx] = np.array([
        #     np.trapz(u[eq_idx1] + u[eq_idx2] -
        #              self.group.u[eq_idx1] - self.group.u[eq_idx2], x),
        # ])
        # Add the constraint to the reference equation: unknown influx is the Langrange multiplier
        res[eq_idx1] = u[self_idx][0]
        # res[eq_idx2] = u[self_idx][0]
        return res

    def jacobian(self, u):
        # TODO: implement analytical / semi-analytical Jacobian
        # convert FD Jacobian to sparse matrix
        return sp.csr_matrix(super().jacobian(u))
