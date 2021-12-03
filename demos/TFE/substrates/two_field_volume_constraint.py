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
        super().__init__(shape=(2,))
        # on which equation/unknowns should the constraint be imposed?
        self.ref_eq = reference_equation
        # this equation brings two extra degrees of freedom (influx Lagrange multipliers)
        self.u = np.zeros(2)
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
        # employ the constraint equations
        # calculate the difference in volumes between current
        # and previous unknowns of the reference equation
        x = self.ref_eq.x[0]
        res[self_idx] = np.array([
            np.trapz(u[eq_idx1] - self.group.u[eq_idx1], x),
            np.trapz(u[eq_idx2] - self.group.u[eq_idx2], x)
        ])
        # Add the constraint to the reference equation: unknown influx is the Langrange multiplier
        res[eq_idx1] = u[self_idx][0]
        res[eq_idx2] = u[self_idx][1]
        return res

    @profile
    def jacobian2(self, u):
        # reference to the equation, shape and indices of the unknowns that we work on
        eq = self.ref_eq
        eq_idx = self.group.idx[eq]
        self_idx = self.group.idx[self]
        # split it into the parts that are referenced by the first two variables
        var_ndofs = np.prod(self.ref_eq.shape[1:])
        eq_idx1 = slice(eq_idx.start + 0*var_ndofs,
                        eq_idx.start + 1*var_ndofs)
        eq_idx2 = slice(eq_idx.start + 1*var_ndofs,
                        eq_idx.start + 2*var_ndofs)
        # obtain the unknowns
        eq_u = u[eq_idx]
        N = eq.ndofs
        # eq_u_old = self.group.u[eq_idx]
        # influx = u[self_idx][0]
        # contribution of d(eq) / du
        deq_du = sp.csr_matrix((N, N))
        # contribution of d(eq) / dinflux
        deq_dv = np.ones((N, 1))
        # contribution of d(constraint) / du
        if self.fixed_volume is None:
            dcnstr_du = np.ones((1, N)) / float(N)
        else:
            dcnstr_du = np.trapz(np.eye(N), eq.x[0]).reshape((1, N))
        # contribution of d(constraint) / dinflux
        dcnstr_dv = sp.csr_matrix((1, 1))
        # stack everything together and return
        print("start")
        J = sp.bmat([[deq_du, deq_dv], [dcnstr_du, dcnstr_dv]])
        print(N)
        print(J.shape)
        print("end")
        return J

    def jacobian(self, u):
        # TODO: implement analytical / semi-analytical Jacobian
        # convert FD Jacobian to sparse matrix
        return sp.csr_matrix(super().jacobian(u))
