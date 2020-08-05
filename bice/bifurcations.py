import numpy as np
from .equation import Equation


class BifurcationExtension(Equation):
    def __init__(self, reference_equation):
        super().__init__()
        # the equation, whose eigenvalue is kept
        self.ref_eq = reference_equation
        # this equation is not coupled to the rest
        self.is_coupled = True
        # this equation brings the degrees of freedom of the reference equation
        # and a +1
        self.u = np.zeros(self.ref_eq.dim + 1)

    def rhs(self, u):
        # check if jacobian of the reference equation has already been calculated,
        # if not, calculate it
        try:
            if self.ref_eq.Gu is None:
                self.ref_eq.Gu = self.ref_eq.jacobian(u[self.ref_eq.idx])
        # if it doesn't exist, calculate it, and store it in reference equation
        except AttributeError:
            self.ref_eq.Gu = self.ref_eq.jacobian(u[self.ref_eq.idx])
        res = np.zeros((u.size))
        u1 = u[self.idx][:-1]
        u1_old = self.u[self.idx][:-1]
        res1 = np.matmul(self.ref_eq.Gu, u1)
        res2 = np.matmul(u1.T, u1_old) - 1.
        res[self.idx] = np.append(res1, res2)

    def mass_matrix(self):
        # couples to no time-derivatives
        return np.zeros(self.problem.dim)

    def plot(self, ax):
        # nothing to plot
        # TODO: should we maybe plot the eigenvector from here in case of fooldpointcontinuation?
        pass
    
