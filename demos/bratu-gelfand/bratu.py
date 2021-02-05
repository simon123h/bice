r"""
Bratu–Gelfand problem

(in parts stolen from scikit-fem's ex23.py, originally using pacopy)

u'' + \lambda e^u = 0,  0 < x < 1,
with `u(0)=u(1)=0` and where `\lambda > 0` is a parameter.

The resulting bifurcation diagram, matches figure 1.1 (left) of Farrell, Birkisson, & Funke (2015).

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dia_matrix
from skfem import asm, condense, InteriorBasis, MeshLine, ElementLineP1
from skfem.models.poisson import laplace, mass
from bice import Problem, Equation

# figures won't steal window focus if the right backend is chosen
matplotlib.use("Tkagg")


class Bratu1dEquation(Equation):

    def __init__(self, n: int):
        super().__init__()
        # construct FEM basis
        self.basis = InteriorBasis(MeshLine(np.linspace(0, 1, n)),
                                   ElementLineP1())
        # construct FEM operators
        self.lap = asm(laplace, self.basis)
        self.mass = asm(mass, self.basis)
        # boundary node indices
        self.D = self.basis.find_dofs()['all'].nodal['u']
        # initial unknowns
        self.u = np.zeros(self.basis.N)
        # parameter
        self.lmbda = 0

    def rhs(self, u):
        res = self.lap @ u - self.lmbda * self.mass @ np.exp(u)
        res[self.D] = u[self.D]
        return res

    # def jacobian(self, u):
    #     jac = self.lap - self.lmbda * dia_matrix((self.mass @ np.exp(u), 0),
    #                                         self.mass.shape)
    #     # TODO: fix boundary conditions!
    #     rhs = self.rhs(u)
    #     cond = condense(jac, rhs, D=self.D)
    #     jac = cond[0]
    #     return jac


class Bratu1d(Problem):
    def __init__(self, N):
        super().__init__()
        self.bratu = Bratu1dEquation(N)
        self.add_equation(self.bratu)
        self.continuation_parameter = (self.bratu, "lmbda")

    def norm(self):
        eq = self.bratu
        u = eq.u
        return np.sqrt(u.T @ (eq.mass @ u))


# construct the problem
problem = Bratu1d(256)

# some settings
problem.settings.neigs = 0
problem.continuation_stepper.ds = 1

# compute bifurcation diagram
fig, ax = plt.subplots(2, 1)
problem.generate_bifurcation_diagram(norm_lims=(0, 6),
                                     max_recursion=0,
                                     ax=ax,
                                     plotevery=10)


fig.savefig("my_bifdiag.png")
