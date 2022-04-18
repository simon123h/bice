
import numpy as np
import scipy.sparse as sp

"""
A deflation operator M for deflated continuation.
Adds singularities to the equation at given solutions u_i
0 = F(u) --> 0 = M(u) * F(u)
with
M(u) = product_i <u_i - u, u_i - u>^-p + shift
The parameters are:
  p: some exponent to the norm <u, v>
  shift: some constant added shift parameter for numerical stability
"""


class DeflationOperator:

    def __init__(self) -> None:
        #: the order of the norm that will be used for the deflation operator
        self.p = 2
        #: small constant in the deflation operator, for numerical stability
        self.shift = 0.5
        #: list of solutions, that will be suppressed by the deflation operator
        self.solutions = []

    def operator(self, u: np.ndarray):
        """obtain the value of the deflation operator for given u"""
        return np.prod([np.dot(u_i - u, u_i - u)**-self.p
                        for u_i in self.solutions]) + self.shift

    def D_operator(self, u: np.ndarray):
        """Jacobian of deflation operator for given u"""
        op = self.operator(u)
        return self.p * op * 2 * \
            np.sum([(uk - u) / np.dot(uk - u, uk - u)
                    for uk in self.solutions], axis=0)

    def deflated_rhs(self, rhs):
        """deflate the rhs of some equation"""
        def new_rhs(u):
            # multiply rhs with deflation operator
            return self.operator(u) * rhs(u)
        # return the function object
        return new_rhs

    def deflated_jacobian(self, rhs, jacobian):
        """generate Jacobian of deflated rhs of some equation or problem"""
        def new_jac(u):
            # obtain operator and operator derivative
            op = self.operator(u)
            D_op = self.D_operator(u)
            # calculate derivative d/du
            return sp.diags(D_op * rhs(u)) + op * jacobian(u)
        # return the function object
        return new_jac

    def add_solution(self, u: np.ndarray) -> None:
        """add a solution to the list of solutions used for deflation"""
        self.solutions.append(u)

    def remove_solution(self, u: np.ndarray) -> None:
        """remove a solution from the list of solutions used for deflation"""
        self.solutions.remove(u)

    def clear_solutions(self) -> None:
        """clear the list of solutions used for deflation"""
        self.solutions = []
