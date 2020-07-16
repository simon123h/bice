from .time_stepper import Euler, RungeKutta4
from .linear_solver import NewtonSolver
import numpy as np


class Problem():

    def __init__(self, dimension=0):
        # the vector of unknowns
        self.u = np.zeros(dimension)
        self.time = 0
        self.time_stepper = RungeKutta4(dt=1e-2)
        self.linear_solver = NewtonSolver()

    # The dimension of the linear system
    @property
    def dim(self):
        return self.u.size

    # Calculate the right hand side of the linear system 0 = rhs(u)
    def rhs(self, u):
        raise NotImplementedError

    # Calculate the Jacobian of the linear system J = d rhs(u) / du
    def jacobian(self, u, eps=1e-10):
        # default implementation: calculate Jacobian with finite differences
        J = np.zeros([self.dim, self.dim], dtype=np.float)
        for i in range(self.dim):
            f1 = self.rhs(u - eps)
            f2 = self.rhs(u + eps)
            J[:, i] = (f1 - f2) / (2 * eps)
        return J

    # Integrate in time with the assigned time-stepper
    def time_step(self):
        self.time_stepper.step(self)

    # Solve the equation rhs(u) = 0 for u with the assigned linear solver
    def solve(self):
        self.linear_solver.solve(self)

