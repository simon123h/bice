import numpy as np
import scipy.optimize

class LinearSolver():
    def solve(self, problem):
        raise NotImplementedError

    
class NewtonSolver(LinearSolver):
    def solve(self, problem):
        return scipy.optimize.newton(problem.rhs, problem.u)
        
