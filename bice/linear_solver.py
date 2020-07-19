import numpy as np
import scipy.optimize

class LinearSolver():

    def solve(self, f, u):
        raise NotImplementedError

    
class NewtonSolver(LinearSolver):

    def solve(self, f, u):
        return scipy.optimize.newton(f, u)
        
