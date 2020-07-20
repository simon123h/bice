import numpy as np

"""
This file describes a data structure for Solutions, Branches and BifurcationDiagrams of a Problem.
"""

class Solution:
    """
    Stores the solution of a problem, including the relevant parameters,
    and some information on the solution.
    """

    def __init__(self, problem):
        # vector of unknowns
        self.u = problem.u
        # time
        self.t = problem.time
        # value of the continuation parameter
        self.p = problem.get_continuation_parameter()
        # eigenvalues
        self.eigenvalues = []
        # eigenvectors
        self.eigenvectors = []
        # is the solution stable?
        self.stability = None
        # is the solution point at any type of bifurcation, if so, which?
        self.bifurcation_type = None


class Branch:
    """
    A branch is obtained from a parameter continuation and stores a list of solution objects,
    the corresponding value of the continuation parameter(s) and the norm.
    """

    # static variable counting the number of Branch instances
    branch_count = 0

    def __init__(self):
        # generate branch ID
        self.branch_count += 1
        # unique identifier of the branch
        self.id = self.branch_count
        # list of solutions along the branch
        self.solutions = []
        # list of continuation parameter values along the branch
        self.parameter_vals = []
        # list of solution norm values along the branch
        self.norm_vals = []

    # is the current branch empty?
    def is_empty(self):
        return len(self.solutions) == 0

    # add a solution to the branch
    def add_solution_point(self, problem):
        # create solution object and add to list
        solution = Solution(problem)
        self.solutions.append(solution)
        # add values of continuation parameter and norm to list
        self.parameter_vals.append(problem.get_continuation_parameter())
        self.norm_vals.append(problem.norm())
        # return reference to solution
        return solution

    # return a list of all bifurcation points on the branch
    def get_bifurcation_points(self):
        return [s for s in self.solutions if s.bifurcation_type is not None]

    # returns the list of parameters and norms of the stable sections only,
    # using numpy masked arrays
    def get_stable_sections(self):
        condition = [s.stability is not True for s in self.solutions]
        pvals = np.ma.masked_where(condition, self.parameter_vals)
        nvals = np.ma.masked_where(condition, self.norm_vals)
        return pvals, nvals


class BifurcationDiagram:
    """
    Basically just a list of branches...
    TODO: longer description needed?
    """

    def __init__(self):
        # list of branches
        self.branches = []
        # make sure there is at least one branch in the list
        self.new_branch()

    # create a new branch
    def new_branch(self):
        self.branches.append(Branch())

    # return the latest branch in the diagram
    def get_current_branch(self):
        # if there is no branches yet, create one
        if not self.branches:
            self.new_branch()
        # return the latest branch
        return self.branches[-1]

    # return a branch by its ID
    def get_branch_by_ID(self, branch_id):
        for branch in self.branches:
            if branch.id == branch_id:
                return branch
        return None

    # remove a branch from the BifurcationDiagram by its ID
    def remove_branch_by_ID(self, branch_id):
        self.branches = [b for b in self.branches if b.id != branch_id]
