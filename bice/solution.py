import numpy as np

"""
This file describes a data structure for Solutions, Branches and BifurcationDiagrams of a Problem.
"""


class Solution:
    """
    Stores the solution of a problem, including the relevant parameters,
    and some information on the solution.
    """

    # static variable counting the total number of Solutions
    solution_count = 0

    def __init__(self, problem):
        # generate solution ID
        self.solution_count += 1
        # unique identifier of the solution
        self.id = self.solution_count
        # reference to the corresponding problem
        self.problem = problem
        # vector of unknowns
        self.u = problem.u.copy()
        # time
        self.t = problem.time
        # value of the continuation parameter
        self.p = problem.get_continuation_parameter()
        # value of the solution norm
        self.norm = problem.norm()
        # eigenvalues
        self.eigenvalues = []
        # eigenvectors
        self.eigenvectors = []
        # optional reference to the corresponding branch
        self.branch = None

    # is the solution point at any type of bifurcation, if so, which?
    def get_bifurcation_type(self):
        # check whether we know the eigenvalues
        if len(self.eigenvalues) == 0:
            # if not, we cannot detect any bifurcation type
            return None
        if self.branch is None:
            # if we're not on a branch, we simply check for eigenvalues, that are zero
            nzero_evs = len(np.where(abs(np.real(self.eigenvalues))
                                     <= self.problem.eigval_zero_tolerance, self.eigenvalues))
        else:
            # if we do know the branch, check whether the eigenvalues changed from the previous step
            # get the previous solution
            prev_sol = self.get_neighboring_solution(-1)
            if prev_sol is None:
                # there is no previous solution, we cannot determine the bifurcation type
                return None
            # check whether any eigenvalues crossed the real axis in between this and the previous step
            npos_eigvals_this = len(
                np.where(np.real(self.eigenvalues) > 0, self.eigenvalues))
            npos_eigvals_prev = len(
                np.where(np.real(prev_sol.eigenvalues) > 0, prev_sol.eigenvalues))
            nzero_evs = abs(npos_eigvals_this - npos_eigvals_prev)
        # return the number of eigenvalues that crossed zero in this step
        if nzero_evs > 0:
            # simply return the number of eigenvalues that crossed zero
            # TODO: better distinction between bifurcation types
            return nzero_evs
        # else, no eigenvalues crossed zero --> no bifurcation
        return None

    # is the solution stable?
    def is_stable(self):
        if len(self.eigenvalues) > 0:
            # if we know the eigenvalues, check for positive eigenvalues
            pos_evs = np.where(abs(np.real(self.eigenvalues))
                               <= self.problem.eigval_zero_tolerance, self.eigenvalues)
            return len(pos_evs) == 0
        else:
            # if we do not know the eigenvalues of this solution, recover the stability from the previous solution in the branch
            if self.branch is None:
                # if we do not know the branch either, we cannot recover the stability
                return None
            # get the previous solution
            prev_sol = self.get_neighboring_solution(-1)
            if prev_sol is None:
                # there is no previous solution, we cannot recover the stability
                return None
            # else, return the stability of the previous solution
            return prev_sol.is_stable()

    # get access to the previous solution in the branch
    def get_neighboring_solution(self, distance):
        # if we don't know the branch, there is no neighboring solutions
        if self.branch is None:
            return None
        else:
            index = self.branch.solutions.index(self) + distance
            # if index out of range, there is no neighbor at requested distance
            if index < 0 or index >= len(self.branch.solutions):
                return None
            # else, return the neighbor
            return self.branch.solutions[index]


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

    # is the current branch empty?
    def is_empty(self):
        return len(self.solutions) == 0

    # add a solution to the branch
    def add_solution_point(self, solution):
        # add solution to list
        self.solutions.append(solution)

    # list of continuation parameter values along the branch
    def parameter_vals(self):
        return [s.p for s in self.solutions]

    # list of solution norm values along the branch
    def norm_vals(self):
        return [s.norm for s in self.solutions]

    # return a list of all bifurcation points on the branch
    def get_bifurcation_points(self):
        return [s for s in self.solutions if s.bifurcation_type() is not None]

    # returns the list of parameters and norms of the stable sections only,
    # using numpy masked arrays
    def get_stable_sections(self):
        condition = [not s.is_stable() for s in self.solutions]
        pvals = np.ma.masked_where(condition, self.parameter_vals())
        nvals = np.ma.masked_where(condition, self.norm_vals())
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
