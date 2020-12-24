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
        Solution.solution_count += 1
        # unique identifier of the solution
        self.id = Solution.solution_count
        # reference to the corresponding problem
        self.problem = problem
        # the dimension / number of unknowns
        self.ndofs = problem.ndofs
        # vector of unknowns
        # TODO: storing each solution may eat up some memory
        #  @simon: do we need to save every solution? we could save some
        #  solutions to the disk instead
        self.u = problem.u.copy()
        # time
        self.t = problem.time
        # value of the continuation parameter
        self.p = problem.get_continuation_parameter()
        # value of the solution norm
        self.norm = problem.norm()
        # number of true positive eigenvalues
        self.nunstable_eigenvalues = None
        # optional reference to the corresponding branch
        self.branch = None

    # how many eigenvalues have crossed the imaginary axis with this solution?
    @property
    def neigenvalues_crossed(self):
        # if we do not know the number of unstable eigenvalues, we have no result
        if self.nunstable_eigenvalues is None:
            return None
        # else, compare with the nearest previous neighbor that has info on eigenvalues
        # get branch points with eigenvalue info
        bps = [s for s in self.branch.solutions if s.nunstable_eigenvalues is not None]
        # find index of previous solution
        index = bps.index(self) - 1
        if index < 0:
            # if there is no previous solution with info on eigenvalues, we have no result
            return None
        # return the difference in unstable eigenvalues to the previous solution
        return self.nunstable_eigenvalues - bps[index].nunstable_eigenvalues

    # is the solution stable?
    def is_stable(self):
        # if we don't know the number of eigenvalues, return None
        if self.nunstable_eigenvalues is None:
            return None
        # if there is any true positive eigenvalues, the solution is not stable
        if self.nunstable_eigenvalues > 0:
            return False
        # otherwise, the solution is considered to be stable (or metastable at least)
        return True

    # is the solution point a bifurcation?
    def is_bifurcation(self):
        return self.neigenvalues_crossed not in [None, 0]

    # get access to the previous solution in the branch
    def get_neighboring_solution(self, distance):
        # if we don't know the branch, there is no neighboring solutions
        if self.branch is None:
            return None
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
        Branch.branch_count += 1
        # unique identifier of the branch
        self.id = Branch.branch_count
        # list of solutions along the branch
        self.solutions = []

    # is the current branch empty?
    def is_empty(self):
        return len(self.solutions) == 0

    # add a solution to the branch
    def add_solution_point(self, solution):
        # assign this branch as the solution's branch
        solution.branch = self
        # add solution to list
        self.solutions.append(solution)

    # remove a solution from the branch
    def remove_solution_point(self, solution):
        self.solutions.remove(solution)

    # list of continuation parameter values along the branch
    def parameter_vals(self):
        return [s.p for s in self.solutions]

    # list of solution norm values along the branch
    def norm_vals(self):
        return [s.norm for s in self.solutions]

    # list all bifurcation points on the branch
    def bifurcations(self):
        return [s for s in self.solutions if s.is_bifurcation()]

    # returns the list of parameters and norms of the branch
    # optional argument only (str) may restrict the data to:
    #    - only="stable": stable parts only
    #    - only="unstable": unstable parts only
    #    - only="bifurcations": bifurcations only
    def data(self, only=None):
        if only is None:
            condition = False
        elif only == "stable":
            condition = [not s.is_stable() for s in self.solutions]
        elif only == "unstable":
            condition = [s.is_stable() for s in self.solutions]
        elif only == "bifurcations":
            condition = [not s.is_bifurcation() for s in self.solutions]
        # mask lists where condition is met and return
        pvals = np.ma.masked_where(condition, self.parameter_vals())
        nvals = np.ma.masked_where(condition, self.norm_vals())
        return (pvals, nvals)


class BifurcationDiagram:
    """
    Basically just a list of branches and methods to act upon.
    Also: a fancy plotting method.
    """

    def __init__(self):
        # list of branches
        self.branches = []
        # make sure there is at least one branch in the list
        self.new_branch()

    # create a new branch
    def new_branch(self):
        branch = Branch()
        self.branches.append(branch)
        return branch

    # return the latest branch in the diagram
    def current_branch(self):
        # if there is no branches yet, create one
        if not self.branches:
            self.new_branch()
        # return the latest branch
        return self.branches[-1]

    # return the latest solution in the diagram
    def current_solution(self):
        return self.current_branch().solutions[-1]

    # return a branch by its ID
    def get_branch_by_ID(self, branch_id):
        for branch in self.branches:
            if branch.id == branch_id:
                return branch
        return None

    # remove a branch from the BifurcationDiagram by its ID
    def remove_branch_by_ID(self, branch_id):
        self.branches = [b for b in self.branches if b.id != branch_id]

    # plot the bifurcation diagram
    def plot(self, ax):
        # plot every branch separately
        for branch in self.branches:
            p, norm = branch.data()
            # ax.plot(p, norm, "o", color="C0")
            ax.plot(p, norm, "--", color="C0")
            p, norm = branch.data(only="stable")
            ax.plot(p, norm, color="C0")
            p, norm = branch.data(only="bifurcations")
            ax.plot(p, norm, "*", color="C2")
            # annotate bifurcations with +/- signs corresponding to their null-eigenvalues
            bifs = [
                s for s in branch.solutions if s.neigenvalues_crossed not in [None, 0]]
            for bif in bifs:
                s = bif.neigenvalues_crossed
                s = "+"*s if s > 0 else "-"*(-s)
                ax.annotate(" "+s, (bif.p, bif.norm))
        ax.plot(np.nan, np.nan, "*", color="C2", label="bifurcations")
        ax.set_xlabel("continuation parameter")
        ax.set_ylabel("norm")
        ax.legend()
