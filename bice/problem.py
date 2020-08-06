import numpy as np
from .time_steppers import RungeKutta4
from .continuation_steppers import PseudoArclengthContinuation
from .solvers import NewtonSolver, EigenSolver
from .solution import Solution, BifurcationDiagram
from .profiling import profile


class Problem():
    """
    All algebraic problems inherit from the 'Problem' class.
    It is an aggregate of (one or many) governing algebraic equations,
    initial and boundary conditions, constraints, etc. Also, it provides
    all the basic properties, methods and routines for treating the problem,
    e.g., time-stepping, solvers or plain analysis of the solution.
    Custom problems should be implemented as children of this class.
    """

    # Constructor: initialize basic ar
    def __init__(self):
        # the list of governing equations in this problem
        self.eq = []
        # The vector of unknowns (NumPy array)
        self.u = np.array([])
        # Number of constraints
        self.nr_constraints = 0
        # Time variable
        self.time = 0
        # the list of equations that are part of this problem
        self.equations = []
        # The time-stepper for integration in time
        self.time_stepper = RungeKutta4(dt=1e-2)
        # The continuation stepper for parameter continuation
        self.continuation_stepper = PseudoArclengthContinuation()
        # The Newton solver for finding roots of equations
        self.newton_solver = NewtonSolver()
        # The eigensolver for eigenvalues and -vectors
        self.eigen_solver = EigenSolver()
        # The bifurcation diagram of the problem holds all branches and their solutions
        self.bifurcation_diagram = BifurcationDiagram()
        # The continuation parameter is defined by passing an object and the name of the
        # object's attribute that corresponds to the continuation parameter as a tuple
        self.continuation_parameter = None
        # storage for the latest eigenvalues that were calculated
        # TODO: what if these become invalid, e.g., due to manual changes to the unknowns?
        self.latest_eigenvalues = None
        # storage for the latest eigenvectors that were calculated
        self.latest_eigenvectors = None
        # how small does an eigenvalue need to be in order to be counted as 'zero'?
        self.eigval_zero_tolerance = 1e-6
        # how many eigenvalues should be computed when problem.solve_eigenproblem() is called?
        # TODO: should have a more verbose name
        self.neigs = 20

    # The dimension of the system
    @property
    def dim(self):
        return self.u.size

    # add an equation to the problem
    def add_equation(self, eq):
        # check if eq already in self.equations
        if eq in self.equations:
            print("Equation is already part of the problem!")
            return
        # append to list of equations
        self.equations.append(eq)
        # append eq's degrees of freedom to the problem dofs
        self.u = np.append(self.u, eq.u)
        # assign this problem to the equation
        eq.problem = self
        # redo the mapping from equation to problem variables
        self.assign_equation_numbers()
        # sort the equations for efficiency
        self.sort_equations()

    # sort the equations, uncoupled first
    # (for sparsity of Jacobian / efficiency of Newton solver)
    def sort_equations(self):
        # backup the old mapping
        old_idx = {eq: eq.idx for eq in self.equations}
        # sort: uncoupled equations first, coupled equations last
        self.equations = [eq for eq in self.equations if not eq.is_coupled] + \
            [eq for eq in self.equations if eq.is_coupled]
        # redo the mapping from equation to problem variables
        self.assign_equation_numbers()
        # move the unknown values to their new positions
        old_u = self.u.copy()
        for eq in old_idx:
            self.u[eq.idx] = old_u[old_idx[eq]]

    # remove an equation from the problem
    def remove_equation(self, eq):
        # check if eq in self.equations
        if eq not in self.equations:
            return
        # remove from the list of equations
        self.equations.remove(eq)
        # remove the equations association with the problem
        eq.problem = None
        # ...and also the lookup table for the unknowns/equation of the problem
        idx = eq.idx
        eq.idx = None
        # write the associated unknowns back into the equation
        eq.u = self.u[idx]
        # remove eq's degrees of freedom from the problem dofs
        self.u = np.delete(self.u, idx)
        # redo the mapping from equation to problem variables
        self.assign_equation_numbers()

    # create the mapping from equation variables to problem variables, in the sense
    # that problem.u[eq.idx] = eq.u where eq.idx is the mapping
    def assign_equation_numbers(self):
        # counter for the current position in problem.u
        i = 0
        # assign index range for each equation according to their dimension
        for eq in self.equations:
            # unknowns / equations indexing
            # NOTE: It is very important for performance that this is a slice,
            #       not a range or anything else. Slices extract coherent parts
            #       of an array, which goes much much faster than extracting values
            #       from positions given by integer indices.
            eq.idx = slice(i, i+eq.dim)
            # increment counter by dimension
            i += eq.dim

    # Calculate the right-hand side of the system 0 = rhs(u)
    @profile
    def rhs(self, u):
        # if there is only one equation, we can return the rhs directly
        if len(self.equations) == 1:
            eq = self.equations[0]
            return self.equations[0].rhs(u)
        # otherwise, we need to assemble the vector
        res = np.zeros(self.dim)
        # add the residuals of each equation
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                res += eq.rhs(u)
            else:
                # uncoupled equations simply work on their own variables, so we do a mapping
                res[eq.idx] += eq.rhs(u[eq.idx])
        # all residuals assembled, return
        return res

    # Calculate the Jacobian of the system J = d rhs(u) / du for the unknowns u
    @profile
    def jacobian(self, u):
        # if there is only one equation, we can return the matrix directly
        if len(self.equations) == 1:
            return self.equations[0].jacobian(u)
        # otherwise, we need to assemble the matrix
        J = np.zeros((self.dim, self.dim))
        # add the Jacobian of each equation
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                J += eq.jacobian(u)
            else:
                # uncoupled equations simply work on their own variables, so we do a mapping
                J[eq.idx, eq.idx] += eq.jacobian(u[eq.idx])
        # all entries assembled, return
        return J

    # The mass matrix determines the linear relation of the rhs to the temporal derivatives:
    # M * du/dt = rhs(u)
    @profile
    def mass_matrix(self):
        # if there is only one equation, we can return the matrix directly
        if len(self.equations) == 1:
            return self.equations[0].mass_matrix()
        # otherwise, we need to assemble the matrix
        M = np.zeros((self.dim, self.dim))
        # add the entries of each equation
        for eq in self.equations:
            if eq.is_coupled:
                # coupled equations work on the full set of variables
                M += eq.mass_matrix()
            else:
                # uncoupled equations simply work on their own variables, so we do a mapping
                M[eq.idx, eq.idx] += eq.mass_matrix()
        # all entries assembled, return
        return M

    # Solve the system rhs(u) = 0 for u with Newton's method
    @profile
    def newton_solve(self):
        # TODO: check for convergence
        self.u = self.newton_solver.solve(self.rhs, self.u)

    # Calculate the eigenvalues and eigenvectors of the Jacobian
    # The method will only calculate as many eigenvalues as requested with self.neigs
    @profile
    def solve_eigenproblem(self):
        return self.eigen_solver.solve(self.jacobian(self.u), self.mass_matrix(), k=self.neigs)

    # Integrate in time with the assigned time-stepper
    @profile
    def time_step(self):
        # perform timestep according to current scheme
        self.time_stepper.step(self)

    # Perform a parameter continuation step
    @profile
    def continuation_step(self):
        # get the current branch in the bifurcation diagram
        branch = self.bifurcation_diagram.current_branch()
        # if the branch is empty, add initial point
        if branch.is_empty():
            sol = Solution(self)
            branch.add_solution_point(sol)
            # if desired, solve the eigenproblem
            if self.continuation_stepper.always_check_eigenvalues:
                # solve the eigenproblem
                eigenvalues, eigenvectors = self.solve_eigenproblem()
                # count number of positive eigenvalues
                sol.nunstable_eigenvalues = len([ev for ev in np.real(
                    eigenvalues) if ev > self.eigval_zero_tolerance])
        # perform the step with a continuation stepper
        self.continuation_stepper.step(self)
        # add the solution to the branch
        sol = Solution(self)
        branch.add_solution_point(sol)
        # if desired, solve the eigenproblem
        if self.continuation_stepper.always_check_eigenvalues:
            # call eigensolver
            eigenvalues, eigenvectors = self.solve_eigenproblem()
            # count number of positive eigenvalues
            sol.nunstable_eigenvalues = len([ev for ev in np.real(
                eigenvalues) if ev > self.eigval_zero_tolerance])
            # temporarily save eigenvalues and eigenvectors for this step
            # NOTE: this is currently only needed for plotting
            self.latest_eigenvalues = eigenvalues
            self.latest_eigenvectors = eigenvectors
            # TODO: optionally locate bifurcations
            # TODO: maybe do some more postprocessing with hook-methods, that can be overwritten

    # return the value of the continuation parameter
    def get_continuation_parameter(self):
        # if no continuation parameter set, return None
        if self.continuation_parameter is None:
            return None
        # else, get the value using the builtin 'getattr'
        obj, attr_name = tuple(self.continuation_parameter)
        return getattr(obj, attr_name)

    # set the value of the continuation parameter
    def set_continuation_parameter(self, val):
        # if no continuation parameter set, do nothing
        if self.continuation_parameter is None:
            return
        # else, assign the new value using the builtin 'setattr'
        obj, attr_name = tuple(self.continuation_parameter)
        setattr(obj, attr_name, val)

    # this method is called after each newton solve
    def actions_after_newton_solve(self):
        # pass it to the equations
        for eq in self.equations:
            eq.actions_after_newton_solve()

    def locate_bifurcation(self):
        # TODO: implement
        pass

    # create a new branch in the bifurcation diagram and prepare for a new continuation
    def new_branch(self):
        # create a new branch in the bifurcation diagram
        self.bifurcation_diagram.new_branch()
        # reset the settings and storage of the continuation stepper
        self.continuation_stepper.factory_reset()

    # the default norm of the solution, used for bifurcation diagrams
    def norm(self):
        # TODO: @simon: if we want to calculate more than one measure,
        #       we could just return an array here, and do the choosing what
        #       to plot in the problem-specific plot function, right?
        # defaults to the L2-norm of the first equation
        return np.linalg.norm(self.equations[0].u)

    # save the current solution to disk
    def save(self, filename):
        np.savetxt(filename, self.u)

    # load the current solution from disk
    def load(self, filename):
        self.u = np.loadtxt(filename)

    # Plot everything to the given axes.
    # Axes may be given explicitly of as a list of axes, that is then expanded.
    # The plot may include the solution of the equations, the bifurcation diagram,
    # the eigenvalues and the eigenvectors.
    @profile
    def plot(self, sol_ax=None, bifdiag_ax=None, eigvec_ax=None, eigval_ax=None):
        # check if any axes are given
        if all(ax is None for ax in [sol_ax, bifdiag_ax, eigval_ax, eigvec_ax]):
            print("Warning: no axes passed to Problem.plot(<axes>). Plotting nothing.")
        # check if an array of axes was passed
        if isinstance(sol_ax, np.ndarray):
            # flatten the array and pass it to the plot function as arguments
            self.plot(*sol_ax.flatten())
            return
        # plot the solution of the equation(s)
        if sol_ax is not None:
            # clear the axes
            sol_ax.clear()
            # plot all equation's solutions
            for eq in self.equations:
                eq.plot(sol_ax)
        # plot the bifurcation diagram
        if bifdiag_ax is not None:
            # clear the axes
            bifdiag_ax.clear()
            # plot current point
            bifdiag_ax.plot(self.get_continuation_parameter(), self.norm(),
                            "x", label="current point", color="black")
            # plot the rest of the bifurcation diagram
            self.bifurcation_diagram.plot(bifdiag_ax)
        if eigval_ax is not None:
            # clear the axes
            eigval_ax.clear()
            # plot the eigenvalues, if any
            if self.latest_eigenvalues is not None:
                ev_re = np.real(self.latest_eigenvalues)
                ev_re_n = np.ma.masked_where(
                    ev_re > self.eigval_zero_tolerance, ev_re)
                ev_re_p = np.ma.masked_where(
                    ev_re <= self.eigval_zero_tolerance, ev_re)
                eigval_ax.plot(ev_re_n, "o", color="C0", label="Re < 0")
                eigval_ax.plot(ev_re_p, "o", color="C1", label="Re > 0")
                eigval_ax.axhline(0, color="gray")
                eigval_ax.legend()
                eigval_ax.set_ylabel("eigenvalues")
            if eigvec_ax is not None:
                # clear the axes
                eigvec_ax.clear()
                # map the eigenvectors onto the equations and plot them
                if self.latest_eigenvectors is not None:
                    ev = self.latest_eigenvectors[0]
                    # backup the unknowns
                    u_old = self.u.copy()
                    # overwrite the unknowns with the eigenvalues (or their real part only)
                    if not np.iscomplexobj(self.u):
                        self.u = ev.real
                    else:
                        self.u = ev
                    # the equation's own plotting method will know best how to plot it
                    for eq in self.equations:
                        eq.plot(eigvec_ax)
                        # adjust the y-label, TODO: do this only for 1d-equations
                        eigvec_ax.set_ylabel("eigenvector")
                    # reassign the correct unknowns to the problem
                    self.u = u_old
