import numpy as np
from bice.time_steppers.runge_kutta import RungeKutta4
from bice.continuation.continuation_steppers import PseudoArclengthContinuation
from .equation import Equation, EquationGroup
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

    # Constructor
    def __init__(self):
        # the equation (or system of equation) that governs the problem
        self.eq = None
        # Time variable
        self.time = 0
        # The time-stepper for integration in time
        self.time_stepper = RungeKutta4(dt=1e-2)
        # The continuation stepper for parameter continuation
        self.continuation_stepper = PseudoArclengthContinuation()
        # The Newton solver for finding roots of equations
        self.newton_solver = NewtonSolver()
        # The eigensolver for eigenvalues and -vectors
        self.eigen_solver = EigenSolver()
        # The settings (tolerances, switches, etc.) are held by this ProblemSettings object
        self.settings = ProblemSettings()
        # The history of the unknown values is accessed and managed with the Problem.history object
        self.history = ProblemHistory(self)
        # The bifurcation diagram of the problem holds all branches and their solutions
        self.bifurcation_diagram = BifurcationDiagram()
        # The continuation parameter is defined by passing an object and the name of the
        # object's attribute that corresponds to the continuation parameter as a tuple
        self.continuation_parameter = None

    # The number of unknowns / degrees of freedom of the problem
    @property
    def ndofs(self):
        return self.eq.ndofs

    # getter for unknowns of the problem
    @property
    def u(self):
        return self.eq.u.ravel()

    # set the unknowns of the problem
    @u.setter
    def u(self, u):
        self.eq.u = u.reshape(self.eq.shape)

    # add an equation to the problem
    def add_equation(self, eq):
        if self.eq is self.list_equations() or self.eq is eq:
            # if the given equation equals self.eq, warn
            print("Error: Equation is already part of the problem!")
        elif isinstance(self.eq, Equation):
            # if there is just a single equation, create a system of equations
            self.eq = EquationGroup([self.eq, eq])
        elif isinstance(self.eq, EquationGroup):
            # if there is a system of equations, add the new equation to it
            self.eq.add_equation(eq)
        elif self.eq is None:
            # else, just assign the given equation
            self.eq = eq
        # TODO: clear history?

    # remove an equation from the problem
    def remove_equation(self, eq):
        if self.eq is eq:
            # if the given equation equals self.eq, remove it
            self.eq = None
        elif isinstance(self.eq, EquationGroup):
            # if there is a group of equations, remove the equation from it
            self.eq.remove_equation(eq)
        else:
            # else, eq could not be removed, warn
            print("Equation was not removed, since it is not part of the problem!")
        # TODO: clear history?

    # list all equations that are part of the problem
    def list_equations(self):
        if isinstance(self.eq, Equation):
            return [self.eq]
        if isinstance(self.eq, EquationGroup):
            return self.eq.list_equations()
        return []

    # Calculate the right-hand side of the system 0 = rhs(u)
    @profile
    def rhs(self, u):
        # adjust the shape and return the rhs of the (system of) equations
        return self.eq.rhs(u.reshape(self.eq.shape)).ravel()

    # Calculate the Jacobian of the system J = d rhs(u) / du for the unknowns u
    @profile
    def jacobian(self, u):
        # adjust the shape and return the Jacobian of the (system of) equations
        return self.eq.jacobian(u.reshape(self.eq.shape))

    # The mass matrix determines the linear relation of the rhs to the temporal derivatives:
    # M * du/dt = rhs(u)
    @profile
    def mass_matrix(self):
        # return the mass matrix of the (system of) equations
        return self.eq.mass_matrix()

    # Solve the system rhs(u) = 0 for u with Newton's method
    @profile
    def newton_solve(self):
        self.u = self.newton_solver.solve(self.rhs, self.u, self.jacobian)

    # Calculate the eigenvalues and eigenvectors of the Jacobian
    # The method will only calculate as many eigenvalues as requested with self.settings.neigs
    @profile
    def solve_eigenproblem(self):
        return self.eigen_solver.solve(
            self.jacobian(self.u), self.mass_matrix(), k=self.settings.neigs)

    # Integrate in time with the assigned time-stepper
    @profile
    def time_step(self):
        # update the history with the current state
        self.history.update(history_type="time")
        # perform timestep according to current scheme
        self.time_stepper.step(self)

    # Perform a parameter continuation step
    @profile
    def continuation_step(self):
        # update the history with the current state
        self.history.update(history_type="continuation")
        # perform the step with a continuation stepper
        self.continuation_stepper.step(self)
        # get the current branch in the bifurcation diagram
        branch = self.bifurcation_diagram.current_branch()
        # add the solution to the branch
        sol = Solution(self)
        branch.add_solution_point(sol)
        # if desired, solve the eigenproblem
        if self.settings.neigs > 0:
            # solve the eigenproblem
            eigenvalues, _ = self.solve_eigenproblem()
            # count number of positive eigenvalues
            sol.nunstable_eigenvalues = len([ev for ev in np.real(
                eigenvalues) if ev > self.settings.eigval_zero_tolerance])
        # optionally locate bifurcations
        if self.settings.always_locate_bifurcations and sol.is_bifurcation():
            # try to locate the exact bifurcation point
            converged = self.locate_bifurcation()
            if converged:
                # remove the point that we previously thought was the bifurcation
                branch.remove_solution_point(sol)
                # add the new solution point
                new_sol = Solution(self)
                branch.add_solution_point(new_sol)
                # adapt the number of unstable eigenvalues from the point that
                # overshot the bifurcation
                new_sol.nunstable_eigenvalues = sol.nunstable_eigenvalues
                # TODO: store bifurcation points separately?
                # TODO: add the original solution point back to the branch?

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

    # locate the closest bifurcation using bisection method
    # (finds point where the real part of the eigenvalue closest to zero vanishes)
    # ev_index: optional index of the eigenvalue that corresponds to the bifurcation
    # tolerance: threshold at which the value is considered zero
    # returns True (False) if the location converged (or not)
    def locate_bifurcation(self, ev_index=None, tolerance=1e-5):
        # backup the initial state
        u_old = self.u
        p_old = self.get_continuation_parameter()
        # backup stepsize
        ds = self.continuation_stepper.ds
        # solve the eigenproblem
        # TODO: or recover them from self.eigen_solver.latest_eigenvalues?
        eigenvalues, _ = self.solve_eigenproblem()
        # get the eigenvalue that corresponds to the bifurcation
        # (the one with the smallest abolute real part)
        if ev_index is None:
            ev_index = np.argsort(np.abs(eigenvalues.real))[0]
        # TODO: location sometimes has troubles, when there is more than one null-eigenvalue
        # store the eigenvalue and its sign
        ev = eigenvalues[ev_index]
        sgn = np.sign(ev.real)
        # bisection interval and current position
        # TODO: it can happen that the bifurcation is at pos 1.001, then we will not find it!
        #       we somehow need to check on a broader interval first or known the sign of the
        #       eigenvalue at the limits of the interval
        intvl = (-1, 1)  # in multiples of step size
        pos = 1
        # bisection method loop
        while abs(ev.real) > tolerance and intvl[1] - intvl[0] > 1e-4:
            if self.settings.verbose:
                self.log("Bisection: [{:.6f} {:.6f}], Re: {:e}".format(
                    *intvl, ev.real))
            # new middle point
            pos_old = pos
            pos = (intvl[0] + intvl[1]) / 2
            # perform the continuation step to new center point
            self.continuation_stepper.ds = ds * (pos - pos_old)
            try:
                # Note that we do not update the history, so the tangent remains unchanged
                # TODO: instead of continuation, we could also update (u, p) and do newton_solve()
                #       may be more stable
                self.continuation_stepper.step(self)
            except np.linalg.LinAlgError as err:
                print("Warning: error while trying to locate a bifurcation point:")
                print(err)
                break
            # solve the eigenproblem and get the new eigenvalue
            eigenvalues, _ = self.solve_eigenproblem()
            ev = eigenvalues[ev_index]
            # check the sign of the eigenvalue and adapt the interval
            intvl = (pos, intvl[1]) if ev.real * sgn < 0 else (intvl[0], pos)
        # restore the original stepsize
        self.continuation_stepper.ds = ds
        # if not converged, restore the initial state
        if abs(ev.real) > tolerance * 100:
            self.u = u_old
            self.set_continuation_parameter(p_old)
            print("Warning: Failed to converge onto bifurcation point.")
            self.log("Corresponding eigenvalue:", ev)
            return False
        # if converged, return True
        return True

    # locate the bifurcation of the given eigenvector
    def locate_bifurcation_using_constraint(self, eigenvector):
        # TODO: does not yet work!
        # make sure it is real, if self.u is real
        if not np.iscomplexobj(self.u):
            eigenvector = eigenvector.real
        # create the bifurcation constraint and add it to the problem
        from bice.continuation import BifurcationConstraint
        bifurcation_constraint = BifurcationConstraint(
            eigenvector, self.continuation_parameter)
        self.add_equation(bifurcation_constraint)
        # perform a newton solve
        self.newton_solve()
        # remove the constraint again
        self.remove_equation(bifurcation_constraint)

    # attempt to switch branches in a bifurcation
    def switch_branch(self, ev_index=None, amplitude=1e-3):
        # try to converge onto a bifurcation nearby
        converged = self.locate_bifurcation(ev_index)
        if not converged:
            # TODO: raise error?
            print(
                "Failed to converge onto a bifurcation point! Branch switching aborted.")
            return
        # recover eigenvalues and -vectors from the eigensolver
        eigenvalues = self.eigen_solver.latest_eigenvalues
        eigenvectors = self.eigen_solver.latest_eigenvectors
        # find the eigenvalue that corresponds to the bifurcation
        # (the one with the smallest abolute real part)
        if ev_index is None:
            ev_index = np.argsort(np.abs(eigenvalues.real))[0]
        self.log(
            "Attempting to switch branch with eigenvector #{:d}".format(ev_index))
        # get the eigenvector that corresponds to the bifurcation
        eigenvector = eigenvectors[ev_index]
        if not np.iscomplexobj(self.u):
            eigenvector = eigenvector.real
        # perturb unknowns in direction of eigenvector
        self.u = self.u + amplitude * np.linalg.norm(self.u) * eigenvector
        # self.new_branch()

    # create a new branch in the bifurcation diagram and prepare for a new continuation
    def new_branch(self):
        # create a new branch in the bifurcation diagram
        self.bifurcation_diagram.new_branch()
        # reset the settings and storage of the continuation stepper
        self.continuation_stepper.factory_reset()
        # clear the history of unknowns because it would otherwise be invalid
        self.history.clear()

    # the default norm of the solution, used for bifurcation diagrams
    def norm(self):
        # TODO: @simon: if we want to calculate more than one measure,
        #       we could just return an array here, and do the choosing what
        #       to plot in the problem-specific plot function, right?
        # defaults to the L2-norm of the unknowns
        return np.linalg.norm(self.u)

    # save the current solution to disk
    def save(self, filename):
        np.savetxt(filename, self.u)

    # load the current solution from disk
    def load(self, filename):
        self.u = np.loadtxt(filename, dtype=self.u.dtype)

    # adapt the problem/equations to the solution (e.g. by mesh refinement)
    def adapt(self):
        for eq in self.list_equations():
            eq.adapt()

    # print()-wrapper for log messages
    # log messages are printed only if verbosity is switched on
    def log(self, *args, **kwargs):
        if self.settings.verbose:
            print(*args, **kwargs)
        else:
            print(self.settings.verbose)

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
            for eq in self.list_equations():
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
            if self.eigen_solver.latest_eigenvalues is not None:
                ev_re = np.real(self.eigen_solver.latest_eigenvalues)
                ev_re_n = np.ma.masked_where(
                    ev_re > self.settings.eigval_zero_tolerance, ev_re)
                ev_re_p = np.ma.masked_where(
                    ev_re <= self.settings.eigval_zero_tolerance, ev_re)
                eigval_ax.plot(ev_re_n, "o", color="C0", label="Re < 0")
                eigval_ax.plot(ev_re_p, "o", color="C1", label="Re > 0")
                eigval_ax.axhline(0, color="gray")
                eigval_ax.legend()
                eigval_ax.set_ylabel("eigenvalues")
            if eigvec_ax is not None:
                # clear the axes
                eigvec_ax.clear()
                # map the eigenvectors onto the equations and plot them
                if self.eigen_solver.latest_eigenvectors is not None:
                    ev = self.eigen_solver.latest_eigenvectors[0]
                    # backup the unknowns
                    u_old = self.u.copy()
                    # overwrite the unknowns with the eigenvalues (or their real part only)
                    if not np.iscomplexobj(self.u):
                        self.u = ev.real
                    else:
                        self.u = ev
                    # the equation's own plotting method will know best how to plot it
                    for eq in self.list_equations():
                        eq.plot(eigvec_ax)
                        # adjust the y-label, TODO: do this only for 1d-equations
                        eigvec_ax.set_ylabel("first eigenvector")
                    # reassign the correct unknowns to the problem
                    self.u = u_old


class ProblemHistory():
    """
    This class manages the history of the unknowns and the time /
    continuation parameter of a given problem.
    The history is needed for implicit time-stepping schemes or for
    the calculation of tangents during parameter continuation.
    Note that this class does not actually store the history of the unknowns,
    which is rather found in the equations itself, in order to support adaption.
    """

    def __init__(self, problem):
        # store reference to the problem
        self.problem = problem
        # maximum length of the history
        self.max_length = 4
        # what 'type' of history is currently stored? options are:
        #  - "time" for a time-stepping history
        #  - "continuation" for a history of continuation steps
        self.type = None
        # storage for the values of the time or the continuation parameter
        self.__t = []
        # storage for the values of the stepsize
        self.__dt = []

    # update the history with the current unknowns of the problem
    def update(self, history_type=None):
        # make sure that the history is of correct type, do not mix different types
        if self.type != history_type:
            # if it is of a different type, clear the history first and assign new type
            self.clear()
            self.type = history_type
        # check the minimum length of the history in each equation
        eq_hist_length = min([len(eq.u_history)
                              for eq in self.problem.list_equations()])
        # make sure that the equation's history length matches the parameter history's length
        self.__t = self.__t[:eq_hist_length]
        # update the history of unknowns in every equation
        for eq in self.problem.list_equations():
            eq.u_history = [eq.u.copy()] + eq.u_history[:self.max_length-1]
        # add the value of the time / continuation parameter and step size to the history
        if self.type == "time":
            val = self.problem.time
            dval = self.problem.time_stepper.dt
        elif self.type == "continuation":
            val = self.problem.get_continuation_parameter()
            dval = self.problem.continuation_stepper.ds
        else:
            val = None
            dval = None
        self.__t = [val] + self.__t[:self.max_length-1]
        self.__dt = [dval] + self.__dt[:self.max_length-1]

    # get the unknowns at some point t in history
    def u(self, t=0):
        # check length of history
        if t >= self.length:
            raise IndexError(
                "Unknowns u[t=-{:d}] requested, but history length is {:d}".format(t, self.length))
        # backup the unknowns
        u_old = self.problem.u
        # set the equation's unknowns from history
        for eq in self.problem.list_equations():
            eq.u = eq.u_history[abs(t)]
        # result is now in the problem's unknowns
        res = self.problem.u
        # reset the unknowns
        self.problem.u = u_old
        # return result
        return res

    # get for the value of the time at some point t in history
    def time(self, t=0):
        # accept negative and positive t
        t = abs(t)
        # check length of history
        if t >= self.length:
            raise IndexError(
                "Unknowns u[t=-{:d}] requested, but history length is {:d}".format(t, self.length))
        # return the value
        return self.__t[t]

    # get for the value of the continuation_parameter at some point t in history
    def continuation_parameter(self, t=0):
        # identical to fetching the time
        return self.time(t)

    # get for the value of the (time / continuation) step size at some point t in history
    def step_size(self, t=0):
        # accept negative and positive t
        t = abs(t)
        # check length of history
        if t >= self.length:
            raise IndexError(
                "Unknowns u[t=-{:d}] requested, but history length is {:d}".format(t, self.length))
        # return the value
        return self.__dt[t]

    # returns the length of the history
    @property
    def length(self):
        return len(self.__t)

    # clears the history
    def clear(self):
        # clear the history type
        self.type = None
        # clear the history of unknowns in each equation
        for eq in self.problem.list_equations():
            eq.u_history = []
        # clear the time / continuation parameter history
        self.__t = []


class ProblemSettings():
    """
    A wrapper class that holds all the settings of a problem.
    """

    def __init__(self):
        # how many eigenvalues should be computed when problem.solve_eigenproblem() is called?
        # TODO: should have a more verbose name
        self.neigs = 20
        # how small does an eigenvalue need to be in order to be counted as 'zero'?
        self.eigval_zero_tolerance = 1e-6
        # should we always try to exactly locate bifurcations when passing one?
        self.always_locate_bifurcations = False
        # should sparse matrices be assumed when solving linear systems?
        self.use_sparse_matrices = True
        # should there be some extra output? useful for debugging
        self.verbose = False