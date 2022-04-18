import numpy as np
from bice.time_steppers.runge_kutta import RungeKutta4
from bice.continuation import PseudoArclengthContinuation
import matplotlib.pyplot as plt
from .equation import Equation, EquationGroup, EqType
from .solvers import NewtonKrylovSolver, EigenSolver
from .solution import Solution, BifurcationDiagram
from .profiling import profile
from typing import Union, Optional
from .types import Matrix


class Problem():
    """
    All algebraic problems inherit from the 'Problem' class.
    It is an aggregate of (one or many) governing algebraic equations,
    initial and boundary conditions, constraints, etc. Also, it provides
    all the basic properties, methods and routines for treating the problem,
    e.g., time-stepping, solvers or plain analysis of the solution.
    Custom problems should be implemented as children of this class.
    """

    eq: Union[None, Equation, EquationGroup]

    # Constructor
    def __init__(self) -> None:
        #: the equation (or system of equation) that governs the problem
        self.eq = None
        #: Time variable
        self.time = 0
        #: The time-stepper for integration in time
        self.time_stepper = RungeKutta4(dt=1e-2)
        #: The continuation stepper for parameter continuation
        self.continuation_stepper = PseudoArclengthContinuation()
        #: The Newton solver for finding roots of equations
        self.newton_solver = NewtonKrylovSolver()
        #: The eigensolver for eigenvalues and -vectors
        self.eigen_solver = EigenSolver()
        #: The settings (tolerances, switches, etc.) are held by this ProblemSettings object
        self.settings = ProblemSettings()
        #: The history of the unknown values is accessed and managed with the Problem.history object
        self.history = ProblemHistory(self)
        #: The bifurcation diagram of the problem holds all branches and their solutions
        self.bifurcation_diagram = BifurcationDiagram()
        #: The continuation parameter is defined by passing an object and the name of the
        #: object's attribute that corresponds to the continuation parameter as a tuple
        self.continuation_parameter = (None, "")

    @property
    def ndofs(self) -> int:
        """The number of unknowns / degrees of freedom of the problem"""
        if self.eq is None:
            return 0
        return self.eq.ndofs

    @property
    def u(self) -> np.ndarray:
        """getter for unknowns of the problem"""
        if self.eq is None:
            return np.array([])
        return self.eq.u.ravel()

    @u.setter
    def u(self, u) -> None:
        """set the unknowns of the problem"""
        assert self.eq is not None
        self.eq.u = u.reshape(self.eq.shape)

    def add_equation(self, eq: EqType) -> None:
        """add an equation to the problem"""
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

    def remove_equation(self, eq: EqType) -> None:
        """remove an equation from the problem"""
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

    def list_equations(self) -> list[Equation]:
        """list all equations that are part of the problem"""
        if isinstance(self.eq, Equation):
            return [self.eq]
        if isinstance(self.eq, EquationGroup):
            return self.eq.list_equations()
        return []

    @profile
    def rhs(self, u: np.ndarray) -> np.ndarray:
        """Calculate the right-hand side of the system 0 = rhs(u)"""
        assert self.eq is not None
        # adjust the shape and return the rhs of the (system of) equations
        return self.eq.rhs(u.reshape(self.eq.shape)).ravel()

    @profile
    def jacobian(self, u) -> Matrix:
        """Calculate the Jacobian of the system J = d rhs(u) / du for the unknowns u"""
        assert self.eq is not None
        # adjust the shape and return the Jacobian of the (system of) equations
        return self.eq.jacobian(u.reshape(self.eq.shape))

    @profile
    def mass_matrix(self) -> Matrix:
        """
        The mass matrix determines the linear relation of the rhs to the temporal derivatives:
        M * du/dt = rhs(u)
        """
        assert self.eq is not None
        # return the mass matrix of the (system of) equations
        return self.eq.mass_matrix()

    @profile
    def newton_solve(self) -> None:
        """Solve the system rhs(u) = 0 for u with Newton's method"""
        self.u = self.newton_solver.solve(self.rhs, self.u, self.jacobian)

    @profile
    def solve_eigenproblem(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the eigenvalues and eigenvectors of the Jacobian
        The method will only calculate as many eigenvalues as requested with self.settings.neigs
        """
        return self.eigen_solver.solve(
            self.jacobian(self.u), self.mass_matrix(), k=self.settings.neigs)

    @profile
    def time_step(self) -> None:
        """Integrate in time with the assigned time-stepper"""
        # update the history with the current state
        self.history.update(history_type="time")
        # perform timestep according to current scheme
        self.time_stepper.step(self)

    @profile
    def continuation_step(self) -> None:
        """Perform a parameter continuation step"""
        # update the history with the current state
        self.history.update(history_type="continuation")
        # perform the step with a continuation stepper
        self.continuation_stepper.step(self)
        # make sure the bifurcation diagram is up to date
        # TODO: this could be encapsulated within the BifurcationDiagram class or somewhere else
        if self.bifurcation_diagram.parameter_name == "":
            self.bifurcation_diagram.parameter_name = self.continuation_parameter[1]
        elif self.bifurcation_diagram.parameter_name != self.continuation_parameter[1]:
            print("Warning: continuation parameter changed from"
                  "{self.bifurcation_diagram.parameter_name:s} to {self.continuation_parameter[1]:s}!"
                  "Will generate a new bifurcation diagram!")
            self.bifurcation_diagram = BifurcationDiagram()
        # get the current branch in the bifurcation diagram
        branch = self.bifurcation_diagram.active_branch
        # add the solution to the branch
        sol = Solution(self)
        branch.add_solution_point(sol)
        # if desired, solve the eigenproblem
        if self.settings.neigs is None or self.settings.neigs > 0:
            # solve the eigenproblem
            eigenvalues, _ = self.solve_eigenproblem()
            # count number of positive eigenvalues
            tol = self.settings.eigval_zero_tolerance
            sol.nunstable_eigenvalues = len(
                [ev for ev in eigenvalues if np.real(ev) > tol])
            sol.nunstable_imaginary_eigenvalues = len(
                [ev for ev in eigenvalues if np.real(ev) > tol and abs(np.imag(ev)) > tol])
        # optionally locate bifurcations
        if self.settings.always_locate_bifurcations and sol.is_bifurcation():
            u_old = self.u.copy()
            p_old = self.get_continuation_parameter()
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
                new_sol.nunstable_imaginary_eigenvalues = sol.nunstable_imaginary_eigenvalues
                # TODO: add the original solution point back to the branch?
            # reset the state to the original solution, assures continuation in right direction
            self.u = u_old
            self.set_continuation_parameter(p_old)

    def get_continuation_parameter(self) -> float:
        """return the value of the continuation parameter"""
        # make sure the continuation parameter is set
        assert self.continuation_parameter[0] is not None
        # get the value using the builtin 'getattr'
        obj, attr_name = self.continuation_parameter
        return getattr(obj, attr_name)

    def set_continuation_parameter(self, val) -> None:
        """set the value of the continuation parameter"""
        # make sure the continuation parameter is set
        assert self.continuation_parameter[0] is not None
        # assign the new value using the builtin 'setattr'
        obj, attr_name = self.continuation_parameter
        setattr(obj, attr_name, float(val))

    def locate_bifurcation(self, ev_index: Optional[int] = None, tolerance: float = 1e-5) -> bool:
        """
        locate the closest bifurcation using bisection method
        (finds point where the real part of the eigenvalue closest to zero vanishes)
        ev_index: optional index of the eigenvalue that corresponds to the bifurcation
        tolerance: threshold at which the value is considered zero
        returns True (False) if the location converged (or not)
        """
        # backup the initial state
        u_old = self.u
        p_old = self.get_continuation_parameter()
        # backup stepsize
        ds = self.continuation_stepper.ds
        # solve the eigenproblem
        # TODO: or recover them from self.eigen_solver.latest_eigenvalues?
        eigenvalues, _ = self.solve_eigenproblem()
        # get the eigenvalue that corresponds to the bifurcation
        # (the one with the smallest absolute real part)
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
        while np.abs(ev.real) > tolerance and intvl[1] - intvl[0] > 1e-4:
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
        if np.abs(ev.real) > tolerance * 100:
            self.u = u_old
            self.set_continuation_parameter(p_old)
            print("Warning: Failed to converge onto bifurcation point.")
            self.log("Corresponding eigenvalue:", ev)
            return False
        # if converged, return True
        return True

    def locate_bifurcation_using_constraint(self, eigenvector: np.ndarray) -> None:
        """locate the bifurcation of the given eigenvector"""
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

    def switch_branch(self, ev_index: Optional[int] = None, amplitude: float = 1e-3, locate: bool = True) -> bool:
        """attempt to switch branches in a bifurcation"""
        # try to converge onto a bifurcation nearby
        if locate:
            converged = self.locate_bifurcation(ev_index)
        else:
            converged = True
        if not converged:
            print(
                "Failed to converge onto a bifurcation point! Branch switching aborted.")
            return False
        # recover eigenvalues and -vectors from the eigensolver
        eigenvalues = self.eigen_solver.latest_eigenvalues
        eigenvectors = self.eigen_solver.latest_eigenvectors
        if eigenvalues is None or eigenvectors is None:
            return False
        # find the eigenvalue that corresponds to the bifurcation
        # (the one with the smallest abolute real part)
        if ev_index is None:
            ev_index = np.argsort(np.abs(eigenvalues.real))[0]
        self.log(
            f"Attempting to switch branch with eigenvector #{ev_index}")
        # get the eigenvector that corresponds to the bifurcation
        eigenvector = eigenvectors[ev_index]
        if not np.iscomplexobj(self.u):
            eigenvector = eigenvector.real
        # perturb unknowns in direction of eigenvector
        self.u = self.u + amplitude * np.linalg.norm(self.u) * eigenvector
        # TODO: deflate the original solution and newton_solve?
        # create a new branch in the bifurcation diagram
        self.new_branch()
        return True

    def new_branch(self) -> None:
        """create a new branch in the bifurcation diagram and prepare for a new continuation"""
        # create a new branch in the bifurcation diagram
        self.bifurcation_diagram.new_branch()
        # reset the settings and storage of the continuation stepper
        self.continuation_stepper.factory_reset()
        # clear the history of unknowns because it would otherwise be invalid
        self.history.clear()

    def norm(self) -> np.floating:
        """the default norm of the solution, used for bifurcation diagrams"""
        # TODO: @simon: if we want to calculate more than one measure,
        #       we could just return an array here, and do the choosing what
        #       to plot in the problem-specific plot function, right?
        # defaults to the L2-norm of the unknowns
        return np.linalg.norm(self.u)

    @profile
    def save(self, filename: Optional[str] = None) -> dict:
        """
        Save the current solution to the file <filename>.
        Returns a dictionary of the serialized data.
        """
        # dict of data to store
        data = {}
        # the number of equations
        equations = self.list_equations()
        data['Problem.nequations'] = len(equations)
        # the problem's time
        data['Problem.time'] = self.time
        # store the value of the continuation parameter
        if self.continuation_parameter[0] is not None:
            data['Problem.p'] = self.get_continuation_parameter()
        # The problem's unknowns won't need to be stored, since unknowns are
        # individually saved by the respective equations.
        # Fill the dict with data from each equation:
        for eq in equations:
            # obtain the equation's dict
            eq_data = eq.save()
            # prepend name of the equation, to make the keys unique and merge with problem's dict
            eq_name = type(eq).__name__ + "."
            data.update({eq_name+k: v for k, v in eq_data.items()})
        # save everything to the file
        if filename is not None:
            np.savez(filename, **data)
        # return the dict
        return data

    @profile
    def load(self, data) -> None:
        """
        Load the current solution from the given data.
        where 'data' can be a filename, a Solution object of a dictionary as returned
        by Problem.save(). Problem.load(...) is the inverse of Problem.save(...).
        """
        # if data is a Solution object:
        if isinstance(data, Solution):
            # get data from a solution object
            data = data.data
        # if data is a string:
        elif isinstance(data, str):
            # load data dictionary from the file
            data = np.load(data, allow_pickle=True)
        # clear the history
        self.history.clear()
        # load the time
        self.time = data['Problem.time']
        # load the value of the continuation parameter
        if self.continuation_parameter[0] is not None:
            self.set_continuation_parameter(data['Problem.p'])
        # let the equations restore their data
        for eq in self.list_equations():
            # strip the name of the equation
            eq_name = type(eq).__name__ + "."
            eq_data = {k.replace(eq_name, ''): v for k,
                       v in data.items() if k.startswith(eq_name)}
            # pass it to the equation, unless the dict is empty
            if eq_data:
                eq.load(eq_data)

    def adapt(self) -> None:
        """adapt the problem/equations to the solution (e.g. by mesh refinement)"""
        for eq in self.list_equations():
            eq.adapt()

    def log(self, *args, **kwargs) -> None:
        """
        print()-wrapper for log messages
        log messages are printed only if verbosity is switched on
        """
        if self.settings.verbose:
            print(*args, **kwargs)

    @profile
    def plot(self, sol_ax=None, bifdiag_ax=None, eigvec_ax=None, eigval_ax=None) -> None:
        """
        Plot everything to the given axes.
        Axes may be given explicitly of as a list of axes, that is then expanded.
        The plot may include the solution of the equations, the bifurcation diagram,
        the eigenvalues and the eigenvectors.
        """
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
                ev_is_imag = np.ma.masked_where(
                    np.abs(np.imag(self.eigen_solver.latest_eigenvalues)) <= self.settings.eigval_zero_tolerance, ev_re)
                eigval_ax.plot(ev_re_n, "o", color="C0", label="Re < 0")
                eigval_ax.plot(ev_re_p, "o", color="C1", label="Re > 0")
                eigval_ax.plot(ev_is_imag, "x", color="black",
                               label="complex", alpha=0.6)
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

    def generate_bifurcation_diagram(self,
                                     # limits for the continuation parameter
                                     parameter_lims=(-1e9, 1e9),
                                     # limits for the norm
                                     norm_lims=(-1e9, 1e9),
                                     # maximum recursion for sub-branch continuation
                                     max_recursion=4,
                                     # maximum number of steps per branch
                                     max_steps=1e9,
                                     # stop when solution reaches starting point again
                                     detect_circular_branches=True,
                                     # axes object to live plot the diagram
                                     ax=None,
                                     # plotting frequency
                                     plotevery=30
                                     ) -> None:
        """
        Automatically generate a full bifurcation diagram within the given bounds.
        Branch switching will be performed automatically up to the given maximum recursion level.
        """
        if ax is not None:
            plt.ion()
        # perform continuation of current branch until bounds are exceeded
        branch = self.bifurcation_diagram.active_branch
        n = 0
        norm = self.norm()
        param = self.get_continuation_parameter()
        u0 = self.u.copy()
        while True:
            # do continuation step
            self.continuation_step()
            # get new parameter and norm values
            param = self.get_continuation_parameter()
            norm = self.norm()
            n += 1
            # Check whether limits were exceeded
            if not parameter_lims[0] <= param <= parameter_lims[1]:
                print("Parameter limits exceeded for current branch. Parameter:", param)
                break
            if not norm_lims[0] <= norm <= norm_lims[1]:
                print("Norm limits exceeded for current branch. Norm:", norm)
                break
            # if maximum number of steps exceeded, abort
            if n > max_steps:
                print("Maximum number of steps exceeded for current branch.")
                break
            # if we are close to the intial solution, the branch is likely a circle, then abort
            distance = np.linalg.norm(self.u - u0) / np.linalg.norm(self.u)
            if n > 20 and distance < self.continuation_stepper.ds and detect_circular_branches:
                print("Branch has likely reached it's starting point again. Exiting this branch.\n"
                      "Set 'detect_circular_branches=False' to prevent this.")
                break
            # print status
            sol = self.bifurcation_diagram.current_solution()
            print(
                f"Branch #{branch.id}, Step #{n}, ds={self.continuation_stepper.ds:.2e}, #+EVs: {sol.nunstable_eigenvalues}")
            if sol.is_bifurcation():
                print(
                    f"Bifurcation found! #Null-EVs: {sol.neigenvalues_crossed}")
            # plot every few steps
            if ax is not None and plotevery is not None and n % plotevery == 0:
                self.plot(ax)
                plt.show(block=False)
                plt.pause(0.0001)
        # return if no more recursion is allowed
        if max_recursion < 1:
            return
        # if recursion is allowed, perform continuation of bifurcated branches
        # for each bifurcation point
        for bif in branch.bifurcations():
            # Hopf branches cannot yet be followed reliably, skip
            # TODO: change this once we have better support for Hopf branches
            if bif.bifurcation_type() == "HP":
                continue
            # load the bifurcation point into the problem
            self.load(bif)
            # attempt branch switching to new branch
            converged = self.switch_branch(locate=False)
            # skip this bifurcation, if we failed to converge onto the bifurcation point
            if not converged:
                continue
            # recursively generate a bifurcation diagram from the new branch
            self.generate_bifurcation_diagram(ax=ax,
                                              parameter_lims=parameter_lims,
                                              norm_lims=norm_lims,
                                              max_recursion=max_recursion-1,
                                              max_steps=max_steps,
                                              plotevery=plotevery)


class ProblemHistory():
    """
    This class manages the history of the unknowns and the time /
    continuation parameter of a given problem.
    The history is needed for implicit time-stepping schemes or for
    the calculation of tangents during parameter continuation.
    Note that this class does not actually store the history of the unknowns,
    which is rather found in the equations itself, in order to support adaption.
    """

    def __init__(self, problem: Problem) -> None:
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

    def update(self, history_type: Optional[str] = None) -> None:
        """update the history with the current unknowns of the problem"""
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

    def u(self, t: int = 0) -> np.ndarray:
        """get the unknowns at some point t in history"""
        # check length of history
        if t >= self.length:
            raise IndexError(
                f"Unknowns u[t=-{t}] requested, but history length is {self.length}")
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

    def time(self, t: int = 0) -> float:
        """get for the value of the time at some point t in history"""
        # accept negative and positive t
        t = abs(t)
        # check length of history
        if t >= self.length:
            raise IndexError(
                f"Unknowns u[t=-{t}] requested, but history length is {self.length}")
        # return the value
        return self.__t[t]

    def continuation_parameter(self, t: int = 0) -> float:
        """get for the value of the continuation_parameter at some point t in history"""
        # identical to fetching the time
        return self.time(t)

    def step_size(self, t: int = 0) -> float:
        """get for the value of the (time / continuation) step size at some point t in history"""
        # accept negative and positive t
        t = abs(t)
        # check length of history
        if t >= self.length:
            raise IndexError(
                f"Unknowns u[t=-{t}] requested, but history length is {self.length}")
        # return the value
        return self.__dt[t]

    @property
    def length(self) -> int:
        """returns the length of the history"""
        return len(self.__t)

    def clear(self) -> None:
        """clears the history"""
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

    def __init__(self) -> None:
        #: How many eigenvalues should be computed when problem.solve_eigenproblem() is called?
        #: Set to 'None' for computing all eigenvalues using a direct solver.
        #: TODO: could have a more verbose name
        self.neigs = 20
        #: How small does an eigenvalue need to be in order to be counted as 'zero'?
        self.eigval_zero_tolerance = 1e-16
        #: Should we always try to exactly locate bifurcations when passing one?
        self.always_locate_bifurcations = False
        #: Should sparse matrices be assumed when solving linear systems?
        self.use_sparse_matrices = True
        #: Should there be some extra output? useful for debugging
        self.verbose = False
