"The core Problem class and helper classes."

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bice.continuation import PseudoArclengthContinuation
from bice.time_steppers.runge_kutta import RungeKutta4
from bice.time_steppers.time_steppers import TimeStepper

from .equation import Equation, EquationGroup, EquationLike
from .profiling import profile
from .solution import BifurcationDiagram, Solution
from .solvers import EigenSolver, NewtonKrylovSolver
from .types import Array, Matrix


class Problem:
    """
    Base class for all algebraic problems.

    It is an aggregate of (one or many) governing algebraic equations,
    initial and boundary conditions, constraints, etc. Also, it provides
    all the basic properties, methods and routines for treating the problem,
    e.g., time-stepping, solvers or plain analysis of the solution.
    Custom problems should be implemented as children of this class.
    """

    eq: None | Equation | EquationGroup

    # Constructor
    def __init__(self) -> None:
        """Initialize the Problem."""
        #: the equation (or system of equation) that governs the problem
        self.eq: EquationLike | None = None
        #: Time variable
        self.time: float = 0.0
        #: The time-stepper for integration in time
        self.time_stepper: TimeStepper = RungeKutta4(dt=1e-2)
        #: The continuation stepper for parameter continuation
        self.continuation_stepper = PseudoArclengthContinuation()
        #: The Newton solver for finding roots of equations
        self.newton_solver = NewtonKrylovSolver()
        #: The eigensolver for eigenvalues and -vectors
        self.eigen_solver = EigenSolver()
        #: The settings (tolerances, switches, etc.) are held by this ProblemSettings
        #: object
        self.settings = ProblemSettings()
        #: The history of the unknown values is accessed and managed with the
        #: Problem.history object
        self.history = ProblemHistory(self)
        #: The bifurcation diagram of the problem holds all branches and their solutions
        self.bifurcation_diagram = BifurcationDiagram()
        #: The continuation parameter is defined by passing an object and the name of
        #: the object's attribute that corresponds to the continuation parameter as a
        #: tuple
        self.continuation_parameter: tuple[Any, str] | None = None

    @property
    def ndofs(self) -> int:
        """
        Return the number of unknowns / degrees of freedom of the problem.

        Returns
        -------
        int
            The number of degrees of freedom.
        """
        if self.eq is None:
            return 0
        return self.eq.ndofs

    @property
    def u(self) -> Array:
        """
        Getter for unknowns of the problem.

        Returns
        -------
        Array
            The flattened vector of unknowns.
        """
        if self.eq is None:
            return np.array([])
        return self.eq.u.ravel()

    @u.setter
    def u(self, u) -> None:
        """
        Set the unknowns of the problem.

        Parameters
        ----------
        u
            The vector of unknowns.
        """
        assert self.eq is not None
        self.eq.u = u.reshape(self.eq.shape)

    def add_equation(self, eq: EquationLike) -> None:
        """
        Add an equation to the problem.

        Parameters
        ----------
        eq
            The equation to add.
        """
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

    def remove_equation(self, eq: EquationLike) -> None:
        """
        Remove an equation from the problem.

        Parameters
        ----------
        eq
            The equation to remove.
        """
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
        """
        List all equations that are part of the problem.

        Returns
        -------
        List[Equation]
            The list of equations.
        """
        if isinstance(self.eq, Equation):
            return [self.eq]
        if isinstance(self.eq, EquationGroup):
            return self.eq.list_equations()
        return []

    @profile
    def rhs(self, u: Array) -> Array:
        """
        Calculate the right-hand side of the system 0 = rhs(u).

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        Array
            The residuals vector.
        """
        assert self.eq is not None
        # adjust the shape and return the rhs of the (system of) equations
        return self.eq.rhs(u.reshape(self.eq.shape)).ravel()

    @profile
    def jacobian(self, u) -> Matrix:
        """
        Calculate the Jacobian of the system J = d rhs(u) / du for the unknowns u.

        Parameters
        ----------
        u
            The vector of unknowns.

        Returns
        -------
        Matrix
            The Jacobian matrix.
        """
        assert self.eq is not None
        # adjust the shape and return the Jacobian of the (system of) equations
        return self.eq.jacobian(u.reshape(self.eq.shape))

    @profile
    def mass_matrix(self) -> Matrix:
        """
        Return the mass matrix.

        The mass matrix determines the linear relation of the rhs to the temporal
        derivatives: M * du/dt = rhs(u).

        Returns
        -------
        Matrix
            The mass matrix.
        """
        assert self.eq is not None
        # return the mass matrix of the (system of) equations
        return self.eq.mass_matrix()

    @profile
    def newton_solve(self) -> None:
        """
        Solve the system rhs(u) = 0 for u with Newton's method.

        Updates self.u with the solution.
        """
        self.u = self.newton_solver.solve(self.rhs, self.u, self.jacobian)

    @profile
    def solve_eigenproblem(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the eigenvalues and eigenvectors of the Jacobian.

        The method will only calculate as many eigenvalues as requested with
        self.settings.neigs.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (eigenvalues, eigenvectors).
        """
        return self.eigen_solver.solve(self.jacobian(self.u), self.mass_matrix(), k=self.settings.neigs)

    @profile
    def time_step(self) -> None:
        """Integrate in time with the assigned time-stepper."""
        # update the history with the current state
        self.history.update(history_type="time")
        # perform timestep according to current scheme
        self.time_stepper.step(self)

    @profile
    def continuation_step(self) -> None:
        """Perform a parameter continuation step."""
        assert self.continuation_parameter is not None
        # update the history with the current state
        self.history.update(history_type="continuation")
        # perform the step with a continuation stepper
        self.continuation_stepper.step(self)
        # make sure the bifurcation diagram is up to date
        # TODO: this could be encapsulated within the BifurcationDiagram class or
        # somewhere else
        if self.bifurcation_diagram.parameter_name is None:
            self.bifurcation_diagram.parameter_name = self.continuation_parameter[1]
        elif self.bifurcation_diagram.parameter_name != self.continuation_parameter[1]:
            print(
                "Warning: continuation parameter changed from"
                "{self.bifurcation_diagram.parameter_name:s} to "
                "{self.continuation_parameter[1]:s}!"
                "Will generate a new bifurcation diagram!"
            )
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
            sol.nunstable_eigenvalues = len([ev for ev in eigenvalues if np.real(ev) > tol])
            sol.nunstable_imaginary_eigenvalues = len([ev for ev in eigenvalues if np.real(ev) > tol and abs(np.imag(ev)) > tol])
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
            # reset the state to the original solution, assures continuation in
            # right direction
            self.u = u_old
            self.set_continuation_parameter(p_old)

    def get_continuation_parameter(self) -> float:
        """
        Return the value of the continuation parameter.

        Returns
        -------
        float
            The current value of the parameter.
        """
        # make sure the continuation parameter is set
        assert self.continuation_parameter is not None
        # get the value using the builtin 'getattr'
        obj, attr_name = self.continuation_parameter
        return float(getattr(obj, attr_name))

    def set_continuation_parameter(self, val) -> None:
        """
        Set the value of the continuation parameter.

        Parameters
        ----------
        val
            The new value for the parameter.
        """
        # make sure the continuation parameter is set
        assert self.continuation_parameter is not None
        # assign the new value using the builtin 'setattr'
        obj, attr_name = self.continuation_parameter
        setattr(obj, attr_name, float(val))

    def locate_bifurcation(
        self,
        ev_index: int | None = None,
        tolerance: float = 1e-5,
    ) -> bool:
        """
        Locate the closest bifurcation using bisection method.

        Finds point where the real part of the eigenvalue closest to zero vanishes.

        Parameters
        ----------
        ev_index
            Optional index of the eigenvalue that corresponds to the bifurcation.
        tolerance
            Threshold at which the value is considered zero.

        Returns
        -------
        bool
            True if the location converged, False otherwise.
        """
        # backup the initial state
        u_old = self.u.copy()
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
        # TODO: location sometimes has troubles, when there is more than one null-
        #       eigenvalue
        # store the eigenvalue and its sign
        ev = eigenvalues[ev_index]
        sgn = np.sign(ev.real)
        # bisection interval and current position
        # TODO: it can happen that the bifurcation is at pos 1.001, then we will not
        #       find it! we somehow need to check on a broader interval first or known
        #       the sign of the eigenvalue at the limits of the interval
        intvl = (-1.0, 1.0)  # in multiples of step size
        pos: float = 1.0
        # bisection method loop
        while np.abs(ev.real) > tolerance and intvl[1] - intvl[0] > 1e-4:
            if self.settings.verbose:
                self.log(f"Bisection: [{intvl[0]:.6f} {intvl[1]:.6f}], Re: {ev.real:e}")
            # new middle point
            pos_old = pos
            pos = (intvl[0] + intvl[1]) / 2.0
            # perform the continuation step to new center point
            self.continuation_stepper.ds = ds * (pos - pos_old)
            try:
                # Note that we do not update the history, so the tangent remains
                # unchanged.
                # TODO: instead of continuation, we could also update (u, p) and do
                #       newton_solve() may be more stable
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
        """
        Locate the bifurcation of the given eigenvector.

        Parameters
        ----------
        eigenvector
            The eigenvector to use for the constraint.
        """
        assert self.continuation_parameter is not None
        # TODO: does not yet work!
        # make sure it is real, if self.u is real
        if not np.iscomplexobj(self.u):
            eigenvector = eigenvector.real
        # create the bifurcation constraint and add it to the problem
        from bice.continuation import BifurcationConstraint

        bifurcation_constraint = BifurcationConstraint(eigenvector, self.continuation_parameter)
        self.add_equation(bifurcation_constraint)
        # perform a newton solve
        self.newton_solve()
        # remove the constraint again
        self.remove_equation(bifurcation_constraint)

    def switch_branch(
        self,
        ev_index: int | None = None,
        amplitude: float = 1e-3,
        locate: bool = True,
    ) -> bool:
        """
        Attempt to switch branches in a bifurcation.

        Parameters
        ----------
        ev_index
            The index of the eigenvalue corresponding to the bifurcation.
        amplitude
            The amplitude of the perturbation.
        locate
            Whether to attempt to locate the bifurcation point first.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # try to converge onto a bifurcation nearby
        if locate:
            converged = self.locate_bifurcation(ev_index)
        else:
            converged = True
        if not converged:
            print("Failed to converge onto a bifurcation point! Branch switching aborted.")
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
        self.log(f"Attempting to switch branch with eigenvector #{ev_index}")
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
        """Create a new branch in the bifurcation diagram and prepare for continuation."""
        # create a new branch in the bifurcation diagram
        self.bifurcation_diagram.new_branch()
        # reset the settings and storage of the continuation stepper
        self.continuation_stepper.factory_reset()
        # clear the history of unknowns because it would otherwise be invalid
        self.history.clear()

    def norm(self) -> np.floating:
        """
        Return the default norm of the solution.

        Used for bifurcation diagrams. Defaults to the L2-norm of the unknowns.

        Returns
        -------
        float
            The norm value.
        """
        # TODO: @simon: if we want to calculate more than one measure,
        #       we could just return an array here, and do the choosing what
        #       to plot in the problem-specific plot function, right?
        return np.linalg.norm(self.u)

    @profile
    def save(self, filename: str | None = None) -> dict:
        """
        Save the current solution to the file <filename>.

        Returns a dictionary of the serialized data.

        Parameters
        ----------
        filename
            Optional filename to save to.

        Returns
        -------
        dict
            The serialized data.
        """
        # dict of data to store
        data: dict[str, Any] = {}
        # the number of equations
        equations = self.list_equations()
        data["Problem.nequations"] = len(equations)
        # the problem's time
        data["Problem.time"] = self.time
        # store the value of the continuation parameter
        if self.continuation_parameter is not None and self.get_continuation_parameter() is not None:
            data["Problem.p"] = self.get_continuation_parameter()
        # The problem's unknowns won't need to be stored, since unknowns are
        # individually saved by the respective equations.
        # Fill the dict with data from each equation:
        for eq in equations:
            # obtain the equation's dict
            eq_data = eq.save()
            # prepend name of the equation, to make the keys unique and merge with
            # problem's dict
            eq_name = type(eq).__name__ + "."
            data.update({eq_name + k: v for k, v in eq_data.items()})
        # save everything to the file
        if filename is not None:
            np.savez(filename, **data)
        # return the dict
        return data

    @profile
    def load(self, data) -> None:
        """
        Load the current solution from the given data.

        Problem.load(...) is the inverse of Problem.save(...).

        Parameters
        ----------
        data
            Filename, Solution object, or dictionary.
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
        self.time = float(data["Problem.time"])
        # load the value of the continuation parameter
        if self.continuation_parameter is not None and "Problem.p" in data:
            self.set_continuation_parameter(data["Problem.p"])
        # let the equations restore their data
        for eq in self.list_equations():
            # strip the name of the equation
            eq_name = type(eq).__name__ + "."
            eq_data = {k.replace(eq_name, ""): v for k, v in data.items() if k.startswith(eq_name)}
            # pass it to the equation, unless the dict is empty
            if eq_data:
                eq.load(eq_data)

    def adapt(self) -> None:
        """Adapt the problem/equations to the solution (e.g. by mesh refinement)."""
        for eq in self.list_equations():
            eq.adapt()

    def log(self, *args, **kwargs) -> None:
        """
        Wrap print() for log messages.

        Log messages are printed only if verbosity is switched on.

        Parameters
        ----------
        *args
            Positional arguments for print.
        **kwargs
            Keyword arguments for print.
        """
        if self.settings.verbose:
            print(*args, **kwargs)

    @profile
    def plot(
        self,
        sol_ax=None,
        bifdiag_ax=None,
        eigvec_ax=None,
        eigval_ax=None,
    ) -> None:
        """
        Plot everything to the given axes.

        Axes may be given explicitly or as a list of axes, that is then expanded.
        The plot may include the solution of the equations, the bifurcation diagram,
        the eigenvalues and the eigenvectors.

        Parameters
        ----------
        sol_ax
            Axes for plotting the solution.
        bifdiag_ax
            Axes for plotting the bifurcation diagram.
        eigvec_ax
            Axes for plotting the eigenvectors.
        eigval_ax
            Axes for plotting the eigenvalues.
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
            bifdiag_ax.plot(
                self.get_continuation_parameter(),
                self.norm(),
                "x",
                label="current point",
                color="black",
            )
            # plot the rest of the bifurcation diagram
            self.bifurcation_diagram.plot(bifdiag_ax)
        if eigval_ax is not None:
            # clear the axes
            eigval_ax.clear()
            # plot the eigenvalues, if any
            if self.eigen_solver.latest_eigenvalues is not None:
                ev_re = np.real(self.eigen_solver.latest_eigenvalues)
                ev_re_n = np.ma.masked_where(ev_re > self.settings.eigval_zero_tolerance, ev_re)
                ev_re_p = np.ma.masked_where(ev_re <= self.settings.eigval_zero_tolerance, ev_re)
                ev_is_imag = np.ma.masked_where(
                    np.abs(np.imag(self.eigen_solver.latest_eigenvalues)) <= self.settings.eigval_zero_tolerance,
                    ev_re,
                )
                eigval_ax.plot(ev_re_n, "o", color="C0", label="Re < 0")
                eigval_ax.plot(ev_re_p, "o", color="C1", label="Re > 0")
                eigval_ax.plot(ev_is_imag, "x", color="black", label="complex", alpha=0.6)
                eigval_ax.axhline(0, color="gray")
                eigval_ax.legend()
                eigval_ax.set_ylabel("eigenvalues")
            if eigvec_ax is not None:
                # clear the axes
                eigvec_ax.clear()
                # map the eigenvectors onto the equations and plot them
                if self.eigen_solver.latest_eigenvectors is not None:
                    ev = self.eigen_solver.latest_eigenvectors[0]
                    # fix orientation of eigenvector
                    sign = np.sign(ev.real.dot(self.u))
                    ev *= sign if sign != 0 else 1
                    # backup the unknowns
                    u_old = self.u.copy()
                    # overwrite the unknowns with the eigenvalues (or their real part
                    # only)
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

    def generate_bifurcation_diagram(
        self,
        parameter_lims=(-1e9, 1e9),
        norm_lims=(-1e9, 1e9),
        max_recursion=4,
        max_steps=1e9,
        detect_circular_branches=True,
        ax=None,
        plotevery=30,
    ) -> None:
        """
        Automatically generate a full bifurcation diagram within the given bounds.

        Branch switching will be performed automatically up to the given maximum
        recursion level.

        Parameters
        ----------
        parameter_lims
            Limits for the continuation parameter (min, max).
        norm_lims
            Limits for the norm (min, max).
        max_recursion
            Maximum recursion depth for sub-branch continuation.
        max_steps
            Maximum number of steps per branch.
        detect_circular_branches
            Stop when solution reaches starting point again.
        ax
            Axes object to live plot the diagram.
        plotevery
            Plotting frequency (every N steps).
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
            # if we are close to the intial solution, the branch is likely a circle,
            # then abort
            distance = np.linalg.norm(self.u - u0) / np.linalg.norm(self.u)
            if n > 20 and distance < self.continuation_stepper.ds and detect_circular_branches:
                print(
                    "Branch has likely reached it's starting point again. Exiting this "
                    "branch.\n"
                    "Set 'detect_circular_branches=False' to prevent this."
                )
                break
            # print status
            sol = self.bifurcation_diagram.current_solution()
            print(f"Branch #{branch.id}, Step #{n}, ds={self.continuation_stepper.ds:.2e}, #+EVs: {sol.nunstable_eigenvalues}")
            if sol.is_bifurcation():
                print(f"Bifurcation found! #Null-EVs: {sol.neigenvalues_crossed}")
            # plot every few steps
            if ax is not None and plotevery is not None and (n - 1) % plotevery == 0:
                self.plot(ax)
                plt.show(block=False)
                plt.pause(0.0001)  # type: ignore
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
            self.generate_bifurcation_diagram(
                ax=ax,
                parameter_lims=parameter_lims,
                norm_lims=norm_lims,
                max_recursion=max_recursion - 1,
                max_steps=max_steps,
                plotevery=plotevery,
                detect_circular_branches=detect_circular_branches,
            )


class ProblemHistory:
    """
    Manages the history of the unknowns and time/continuation parameter.

    The history is needed for implicit time-stepping schemes or for
    the calculation of tangents during parameter continuation.
    Note that this class does not actually store the history of the unknowns,
    which is rather found in the equations itself, in order to support adaption.
    """

    def __init__(self, problem: Problem) -> None:
        """Initialize the ProblemHistory."""
        # store reference to the problem
        self.problem = problem
        # maximum length of the history
        self.max_length = 4
        # what 'type' of history is currently stored? options are:
        #  - "time" for a time-stepping history
        #  - "continuation" for a history of continuation steps
        self.type: str | None = None
        # storage for the values of the time or the continuation parameter
        self.__t: list[float] = []
        # storage for the values of the stepsize
        self.__dt: list[float] = []

    def update(self, history_type: str | None = None) -> None:
        """
        Update the history with the current unknowns of the problem.

        Parameters
        ----------
        history_type
            "time" or "continuation".
        """
        # make sure that the history is of correct type, do not mix different types
        if self.type != history_type:
            # if it is of a different type, clear the history first and assign new type
            self.clear()
            self.type = history_type
        # check the minimum length of the history in each equation
        eq_hist_length = min([len(eq.u_history) for eq in self.problem.list_equations()])
        # make sure that the equation's history length matches the parameter history's
        # length
        self.__t = self.__t[:eq_hist_length]
        # update the history of unknowns in every equation
        for eq in self.problem.list_equations():
            eq.u_history = [eq.u.copy()] + eq.u_history[: self.max_length - 1]
        # add the value of the time / continuation parameter and step size to the history
        val: float
        dval: float
        if self.type == "continuation":  # for continuation
            val = self.problem.get_continuation_parameter()
            dval = self.problem.continuation_stepper.ds
        else:  # for time stepping
            val = self.problem.time
            dval = self.problem.time_stepper.dt
        self.__t = [val] + self.__t[: self.max_length - 1]
        self.__dt = [dval] + self.__dt[: self.max_length - 1]

    def u(self, t: int = 0) -> Array:
        """
        Get the unknowns at some point t in history.

        Parameters
        ----------
        t
            Time step index (backwards).

        Returns
        -------
        Array
            The unknowns at that point.
        """
        # check length of history
        if t >= self.length:
            raise IndexError(f"Unknowns u[t=-{t}] requested, but history length is {self.length}")
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
        """
        Get the value of the time at some point t in history.

        Parameters
        ----------
        t
            Time step index (backwards).

        Returns
        -------
        float
            The time value.
        """
        # accept negative and positive t
        t = abs(t)
        # check length of history
        if t >= self.length:
            raise IndexError(f"Unknowns u[t=-{t}] requested, but history length is {self.length}")
        # return the value
        return self.__t[t]

    def continuation_parameter(self, t: int = 0) -> float:
        """
        Get the value of the continuation_parameter at some point t in history.

        Parameters
        ----------
        t
            Step index (backwards).

        Returns
        -------
        float
            The parameter value.
        """
        # identical to fetching the time
        return self.time(t)

    def step_size(self, t: int = 0) -> float:
        """
        Get the value of the (time / continuation) step size at some point t in history.

        Parameters
        ----------
        t
            Step index (backwards).

        Returns
        -------
        float
            The step size.
        """
        # accept negative and positive t
        t = abs(t)
        # check length of history
        if t >= self.length:
            raise IndexError(f"Unknowns u[t=-{t}] requested, but history length is {self.length}")
        # return the value
        return self.__dt[t]

    @property
    def length(self) -> int:
        """
        Return the length of the history.

        Returns
        -------
        int
            The history length.
        """
        return len(self.__t)

    def clear(self) -> None:
        """Clear the history."""
        # clear the history type
        self.type = None
        # clear the history of unknowns in each equation
        for eq in self.problem.list_equations():
            eq.u_history = []
        # clear the time / continuation parameter history
        self.__t = []


class ProblemSettings:
    """A wrapper class that holds all the settings of a problem."""

    def __init__(self) -> None:
        """Initialize the ProblemSettings."""
        #: How many eigenvalues should be computed when problem.solve_eigenproblem()
        #: is called?
        #: Set to 'None' for computing all eigenvalues using a direct solver.
        #: TODO: could have a more verbose name
        self.neigs: int | None = 20
        #: How small does an eigenvalue need to be in order to be counted as 'zero'?
        self.eigval_zero_tolerance = 1e-16
        #: Should we always try to exactly locate bifurcations when passing one?
        self.always_locate_bifurcations = False
        #: Should sparse matrices be assumed when solving linear systems?
        self.use_sparse_matrices = True
        #: Should there be some extra output? useful for debugging
        self.verbose = False
