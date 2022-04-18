import numpy as np
import scipy.sparse as sp
import scipy.linalg
from bice.core.equation import Equation
from bice.core.profiling import profile
import matplotlib.pyplot as plt

# TODO: make this work for multiple variables, regarding ref_eq.shape


class TimePeriodicOrbitHandler(Equation):
    """
    The TimePeriodicOrbitHandler is an Equation that has an implicit
    time-stepping on a periodic time-mesh of (unknown) period length T.
    In a Problem, it can be used for solving periodic orbits, e.g. in a
    Hopf continuation. The referenced equation *should not* simultaneously
    be part of the problem.
    """

    # reference equation, initial guess for period length, initial number of points in time
    def __init__(self, reference_equation, T, Nt) -> None:
        super().__init__(shape=(Nt*reference_equation.ndofs+1))
        # which equation to treat?
        self.ref_eq = reference_equation
        # the list of considered timesteps (in normalized time t' = t / T)
        self.dt = np.repeat(1./Nt, Nt)
        # the vector of unknowns: unknowns of the reference equation for every timestep
        u1 = np.tile(self.ref_eq.u, Nt)
        # the period is also an unknown, append it to u
        # TODO: a good guess for T is 2pi / Im(lambda), with the unstable eigenvalue lambda
        self.u = np.append(u1, T)
        # the finite differences matrix to compute the temporal derivative du/dt from given u[t]
        self.ddt = self.build_ddt_matrix()
        # cache for storing the Jacobians J[u, t_i] = d(ref_eq.rhs)/du for each point in time t_i
        self._jacobian_cache = []

    @property
    def T(self) -> float:
        """Access the period length"""
        return self.u[-1]

    @T.setter
    def T(self, v: float) -> None:
        self.u[-1] = v

    @property
    def t(self) -> np.ndarray:
        """Return the temporal domain vector"""
        return np.cumsum(self.dt)

    @property
    def Nt(self) -> int:
        """Number of points in time"""
        return len(self.dt)

    def u_orbit(self) -> np.ndarray:
        """The unknowns in separate arrays for each point in time"""
        # split the period and reshape to (Nt, *ref_eq.shape)
        return self.u[:-1].reshape((self.Nt, *self.ref_eq.shape))

    def build_ddt_matrix(self) -> sp.csr_matrix:
        """Build the time-derivative operator ddt, using periodic finite differences"""
        # time-derivative operator
        # TODO: build using FinDiff or numdifftoos.fornberg
        #       then we can also support non-uniform time-grids
        Nt = self.Nt
        I = np.eye(Nt)
        dt = self.dt[0]
        ddt = np.zeros((Nt, Nt))
        ddt += -3*np.roll(I, -4, axis=1)
        ddt += 32*np.roll(I, -3, axis=1)
        ddt += -168*np.roll(I, -2, axis=1)
        ddt += 672*np.roll(I, -1, axis=1)
        ddt -= 672*np.roll(I, 1, axis=1)
        ddt -= -168*np.roll(I, 2, axis=1)
        ddt -= 32*np.roll(I, 3, axis=1)
        ddt -= -3*np.roll(I, 4, axis=1)
        # TODO: the minus should not be here!
        ddt /= -dt * 840
        # convert to sparse
        return sp.csr_matrix(ddt)

    @profile
    def rhs(self, u: np.ndarray) -> np.ndarray:
        """Calculate the rhs of the full system of equations"""
        # number of unknowns of a single equation
        N = self.ref_eq.ndofs
        # split the unknowns into:
        # ... period length
        T = u[-1]
        # ... u's per timestep
        u = u[:-1].reshape((self.Nt, *self.ref_eq.shape))
        # calculate the time derivative using FD
        dudt = self.ddt.dot(u) / T
        # same for the old variables
        T_old = self.u[-1]
        u_old = self.u[:-1].reshape((self.Nt, *self.ref_eq.shape))
        dudt_old = self.ddt.dot(u_old) / T_old
        # mass matrix
        M = self.ref_eq.mass_matrix()
        # setup empty result vector
        res = np.zeros(self.ndofs)
        # add the rhs contributions for each timestep
        for i in range(self.Nt):
            # 0 = rhs(u) - dudt for each u(t_i)
            # TODO: .ravel() will be required somewhere here
            res[i*N:(i+1)*N] = self.ref_eq.rhs(u[i]) - M.dot(dudt[i])
            # phase condition: \int_0^1 dt <u, dudt_old> = 0
            # TODO: use a better approximation for dt in integral?
            res[-1] += np.dot(u[i], dudt_old[i]) * self.dt[i]
        # return the result
        return res

    @profile
    def jacobian(self, u: np.ndarray) -> sp.csr_matrix:
        """Calculate the Jacobian of rhs(u)"""
        # split the unknowns into:
        # ... period length
        T = u[-1]
        # ... u's per timestep
        u = u[:-1].reshape((self.Nt, *self.ref_eq.shape))
        # calculate the time derivative
        dudt = self.ddt.dot(u) / T
        # same for the old variables
        T_old = self.u[-1]
        u_old = self.u[:-1].reshape((self.Nt, *self.ref_eq.shape))
        dudt_old = self.ddt.dot(u_old) / T_old
        # mass matrix
        M = self.ref_eq.mass_matrix()
        # Jacobian of reference equation for each time step
        # also, cache the Jacobians for later Floquet multiplier computation
        self._jacobian_cache = [
            sp.csr_matrix(self.ref_eq.jacobian(u[i])) for i in range(self.Nt)]
        # The different contributions to the jacobian: ((#1, #2), (#3, #4))
        # 1.: bulk equations du: d ( rhs(u) - M*dudt ) / du
        # jacobian of M.dot(dudt) w.r.t. u
        d_bulk_du = sp.block_diag([self._jacobian_cache[i]
                                   for i in range(self.Nt)]) - sp.kron(self.ddt, M) / T
        # 2.: bulk equations dT: d ( rhs(u) - M*dudt ) / dT
        d_bulk_dT = sp.csr_matrix(np.concatenate(
            [M.dot(dudt[i]) / T for i in range(self.Nt)])).T
        # 3.: constraint equation du: d ( \int_0^1 dt <u, dudt_old> = 0 ) / du
        d_cnst_du = sp.csr_matrix(np.concatenate(
            [dudt_old[i] * self.dt[i] for i in range(self.Nt)]))
        # 4.: cnst equation dT: d ( \int_0^1 dt <u, dudt_old> = 0 ) / du = 0
        d_cnst_dT = 0*sp.csr_matrix((1, 1))
        # combine the contributions to the full jacobian and return
        final_jac = sp.bmat([[d_bulk_du, d_bulk_dT],
                             [d_cnst_du, d_cnst_dT]])
        return sp.csr_matrix(final_jac)

    def monodromy_matrix(self, use_cache: bool = True) -> sp.csr_matrix:
        """
        Calculate the monodromy matrix A that is used to calculate the stability of the orbit
        using the Floquet multipliers (eigenvalues of A).

        The matrix is computed as the product of the reference equations's Jacobians for
        each point in time, i.e.:
        A = J[u(t_N), t_N] * J[u(t_{N-1}), t_{N-1}] * ... * J[u(t_0), t_0]

        The Jacobians J[u, t] are cached when the total Jacobian of the orbit handler
        (TimePeriodicOrbitHandler.jacobian(u)) is computed. This normally happens during solving
        (unless using a Krylov method), so they should be up to date as the stability calculation
        should normally happen after solving. If caching is not desired, the cache can be ignored
        by setting `use_cache = False`.
        """
        # store whether there was something cached
        had_cache = len(self._jacobian_cache) > 0
        # check if the number of cached Jacobians matches the number of time steps
        if len(self._jacobian_cache) != self.Nt:
            # if not, we will definitely need to regenerate the Jacobians
            use_cache = False
        # If we are not using the cached Jacobians, regenerate Jacobians by computing the
        # orbit handlers Jacobian (this method updates the cache)
        if not use_cache:
            _ = self.jacobian(self.u)
        # Cache should now be up to date
        # multiply the Jacobians in reversed order
        jacs = self._jacobian_cache[::-1]
        mon_mat = jacs[0]
        for i in range(1, self.Nt):
            mon_mat = mon_mat.dot(jacs[i])
        # If there was no cache, it is likely that we are using some matrix free method
        # (e.g. Krylov subspace methods) for Jacobian estimation that does not generate a cache.
        # If so, invalidate the cache:
        if not had_cache:
            self._jacobian_cache = []
        # return the monodromy matrix
        return mon_mat

    def floquet_multipliers(self, k: int = 20, use_cache: bool = True) -> np.ndarray:
        """
        Calculate the Floquet multipliers to obtain the stability of the orbit.
        The Floquet multipliers are the eigenvalues of the monodromy matrix
        (cf. TimePeriodicOrbitHander.monodromy_matrix(...)).

        k is the number of desired Floquet multipliers to be calculated by the iterative eigensolver

        If `use_cache=True` (default), the Jacobians will be cached from the last solving step.
        This is typically a good choice, because it saves computation time and the solving should
        happen right before the stability calculation. However, if a (matrix free) Krylov method
        is used for solving, the Jacobians will never actually be computed, thus requiring
        `use_cache=False`.
        """
        # obtain the monodromy matrix and mass matrix
        A = self.monodromy_matrix(use_cache)
        M = self.ref_eq.mass_matrix()
        # number of degrees of freedom of the original equation (--> matrices are NxN)
        N = self.ref_eq.ndofs
        # make sure we do not request more Floquet multipliers than degrees of freedom
        k = min(k, N)
        # if N is very small, fallback to dense matrices
        if N < 100:
            # make sure both are dense
            if sp.issparse(A):
                A = A.todense()
            if sp.issparse(M):
                M = M.todense()
            # calculate eigenvalues and return
            eigval, _ = scipy.linalg.eig(A, M)
            return eigval[:k]
        else:
            # make sure both are sparse
            if not sp.issparse(A):
                A = sp.csr_matrix(A)
            if not sp.issparse(M):
                M = sp.csr_matrix(M)
            # calculate eigenvalues and return
            eigval, _ = sp.linalg.eigs(A, k=k, M=M, sigma=1)
            return eigval

    # TODO: test this
    # TODO: ddt does currently not support non-uniform time, make uniform?
    def adapt(self) -> tuple:
        """Adapt the time mesh to the solution"""
        # number of unknowns of a single equation
        N = self.ref_eq.ndofs
        # split the unknowns into:
        # ... period length
        T = self.u[-1]
        # ... u's per timestep
        u = self.u[:-1].reshape((self.Nt, *self.ref_eq.shape))
        # calculate the time derivative
        dudt = self.ddt.dot(u)
        # estimate for the relative error in the time derivative
        # TODO: maybe there is something better than this
        error_estimate = np.array(
            [np.linalg.norm(dudt[i]) / np.linalg.norm(u[i]) for i in range(self.Nt)])
        # define error tolerances
        min_error = 1e-5
        max_error = 1e-3
        min_steps = 10
        max_steps = 1e4
        # (un)refinement loop
        i = 0
        dt = self.dt.copy()
        while i < len(dt):
            e = error_estimate[i]
            # unrefinement
            if e < min_error and len(dt) > min_steps:
                u = np.delete(u, i, axis=0)
                dt = np.delete(dt, i)
                error_estimate = np.delete(error_estimate, i)
                i += 1
            # refinement
            if e > max_error and len(dt) < max_steps:
                i2 = (i+1) % len(dt)
                u = np.insert(u, i, 0.5 * (u[i] + u[i2]), axis=0)
                dt = np.insert(dt, i, 0.5 * (dt[i] + dt[i2]))
                error_estimate = np.insert(
                    error_estimate, i, 0.5 * (min_error + max_error))
                i += 1
            i += 1
        # build new u
        u = np.append(u.ravel(), [T])
        # update shape of equation
        self.reshape(len(dt)*N + 1)
        # assign new variables and timesteps
        self.u = u
        self.dt = dt
        # if the equation belongs to a group of equations, redo it's mapping of the unknowns
        if self.group is not None:
            self.group.map_unknowns()
        # rebuild FD time-derivative matrix
        self.ddt = self.build_ddt_matrix()
        # invalidate the cached Jacobians
        self._jacobian_cache = []
        # return min/max error estimates
        return (min(error_estimate), max(error_estimate))

    def save(self) -> dict:
        """Save the state of the equation, including the dt-values"""
        data = super().save()
        data.update({'dt': self.dt})
        return data

    def load(self, data) -> None:
        """Load the state of the equation, including the dt-values"""
        self.dt = data['dt']
        # rebuild FD time-derivative matrix
        self.ddt = self.build_ddt_matrix()
        # invalidate the cached Jacobians
        self._jacobian_cache = []

        super().load(data)

    def plot(self, ax) -> None:
        """Plot the solutions for different timesteps"""
        orbit = self.u_orbit().T
        num_plots = min(40, self.Nt)
        cmap = plt.cm.viridis
        ax.set_prop_cycle(plt.cycler(
            'color', cmap(np.linspace(0, 1, num_plots))))
        for i in range(0, self.Nt, self.Nt//num_plots):
            self.ref_eq.u = orbit.T[i]
            self.ref_eq.plot(ax)
