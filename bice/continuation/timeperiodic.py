import numpy as np
from bice.core.equation import Equation

# TODO: make this work for multiple variables, regarding ref_eq.shape
# TODO: maybe use (Nt, N) as the shape


class TimePeriodicOrbitHandler(Equation):
    """
    The TimePeriodicOrbitHandler is an Equation that has an implicit
    time-stepping on a periodic time-mesh of (unknown) period length T.
    In a Problem, it can be used for solving periodic orbits, e.g. in a
    Hopf continuation. The referenced equation *should not* simultaneously
    be part of the problem.
    """

    # reference equation, initial guess for period length, initial number of points in time
    def __init__(self, reference_equation, T, Nt):
        super().__init__(shape=(Nt*reference_equation.u.size+1))
        # which equation to treat?
        self.ref_eq = reference_equation
        # the list of considered timesteps (in normalized time t' = t / T)
        self.dt = np.repeat(1./Nt, Nt)
        # the vector of unknowns: unknowns of the reference equation for every timestep
        u1 = np.tile(self.ref_eq.u, Nt)
        # the period is also an unknown, append it to u
        # TODO: a good guess for T is 2pi / Im(lambda), with the unstable eigenvalue lambda
        self.u = np.append(u1, T)

    # access the period length
    @property
    def T(self):
        return self.u[-1]

    @T.setter
    def T(self, v):
        self.u[-1] = v

    # number of points in time
    @property
    def Nt(self):
        return len(self.dt)

    # the unknowns in separate arrays for each point in time
    def u_orbit(self):
        # split the period and reshape to (Nt, N)
        return self.u[:-1].reshape((self.Nt, self.ref_eq.dim))

    # return the time derivative for a given list of u's at each timestep

    def dudt(self, u):
        # calculate central difference
        ul = np.roll(u, 1, axis=0)
        ur = np.roll(u, -1, axis=0)
        dtl = 1 / np.roll(self.dt, 1, axis=0)
        dtr = 1 / self.dt
        return (ur.T * dtr + (dtl - dtr) * u.T - ul.T * dtl).T / 2

    # return the jacobian of the time derivative for a given list of u's at each timestep
    def ddudtdu(self, u):
        # TODO: implement!
        pass

    # calculate the rhs of the full system of equations
    def rhs(self, u):
        # dimension of a single equation
        N = self.ref_eq.dim
        # number of timesteps
        Nt = self.Nt
        # split the unknowns into:
        # ... period length
        T = u[-1]
        # ... u's per timestep
        u = u[:-1].reshape((Nt, N))
        # calculate the time derivative
        dudt = self.dudt(u) / T
        # same for the old variables
        T_old = self.u[-1]
        u_old = self.u[:-1].reshape((Nt, N))
        dudt_old = self.dudt(u_old) / T_old
        # mass matrix
        M = self.ref_eq.mass_matrix()
        # setup empty result vector
        res = np.zeros(Nt*N + 1)
        # add the rhs contributions for each timestep
        for i in range(Nt):
            # 0 = rhs(u) - dudt for each u(t_i)
            res[i*N:(i+1)*N] = self.ref_eq.rhs(u[i]) - M.dot(dudt[i])
            # phase condition: \int_0^1 dt <u, dudt_old> = 0
            # TODO: use a better approximation for dt in integral?
            res[-1] += np.dot(u[i], dudt_old[i]) * self.dt[i]
        # return the result
        return res

    # calculate the Jacobian of rhs(u)
    def jacobian_INCOMPLETE(self, u):
        # dimension of a single equation
        N = self.ref_eq.dim
        # number of timesteps
        Nt = len(self.dt)
        # split the unknowns into:
        # ... period length
        T = u[-1]
        # ... u's per timestep
        u = u[:-1].reshape((Nt, N))
        # setup empty result matrix
        jac = np.zeros((Nt*N + 1, Nt*N + 1))
        # calculate jacobian of dudt
        # jac = self.ddudtdu(u) # TODO: this is still missing
        # calculate the time derivative
        dudt = self.dudt(u) / T
        # same for the old variables
        T_old = self.u[-1]
        u_old = self.u[:-1].reshape((Nt, N))
        dudt_old = self.dudt(u_old) / T_old
        # add the jacobian contributions of rhs for each timestep
        for i in range(Nt):
            # 0 = ref_eq.jacobian(u) for each u(t_i)
            jac[i*N:(i+1)*N, i*N:(i+1)*N] += self.ref_eq.jacobian(u[i])
            # phase condition: d [\int_0^1 dt <u, dudt_old>] du = \int_0^1 dt dudt_old
            jac[-1, i*N:(i+1)*N] += dudt_old[i] * self.dt[i]
            # add the T-derivative
            jac[i*N:(i+1)*N, -1] += -dudt[i] / T
        # no T-dependency in phase condition, so jac[-1, -1] = 0

    # adapt the time mesh to the solution
    def adapt(self, min_error=1e-5, max_error=1e-3, min_steps=10, max_steps=1e4):
        # dimension of a single equation
        N = self.ref_eq.dim
        # number of timesteps
        Nt = len(self.dt)
        # split the unknowns into:
        # ... period length
        T = self.u[-1]
        # ... u's per timestep
        u = self.u[:-1].reshape((Nt, N))
        # calculate the time derivative
        dudt = self.dudt(u)
        # estimate for the relative error in the time derivative
        # TODO: maybe there is something better than this
        error_estimate = np.array(
            [np.linalg.norm(dudt[i]) / np.linalg.norm(u[i]) for i in range(Nt)])
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
        u = np.append(u.reshape(u.size), [T])
        # if the size of u changed, re-add the equation to the problem
        if len(dt) != Nt:
            # store reference to the equation's problem
            problem = self.problem
            # remove the equation from the problem (if assigned)
            if problem is not None:
                problem.remove_equation(self)
            # update shape of equation
            self.shape = (len(dt)*N + 1,)
            # assign new variables and timesteps
            self.u = u
            self.dt = dt
            # re-add the equation to the problem
            if problem is not None:
                problem.add_equation(self)
        else:
            # else, we can simply overwrite the variables
            self.u = u
            self.dt = dt
        # return min/max error estimates
        return (min(error_estimate), max(error_estimate))
