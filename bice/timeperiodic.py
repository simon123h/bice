from .equation import Equation
import numpy as np

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
    def __init__(self, reference_equation, T, Nt):
        super().__init__()
        # which equation to treat?
        self.ref_eq = reference_equation
        # the list of considered points in time (normalized time t' = t / T)
        self.ts = np.linspace(0, 1, Nt, endpoint=False)
        # the vector of unknowns: unknowns of the reference equation for every timestep
        self.u = np.repeat(self.ref_eq.u, Nt)
        # the period is also an unknown, append it to u
        # TODO: a good guess for T is 2pi / Im(lambda), with the unstable eigenvalue lambda
        self.u = np.append(self.u, T)

    # access the period length
    def T(self):
        return self.u[-1]

    # return the time derivative for a given list u's at each timestep using central differences
    def dudt(self, eq_u):
        return (np.roll(eq_u, 1) - np.roll(eq_u, -1)) / (np.roll(self.ts, 1) - np.roll(self.ts, -1))

    # calculate the rhs of the full system of equations
    def rhs(self, u):
        # dimension of a single equation
        N = self.ref_eq.dim
        # number of timesteps
        Nt = len(self.ts)
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
            res[-1] += np.dot(u[i], dudt_old[i]) * \
                (self.ts[(i+1) % Nt] - self.ts[i])
        # return the result
        return res
