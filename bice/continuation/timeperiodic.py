import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from bice.core.equation import Equation

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
        self.ddt = self.build_ddt_matrix()

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
        # split the period and reshape to (Nt, *ref_eq.shape)
        return self.u[:-1].reshape((self.Nt, *self.ref_eq.shape))

    # build the time-derivative operator ddt, using periodic finite differences
    def build_ddt_matrix(self):
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

    # calculate the rhs of the full system of equations
    def rhs(self, u):
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

    # calculate the Jacobian of rhs(u)
    def jacobian(self, u):
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
        # The different contributions to the jacobian: ((#1, #2), (#3, #4))
        # 1.: bulk equations du: d ( rhs(u) - M*dudt ) / du
        # jacobian of M.dot(dudt) w.r.t. u
        d_bulk_du = sp.block_diag([self.ref_eq.jacobian(u[i])
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
        return sp.bmat([[d_bulk_du, d_bulk_dT],
                        [d_cnst_du, d_cnst_dT]])

    # adapt the time mesh to the solution
    # TODO: test this
    # TODO: ddt does currently not support non-uniform time, make uniform?
    def adapt(self):
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
        # return min/max error estimates
        return (min(error_estimate), max(error_estimate))

    # save the state of the equation, including the dt-values
    def save(self):
        data = super().save()
        data.update({'dt': self.dt})
        return data

    # load the state of the equation, including the dt-values
    def load(self, data):
        self.dt = data['dt']
        super().load(data)
