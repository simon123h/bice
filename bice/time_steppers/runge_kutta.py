from typing import TYPE_CHECKING

import numpy as np

from .time_steppers import TimeStepper

if TYPE_CHECKING:
    from bice.core.problem import Problem


class RungeKutta4(TimeStepper):
    """
    Classical Runge-Kutta-4 scheme
    """

    # perform timestep
    def step(self, problem: 'Problem') -> None:
        k1 = problem.rhs(problem.u)
        problem.time += self.dt/2.
        k2 = problem.rhs(problem.u + self.dt / 2 * k1)
        k3 = problem.rhs(problem.u + self.dt / 2 * k2)
        problem.time += self.dt/2.
        k4 = problem.rhs(problem.u + self.dt * k3)
        problem.u += self.dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4)


# Runge-Kutta-Fehlberg-4-5 scheme with adaptive step size
class RungeKuttaFehlberg45(TimeStepper):
    """
    Runge-Kutta-Fehlberg(45) scheme with adaptive step size.
    Local truncation error is estimated by comparison of
    RK4 and RK5 schemes and determines the optimal step size.
    """

    # Coefficients borrowed from:
    # https://github.com/LorranSutter/DiscreteMethods/blob/master/discreteMethods.py

    # Coefficients related to the independent variable of the evaluations
    _a2 = 2.500000000000000e-01  # 1/4
    _a3 = 3.750000000000000e-01  # 3/8
    _a4 = 9.230769230769231e-01  # 12/13
    _a5 = 1.000000000000000e+00  # 1
    _a6 = 5.000000000000000e-01  # 1/2

    # Coefficients related to the dependent variable of the evaluations
    _b21 = 2.500000000000000e-01  # 1/4
    _b31 = 9.375000000000000e-02  # 3/32
    _b32 = 2.812500000000000e-01  # 9/32
    _b41 = 8.793809740555303e-01  # 1932/2197
    _b42 = -3.277196176604461e+00  # -7200/2197
    _b43 = 3.320892125625853e+00  # 7296/2197
    _b51 = 2.032407407407407e+00  # 439/216
    _b52 = -8.000000000000000e+00  # -8
    _b53 = 7.173489278752436e+00  # 3680/513
    _b54 = -2.058966861598441e-01  # -845/4104
    _b61 = -2.962962962962963e-01  # -8/27
    _b62 = 2.000000000000000e+00  # 2
    _b63 = -1.381676413255361e+00  # -3544/2565
    _b64 = 4.529727095516569e-01  # 1859/4104
    _b65 = -2.750000000000000e-01  # -11/40

    # Coefficients related to the truncation error
    # Obtained through the difference of the 5th and 4th order RK methods:
    #     R = (1/h)|y5_i+1 - y4_i+1|
    _r1 = 2.777777777777778e-03  # 1/360
    _r3 = -2.994152046783626e-02  # -128/4275
    _r4 = -2.919989367357789e-02  # -2197/75240
    _r5 = 2.000000000000000e-02  # 1/50
    _r6 = 3.636363636363636e-02  # 2/55

    # Coefficients related to RK 4th order method
    _c1 = 1.157407407407407e-01  # 25/216
    _c3 = 5.489278752436647e-01  # 1408/2565
    _c4 = 5.353313840155945e-01  # 2197/4104
    _c5 = -2.000000000000000e-01  # -1/5

    def __init__(self, dt: float = 1e-2, error_tolerance: float = 1e-3) -> None:
        super().__init__(dt)
        # Local truncation error tolerance
        self.error_tolerance = error_tolerance
        # Maximum number of iterations when steps are rejected
        self.max_rejections = 30
        # counter for the number of rejections in current step
        self.rejection_count = 0

    # perform timestep and adapt step size
    def step(self, problem: 'Problem') -> None:
        # Store evaluation values
        t = problem.time
        k1 = self.dt * problem.rhs(problem.u)
        problem.time = t + self._a2 * self.dt
        k2 = self.dt * problem.rhs(problem.u + self._b21 * k1)
        problem.time = t + self._a3 * self.dt
        k3 = self.dt * problem.rhs(problem.u +
                                   self._b31 * k1 + self._b32 * k2)
        problem.time = t + self._a4 * self.dt
        k4 = self.dt * \
            problem.rhs(problem.u + self._b41 * k1 +
                        self._b42 * k2 + self._b43 * k3)
        problem.time = t + self._a5 * self.dt
        k5 = self.dt * problem.rhs(problem.u + self._b51 *
                                   k1 + self._b52 * k2 + self._b53 * k3 + self._b54 * k4)
        problem.time = t + self._a6 * self.dt
        k6 = self.dt * problem.rhs(problem.u + self._b61 * k1 + self._b62 *
                                   k2 + self._b63 * k3 + self._b64 * k4 + self._b65 * k5)

        # Calulate local truncation error
        eps = np.linalg.norm(self._r1 * k1 + self._r3 * k3 + self._r4 *
                             k4 + self._r5 * k5 + self._r6 * k6) / self.dt

        # Calculate next step size
        # NOTE: we may adjust the safety factor here
        dt_old = self.dt
        if eps != 0:
            self.dt = self.dt * \
                min(max(1 * (self.error_tolerance / float(eps))**0.25, 0.5), 2)

        # If it is less than the tolerance, the step is accepted and RK4 value is stored
        if eps <= self.error_tolerance:
            # update problem variables
            problem.time = t + dt_old
            problem.u += self._c1 * k1 + self._c3 * k3 + self._c4 * k4 + self._c5 * k5
            # reset rejection count
            self.rejection_count = 0
        elif self.rejection_count < self.max_rejections:
            # if step rejected: repeat step with the adjusted step size
            self.rejection_count += 1
            self.step(problem)
        else:
            # if we rejected too many steps already: abort with Exception
            raise Exception(
                f"Runge-Kutta-Fehlberg time-stepper exceeded maximum"
                f"number of rejected steps: {self.rejection_count}")
