import numpy as np


class TimeStepper():

    def __init__(self, dt=1e-2):
        self.dt = dt

    # perform a timestep on a problem
    def step(self, problem):
        raise NotImplementedError

# Explicit Euler scheme


class Euler(TimeStepper):
    # perform timestep
    def step(self, problem):
        problem.u += self.dt * problem.rhs(problem.u)
        problem.time += self.dt

# Classical Runge-Kutta-4 scheme


class RungeKutta4(TimeStepper):
    # perform timestep
    def step(self, problem):
        k1 = problem.rhs(problem.u)
        problem.time += self.dt/2.
        k2 = problem.rhs(problem.u + self.dt / 2 * k1)
        k3 = problem.rhs(problem.u + self.dt / 2 * k2)
        problem.time += self.dt/2.
        k4 = problem.rhs(problem.u + self.dt * k3)
        problem.u += self.dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4)


# Runge-Kutta-Fehlberg-4-5 scheme with adaptive stepsize
class RungeKuttaFehlberg45(TimeStepper):

    # Coefficients related to the independent variable of the evaluations
    a2 = 2.500000000000000e-01  # 1/4
    a3 = 3.750000000000000e-01  # 3/8
    a4 = 9.230769230769231e-01  # 12/13
    a5 = 1.000000000000000e+00  # 1
    a6 = 5.000000000000000e-01  # 1/2

    # Coefficients related to the dependent variable of the evaluations
    b21 = 2.500000000000000e-01  # 1/4
    b31 = 9.375000000000000e-02  # 3/32
    b32 = 2.812500000000000e-01  # 9/32
    b41 = 8.793809740555303e-01  # 1932/2197
    b42 = -3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00  # 7296/2197
    b51 = 2.032407407407407e+00  # 439/216
    b52 = -8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00  # 3680/513
    b54 = -2.058966861598441e-01  # -845/4104
    b61 = -2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00  # 2
    b63 = -1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01  # 1859/4104
    b65 = -2.750000000000000e-01  # -11/40

    # Coefficients related to the truncation error
    # Obtained through the difference of the 5th and 4th order RK methods:
    #     R = (1/h)|y5_i+1 - y4_i+1|
    r1 = 2.777777777777778e-03  # 1/360
    r3 = -2.994152046783626e-02  # -128/4275
    r4 = -2.919989367357789e-02  # -2197/75240
    r5 = 2.000000000000000e-02  # 1/50
    r6 = 3.636363636363636e-02  # 2/55

    # Coefficients related to RK 4th order method
    c1 = 1.157407407407407e-01  # 25/216
    c3 = 5.489278752436647e-01  # 1408/2565
    c4 = 5.353313840155945e-01  # 2197/4104
    c5 = -2.000000000000000e-01  # -1/5

    def __init__(self, dt=1e-2):
        super().__init__(dt)
        # Local truncation error tolerance
        self.error_tolerance = 1e-3

    # perform timestep and adapt stepsize
    def step(self, problem):
        # Store evaluation values
        t = problem.time
        k1 = self.dt * problem.rhs(problem.u)
        problem.time = t + self.a2 * self.dt
        k2 = self.dt * problem.rhs(problem.u + self.b21 * k1)
        problem.time = t + self.a3 * self.dt
        k3 = self.dt * problem.rhs(problem.u + self.b31 * k1 + self.b32 * k2)
        problem.time = t + self.a4 * self.dt
        k4 = self.dt * \
            problem.rhs(problem.u + self.b41 * k1 +
                        self.b42 * k2 + self.b43 * k3)
        problem.time = t + self.a5 * self.dt
        k5 = self.dt * problem.rhs(problem.u + self.b51 *
                                   k1 + self.b52 * k2 + self.b53 * k3 + self.b54 * k4)
        problem.time = t + self.a6 * self.dt
        k6 = self.dt * problem.rhs(problem.u + self.b61 * k1 + self.b62 *
                                   k2 + self.b63 * k3 + self.b64 * k4 + self.b65 * k5)

        # Calulate local truncation error
        eps = np.linalg.norm(self.r1 * k1 + self.r3 * k3 + self.r4 *
                             k4 + self.r5 * k5 + self.r6 * k6) / self.dt

        # If it is less than the tolerance, the step is accepted and RK4 value is stored
        if eps <= self.error_tolerance:
            problem.time = t + self.dt
            problem.u += self.c1 * k1 + self.c3 * k3 + self.c4 * k4 + self.c5 * k5

        # calculate next step size
        if eps != 0:
            self.dt = self.dt * \
                min(max(0.84 * (self.error_tolerance / eps)**0.25, 0.5), 2)
