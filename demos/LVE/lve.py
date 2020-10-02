#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use("GTK3Agg")  # noqa
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")  # noqa, needed for relative import of package
from bice import Problem, Equation
from bice.time_steppers import Euler, ImplicitEuler, RungeKutta4, BDF, BDF2
from bice.timeperiodic import TimePeriodicOrbitHandler
import time

# The Lotka-Volterra equations (predator prey model)
class LotkaVolterraEquation(Equation):
    def __init__(self):
        super().__init__(shape=(2))
        # parameters
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1
        # initial condition
        self.u = np.array([1., 0.7])

    def rhs(self, u):
        x = u[0]
        y = u[1]
        return np.array([
            (self.a - self.b * y) * x,
            (self.d * x - self.c) * y
        ])


# The Lotka-Volterra equations (predator prey model)
class LotkaVolterra(Problem):

    def __init__(self):
        super().__init__()
        # create and add the equation
        self.lve = LotkaVolterraEquation()
        self.add_equation(self.lve)
        # time stepper
        # self.time_stepper = BDF(self)
        # self.time_stepper = BDF2(dt=1e-1)
        # self.time_stepper = Euler(dt=5e-4)
        # self.time_stepper = ImplicitEuler(dt=1e-2)
        self.time_stepper = RungeKutta4(dt=2e-1)


# create problem
problem = LotkaVolterra()

# time-stepping
data = []
# for n in range(10000):
start = time.time()
while problem.time < 1e3:
    data.append(problem.u.copy())
    problem.time_step()
end = time.time()
print("# steps:", len(data))
print("time   :", end - start)


data = np.array(data)
plt.plot(data[:, 0], data[:, 1], label="time-stepping solution")

# period length and number of timesteps
T = 6.34
Nt = 60

# create TimePeriodicOrbitHandler
orbitHandler = TimePeriodicOrbitHandler(problem.lve, T, Nt)

# add initial condition to the TimePeriodicOrbitHandler
uu = []
problem.time_stepper.dt = T / Nt
for i in range(Nt):
    uu.append(problem.lve.u.copy())
    problem.time_step()
orbitHandler.u = np.append(uu, T)

# replace the Problem's equation with the Handler
problem.remove_equation(problem.lve)
problem.add_equation(orbitHandler)

x, y = orbitHandler.u[:-1].reshape((Nt, 2)).T
plt.plot(y, x, "x", color="green", label="continuation initial condition")

n = 0
while n < 20:
    problem.newton_solve()
    print("T =", orbitHandler.u[-1])
    x, y = orbitHandler.u_orbit().T
    plt.plot(y, x, color="grey")
    # problem.lve.a += 0.05
    problem.lve.b += 0.05
    n += 1
    # NOTE: timestep adaption does not work well in this demo, because the solution is not unique
    # the LVE have a (complicated) conserved quantity, that we do not conserve, i.e., there is an
    # invariance to the solution. In timestep adaption, we easily shift to other solutions of the
    # solution family. We should really impose a constraint...
    # orbitHandler.adapt(1.3, 1.6, min_steps=30)


plt.legend()
plt.show()