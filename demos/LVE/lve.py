#!/usr/bin/python3
import time
import matplotlib

from bice.core.solvers import MyNewtonSolver  # noqa
matplotlib.use("GTK3Agg")  # noqa
from bice.continuation import TimePeriodicOrbitHandler, NaturalContinuation
from bice import Problem, Equation, time_steppers
import numpy as np
import matplotlib.pyplot as plt

# The Lotka-Volterra equations (predator prey model)
class LotkaVolterraEquations(Equation):
    def __init__(self):
        super().__init__()
        # parameters
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1
        # initial condition
        self.u = np.array([1., 0.7])

    def rhs(self, u):
        x, y = u
        return np.array([
            (self.a - self.b * y) * x,
            (self.d * x - self.c) * y
        ])


# The Lotka-Volterra equations (predator prey model)
class LotkaVolterra(Problem):

    def __init__(self):
        super().__init__()
        # create and add the equation
        self.lve = LotkaVolterraEquations()
        self.add_equation(self.lve)
        # time stepper
        # self.time_stepper = time_steppers.BDF(self)
        # self.time_stepper = time_steppers.BDF2(dt=1e-1)
        # self.time_stepper = time_steppers.Euler(dt=5e-4)
        # self.time_stepper = time_steppers.ImplicitEuler(dt=1e-2)
        self.time_stepper = time_steppers.RungeKutta4(dt=2e-1)


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
orbitHandler = TimePeriodicOrbitHandler(problem.eq, T, Nt)

# add initial condition to the TimePeriodicOrbitHandler
uu = []
problem.time_stepper.dt = T / Nt
for i in range(Nt):
    uu.append(problem.lve.u.copy())
    problem.time_step()
orbitHandler.u = np.append(uu, T)

# replace the Problem's equation with the Handler
problem.remove_equation(problem.eq)
problem.add_equation(orbitHandler)

x, y = orbitHandler.u_orbit().T
plt.plot(y, x, "x", color="green", label="continuation initial condition")

# Use simple MyNewtonSolver, because it is faster for small problems
problem.newton_solver = MyNewtonSolver()

# set up the natural continuation stepper
problem.continuation_stepper = NaturalContinuation()
problem.continuation_stepper.ds = 0.05
problem.continuation_parameter = (problem.lve, "b")
problem.neigs = 0

# perform continuation steps
n = 0
while n < 20:
    problem.continuation_step()
    n += 1
    # plot
    x, y = orbitHandler.u_orbit().T
    plt.plot(y, x, color="grey")
    print("\nStep", n)
    print("T =", orbitHandler.T)
    print("param =", problem.get_continuation_parameter())

    # calculate stability of orbits
    floquet_mul = orbitHandler.floquet_multipliers()
    print("Floquet multipliers:", floquet_mul)
    tol = 1e-5
    if np.any([abs(mul) > 1+tol for mul in floquet_mul]):
        print("(unstable)")
    else:
        print("(stable)")

    # NOTE: timestep adaption does not work well in this demo, because the solution is not unique
    # the LVE have a (complicated) conserved quantity, that we do not conserve, i.e., there is an
    # invariance to the solution. In timestep adaption, we easily shift to other solutions of the
    # solution family. We should really impose a constraint...
    # orbitHandler.adapt(1.3, 1.6, min_steps=30)


plt.legend()
plt.show()
