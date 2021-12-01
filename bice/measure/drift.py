
"""
This file provides a method that determines the drift velocity of the solution u(x, t) of a
partial differential equation.

=== Derivation of the drift velocity formula ===
cf. B. Wallmeyer, master thesis (2014); S. Hartmann, master thesis (2018)
Consider the field of u(x, t) with spatial coordinates (arbitrary dimension) x and time t.
Approximate time evolution of the field for infinitesimal small time interval dt:
u(x, t + dt) = u(x + v dt, t) + v_h dt + Res(x, t) dt
with lateral drift velocity vector v, 'vertical' velocity v_h and some residuals/deviation Res.
Using Taylor expansion of u(x + v*dt, t) for small dt we follow:
u(x, t + dt) = u(x, t) + v dt * grad u(x + v*dt, t) + v_h dt + Res(x, t) dt.
Divide by dt and use finite difference:
du/dt = v * grad u + v_h + Res.
We now minimize the total deviation G(v, v_h) = int_K Res^2 dx on the domain K by requiring that
the partial derivatives of G w.r.t. (v, v_h) vanish. We obtain:
v_h = 1/K int_K du/dt dx - 1/K v * int_K grad u dx = avg(du/dt) - v * avg(grad u)
with the spatial average avg(...) = 1/K int_K ... dx.
Using v_h, we also obtain from the minimization:
int_K [du/dx_i - avg(du/dx_i)] * [du/dt - v * grad u] dx = 0.
This is equal to a linear system A*v = b, that we can solve for v.
"""

import numpy as np


def calculateDriftVelocity(eq):
    """
    Calculate the lateral drift velocity of the spatio-temporal solution u(x, t) of a given
    partial differential equation eq, where x can be of arbitrary dimension.
    """
    # get the dimension of u and the spatial dimension of x
    N = eq.ndofs
    sdim = len(eq.x)
    # TODO: add support for multi-variable equations?
    #       --> calculate drift of single variable only
    # TODO: what if len(eq.shape) > 1?
    # calculate the time derivative
    dudt = eq.du_dt(eq.u)
    # calculate the spatial derivatives
    dudx = [eq.du_dx(eq.u, direction=d) for d in range(sdim)]
    # calculate the average of the spatial derivatives
    dudx_avg = [np.average(dudx[d]) for d in range(sdim)]
    # create the linear system, solution is the drift velocity vector
    # TODO: the sums in the linear system actually correspond to a spatial integration,
    #       --> not valid for irregular spatial grid, needs fix
    A = np.zeros((N, N))
    b = np.zeros(N)
    for i in range(sdim):
        for j in range(sdim):
            A[i, j] = np.sum((dudx[i] - dudx_avg[i])*dudx[j])
        b[i] = -np.sum((dudx[i] - dudx_avg[i])*dudt)
    # solve the linear system and return result vector
    return np.linalg.solve(A, b)
