#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from acPFC_fd import acPFC
import sys


filepath = '/local0/m_holl20/cPFC3_1d/semiactive_cPFC3_v0_0.00/'
shutil.rmtree(filepath + 'out', ignore_errors=True)
os.makedirs(filepath + 'out/img', exist_ok=True)

# create problem
problem = acPFC(N=512, L=32*np.pi)

# create figure

fig = plt.figure(figsize=(16,9))
ax_sol = fig.add_subplot(111)
plotID = 0
#shooting gauss
for idx in [0, 1]:
    i = 0
    cond = True
    plotevery = 10
    dudtnorm = 1
    n = 0


    T = 5000.

    while cond:
        i += 1
        print('loop ', i)
        n = 0
        problem.time = 0.
        problem.time_stepper.factory_reset()

        while problem.time < T:
            if n % plotevery == 0:
                problem.plot(ax_sol)
                fig.savefig(filepath + f'out/img/{plotID:07d}.svg')
                plotID += 1
                print(f"step #: {n}")
                print(f"time:   {problem.time}")
                print(f"dt:     {problem.time_stepper.dt}")
                print(f"|dudt|: {dudtnorm}")
                print('mean u0', problem.acpfc.u[0].mean())

            n += 1
            # perform timestep
            problem.time_step()
            dudtnorm = np.linalg.norm(problem.rhs(problem.u))
            if np.max(problem.u) > 1e12:
                print("diverged")
                break
        cond = problem.acpfc.add_gauss_to_sol(idx)
# iterate over activity

T = 10000
for v0 in np.arange(0., 0.7, 0.05):
    problem.time = 0
    problem.time_stepper.factory_reset()
    i = 0
    plotID = 0
    print(v0)
    problem.acpfc.v0 = v0
    # set new path
    filepath = f'/local0/m_holl20/cPFC3_1d/semiactive_cPFC3_v0_{v0:1.2f}/'
    shutil.rmtree(filepath + 'out', ignore_errors=True)
    os.makedirs(filepath + 'out/img', exist_ok=True)
    while problem.time < T:
            if i % plotevery == 0:
                problem.plot(ax_sol)
                fig.savefig(filepath + f'out/img/{plotID:07d}.svg')
                problem.save(filename=filepath + f'out/{plotID:07d}.npz')

                plotID += 1
                print(f"step #: {i}")
                print(f"time:   {problem.time}")
                print(f"dt:     {problem.time_stepper.dt}")
                print(f"|dudt|: {dudtnorm}")
            i += 1
            # perform timestep
            problem.time_step()
            dudtnorm = np.linalg.norm(problem.rhs(problem.u))
            if np.max(problem.u) > 1e12:
                print("diverged")
                break
