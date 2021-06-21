#!/usr/bin/python3
import shutil
import os
import sys
import numpy as np
import matplotlib
#matplotlib.use('svg')
import matplotlib.pyplot as plt
sys.path.append("../..")  # noqa, needed for relative import of package
from acPFC1d import acPFCProblem
import sys

#filepath = '/local0/m_holl20/cPFC3_1d/semiactive_cPFC3_v0_{:1.2f}/'
filepath = '/home/max/promotion/timesims/cPFC3_1d/semiactive_cPFC3_v0_{:1.2f}/'

Nx = 512
Lx = 32*np.pi

problem = acPFCProblem(N=Nx, L=Lx)
problem.continuation_parameter = None


x = np.linspace(-Lx/2, Lx/2, Nx)

for v0  in [0.]:#np.arange(0., 0.7, 0.05):
    path = filepath.format(v0) + 'out/'
    filelist = [f for f in os.listdir(path) if f.endswith('.npz')]
    filelist.sort()
    times = []
    sol_phi1 = []
    sol_phi2 = []
    sol_P = []
    for f in filelist:
        problem.load(path + f)
        times += [problem.time]
        sol_phi1 += [problem.acpfc.u[0]]
        sol_phi2 += [problem.acpfc.u[1]]
        sol_P += [problem.acpfc.u[2]]
    times = np.array(times)
    X, T = np.meshgrid(x, times)
    print(times)
    sol_phi1 = np.array(sol_phi1)
    sol_phi2 = np.array(sol_phi2)
    sol_P = np.array(sol_P)
    fig = plt.figure(figsize=(4,9))
    ax_phi1 = fig.add_subplot(311)
    ax_phi2 = fig.add_subplot(313)
    ax_P = fig.add_subplot(312)

    ax_phi1.pcolormesh(T, X, sol_phi1, shading='nearest', cmap='Reds')
    ax_phi2.pcolormesh(T, X, sol_phi2, shading='nearest', cmap='Blues')
    ax_P.pcolormesh(T, X, sol_P, shading='nearest', cmap='Greens')
    plt.show()
    