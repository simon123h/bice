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

filepath = '/local0/m_holl20/cPFC3_1d/semiactive_cPFC3_v0_{:1.2f}/'

problem = acPFCProblem(N=512, L=32*np.pi)
problem.continuation_parameter = None

fig = plt.figure(figsize=(16,9))
ax_sol = fig.add_subplot(111)


for v0  in np.arange(0., 0.7, 0.05):
    path = filepath.format(v0) + 'out/'
    filelist = [f for f in os.listdir(path) if f.endswith('.npz')]
    for f in filelist:
        problem.load(path + f)
        problem.plot(ax_sol)
