#!/usr/bin/python3
from src.demo_problems import LotkaVolterra
import numpy as np

import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

lve = LotkaVolterra()

data = []

for n in range(10000):
    data.append(lve.u.copy())
    lve.time_step()
    # print("Step #{:05d}".format(n))

data = np.array(data).T
plt.plot(data[0], data[1])
plt.show()

