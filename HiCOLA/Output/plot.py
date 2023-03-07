#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:35:20 2023

@author: gupta
"""

import numpy as np
from matplotlib import pyplot as plt

a, beta, chiod = np.loadtxt('cuGal_f0p2_14dp_WMAP_force.txt',unpack=True)
a2, beta2, chiod2 = np.loadtxt('cG-f0.2_force.txt', unpack=True)

fig, ax = plt.subplot()
ax.plot(a, beta, label='coupling new')
ax.plot(a2, beta2, label='coupling old')
ax.plot(a, chiod, label='screening new')
ax.plot(a2, chiod2, label='screening old')
ax.legend()
