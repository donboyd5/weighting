#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 07:54:35 2021

@author: donboyd
"""

# %% imports
# for checking:
# import sys; print(sys.executable)
# print(sys.path)

import numpy as np

import scipy
from scipy.optimize import lsq_linear
from numpy.random import seed
from timeit import default_timer as timer

import src.make_test_problems as mtp
import src.microweight as mw

# import src.poisson as ps

# print(mw.__doc__)

# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% functions
def targs(targvec, div=50, seed=seed(1234)):
    r = np.random.randn(targvec.size) / 50  # random normal
    targets = (targvec * (1 + r)).flatten()
    return targets


# %% make problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10, s=2, k=2)
p = mtp.Problem(h=40, s=2, k=3)


# %% new info
