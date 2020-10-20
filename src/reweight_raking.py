# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 06:20:10 2020

@author: donbo
"""

# %% imports
import warnings
import numpy as np
from timeit import default_timer as timer
from collections import namedtuple

import src.raking as raking

# wh = p.wh
# xmat = p.xmat
# targets = p.targets * (1 + noise)
# q = 1


# %% primary function

def rw_rake(wh, xmat, targets, max_iter):

    a = timer()
    g = raking.rake(wh=wh, xmat=xmat, targets=targets, max_iter=max_iter)

    wh_opt = g * wh
    targets_opt = np.dot(xmat.T, wh_opt)
    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'wh_opt',
              'targets_opt',
              'g')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 wh_opt=wh_opt,
                 targets_opt=targets_opt,
                 g=g)

    return res

