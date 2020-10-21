# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 06:20:10 2020

@author: donbo
"""

# %% imports
# import warnings
import numpy as np
from timeit import default_timer as timer
from collections import namedtuple

import src.utilities as ut
import src.raking as raking


# %% default options
user_defaults = {
    'max_iter': 10
    }


# %% primary function

def rw_rake(wh, xmat, targets, user_options):

    a = timer()

    # update options to override defaults and add any passed options
    if user_options is None:
        user_options = user_defaults
    else:
        user_options = {**user_defaults, **user_options}

    # create named tuple from dict to gain dot access to members
    uo = ut.dict_nt(user_options)

    g = raking.rake(wh=wh, xmat=xmat, targets=targets, max_iter=uo.max_iter)

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

