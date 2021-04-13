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
    'max_rake_iter': 10
    }

solver_defaults = {}

options_defaults = {**solver_defaults, **user_defaults}


# %% primary function

def rw_rake(wh, xmat, targets, options):

    a = timer()

    # update options with any user-supplied options
    if options is None:
        options_all = options_defaults.copy()
    else:
        options_all = options_defaults.copy()
        options_all.update(options)
        # options_all = {**options_defaults, **options}

    # convert dict to named tuple for ease of use
    opts = ut.dict_nt(options_all)

    g = raking.rake(wh=wh, xmat=xmat, targets=targets, max_iter=opts.max_rake_iter)

    wh_opt = g * wh
    targets_opt = np.dot(xmat.T, wh_opt)
    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'wh_opt',
              'targets_opt',
              'g',
              'opts')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 wh_opt=wh_opt,
                 targets_opt=targets_opt,
                 g=g,
                 opts=opts)

    return res

