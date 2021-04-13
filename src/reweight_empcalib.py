# -*- coding: utf-8 -*-
"""
Empirical calibration

@author: donbo
"""


# %% imports

import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from timeit import default_timer as timer

# pip install -q git+https://github.com/google/empirical_calibration
import empirical_calibration as ec

import src.utilities as ut


# %% default options

solver_defaults = {
    'baseline_weights': None,
    'target_weights': None,
    'objective': 'ENTROPY',
    'autoscale': True,
    'increment': 0.001
    }

user_defaults = {
    }

options_defaults = {**solver_defaults, **user_defaults}


# %% constants

SMALL_POSITIVE = np.nextafter(np.float64(0), np.float64(1))
# not sure if needed: a small nonzero number that can be used as a divisor
SMALL_DIV = SMALL_POSITIVE * 1e16
# 1 / SMALL_DIV  # does not generate warning

QUADRATIC = ec.Objective.QUADRATIC
ENTROPY = ec.Objective.ENTROPY


# %% gec primary function
def gec(wh, xmat, targets,
        options=None):

    a = timer()

    # update options with any user-supplied options
    if options is None:
        options_all = options_defaults.copy()
    else:
        options_all = options_defaults.copy()
        options_all.update(options)
        # options_all = {**options_defaults, **options}

    if options_all['objective'] == 'ENTROPY':
        options_all['objective'] = ENTROPY
    elif options_all['objective'] == 'QUADRATIC':
        options_all['objective'] = QUADRATIC

    # convert dict to named tuple for ease of use
    opts = ut.dict_nt(options_all)

    # small_positive = np.nextafter(np.float64(0), np.float64(1))
    wh = np.where(wh == 0, SMALL_POSITIVE, wh)
    wh = np.full(wh.shape, wh.mean())

    pop = wh.sum()
    tmeans = targets / pop

    # ompw:  optimal means-producing weights
    ompw, l2_norm = ec.maybe_exact_calibrate(
        covariates=xmat,
        target_covariates=tmeans.reshape((1, -1)),
        # baseline_weights=wh,
        # target_weights=np.array([[.25, .75]]), # target priorities
        # target_weights=target_weights,
        autoscale=opts.autoscale,  # doesn't always seem to work well
        # note that QUADRATIC weights often can be zero
        objective=opts.objective,  # ENTROPY or QUADRATIC
        increment=opts.increment
    )
    # print(l2_norm)

    # wh, when multiplied by g, will yield the targets
    g = ompw * pop / wh
    g = np.array(g, dtype=float).reshape((-1, ))  # djb
    wh_opt = g * wh
    targets_opt = np.dot(xmat.T, wh_opt)
    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'wh_opt',
              'targets_opt',
              'g',
              'opts',
              'l2_norm')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 wh_opt=wh_opt,
                 targets_opt=targets_opt,
                 g=g,
                 opts=opts,
                 l2_norm=l2_norm)

    return res
