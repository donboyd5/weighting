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
    'target_weights': None,
    'objective': 'ENTROPY',
    'autoscale': True,
    'increment': 0.001
    }


# %% constants

SMALL_POSITIVE = np.nextafter(np.float64(0), np.float64(1))
# not sure if needed: a small nonzero number that can be used as a divisor
SMALL_DIV = SMALL_POSITIVE * 1e16
# 1 / SMALL_DIV  # does not generate warning

QUADRATIC = ec.Objective.QUADRATIC
ENTROPY = ec.Objective.ENTROPY


# %% gec primary function
def gec(wh, xmat, targets,
        solver_options=None):

    a = timer()

    # update options with any user-supplied options
    if solver_options is None:
        solver_options = solver_defaults
    else:
        solver_options = {**solver_defaults, **solver_options}

    if solver_options['objective'] == 'ENTROPY':
        solver_options['objective'] = ENTROPY
    elif solver_options['objective'] == 'QUADRATIC':
        solver_options['objective'] = QUADRATIC

    so = ut.dict_nt(solver_options)

    # small_positive = np.nextafter(np.float64(0), np.float64(1))
    wh = np.where(wh == 0, SMALL_POSITIVE, wh)

    pop = wh.sum()
    tmeans = targets / pop

    # ompw:  optimal means-producing weights
    ompw, l2_norm = ec.maybe_exact_calibrate(
        covariates=xmat,
        target_covariates=tmeans.reshape((1, -1)),
        baseline_weights=wh,
        # target_weights=np.array([[.25, .75]]), # target priorities
        # target_weights=target_weights,
        autoscale=so.autoscale,  # doesn't always seem to work well
        # note that QUADRATIC weights often can be zero
        objective=so.objective,  # ENTROPY or QUADRATIC
        increment=so.increment
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
              'l2_norm')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 wh_opt=wh_opt,
                 targets_opt=targets_opt,
                 g=g,
                 l2_norm=l2_norm)

    return res
