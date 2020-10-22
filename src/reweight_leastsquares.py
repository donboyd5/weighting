# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:24:14 2020

@author: donbo
"""

# %% imports
import numpy as np
from timeit import default_timer as timer
from collections import namedtuple
import scipy
from scipy.optimize import lsq_linear

import src.utilities as ut


# %% default options

solver_defaults = {
    'xlb': 0.0,
    'xub': 100.0,
    'method': 'bvls',  # bvls or trf
    'tol': None,  # 1e-6
    'lsmr_tol': 'auto',  # 'auto',  # None
    'max_iter': 50,
    'verbose': 2
    }

user_defaults = {
    'scaling': False
    }

options_defaults = {**solver_defaults, **user_defaults}


# %% primary function
def rw_lsq(wh, xmat, targets,
           options=None):

    a1 = timer()

    # update options with any user-supplied options
    if options is None:
        options_all = options_defaults.copy()
    else:
        options_all = options_defaults.copy()
        options_all.update(options)
        # options_all = {**options_defaults, **options}

    # convert dict to named tuple for ease of use
    opts = ut.dict_nt(options_all)

    h = wh.size

    # we are solving Ax = b, where
    #   b are the targets and
    #   A x multiplication gives calculated targets
    # using sparse matrix As instead of A

    # scale the targets to 100
    # TODO: deal with targets that are zero
    if opts.scaling is True:
        diff_weights = np.where(targets != 0, 100 / targets, 1)
    else:
        diff_weights = np.ones_like(targets)

    b = targets * diff_weights
    b

    wmat = xmat * diff_weights
    At = np.multiply(wh.reshape(-1, 1), wmat)
    A = At.T

    if opts.method != 'bvls':
        As = scipy.sparse.coo_matrix(A)
    else:
        # sparse matrices not allowed with bvls
        As = A

    lb = np.full(h, opts.xlb)
    ub = np.full(h, opts.xub)

    if opts.tol is None:
        lsq_info = lsq_linear(As, b, bounds=(lb, ub),
                         method=opts.method,
                         lsmr_tol=opts.lsmr_tol,
                         max_iter=opts.max_iter,
                         verbose=opts.verbose)
    else:
        lsq_info = lsq_linear(As, b, bounds=(lb, ub),
                         method=opts.method,
                         tol=opts.tol,  # tol=1e-6,
                         lsmr_tol=opts.lsmr_tol,
                         max_iter=opts.max_iter,
                         verbose=opts.verbose)

    g = lsq_info.x
    wh_opt = wh * g
    targets_opt = np.dot(xmat.T, wh_opt)

    b1 = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'wh_opt',
              'targets_opt',
              'g',
              'opts',
              'lsq_info')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b1 - a1,
                 wh_opt=wh_opt,
                 targets_opt=targets_opt,
                 g=g,
                 opts=opts,
                 lsq_info=lsq_info)

    return res

