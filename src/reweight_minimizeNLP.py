# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 05:25:57 2020

@author: donbo
"""


# %% imports
import numpy as np
from timeit import default_timer as timer
from collections import namedtuple
import scipy
from scipy.optimize import minimize
from scipy.sparse import identity

import src.utilities as ut


# %% default options

solver_defaults = {
    'verbose': 2,
    'gtol': 1e-4,
    'xtol': 1e-4,
    'initial_tr_radius': 1,  # default 1
    'factorization_method': 'AugmentedSystem'
    }

user_defaults = {
    'max_iter': 100
    }

options_defaults = {**solver_defaults, **user_defaults}
# print(options_defaults.keys())


# %% primary function
def rw_minNLP(wh, xmat, targets,
              options=None):

    a1 = timer()

    # update options with any user-supplied options
    if options is None:
        options_all = options_defaults.copy()
    else:
        options_all = options_defaults.copy()
        options_all.update(options)
        # options_all = {**options_defaults, **options}

    # rename dict keys as needed to reflect lsq naming
    options_all['maxiter'] = options_all.pop('max_iter')

    # convert dict to named tuple for ease of use
    opts = ut.dict_nt(options_all)

    h = wh.size

    # scale the targets to 100
    #diff_weights = np.where(targets != 0, 100 / targets, 1)
    diff_weights = np.ones_like(targets)

    b = targets * diff_weights
    b
    tol = .0001
    clb = b - tol * np.abs(b)
    cub = b + tol * np.abs(b)

    wmat = xmat * diff_weights

    At = np.multiply(wh.reshape(-1, 1), wmat)
    A = At.T
    As = scipy.sparse.coo_matrix(A)

    lincon = scipy.optimize.LinearConstraint(As, clb, cub)

    bnds = scipy.optimize.Bounds(0, 1e5)

    x0 = np.ones_like(wh)

    nlp_info = minimize(
        xm1_sq,
        x0,
        method='trust-constr',
        bounds=bnds,
        constraints=lincon,
        jac=xm1_sq_grad,
        # hess='2-point',
        # hess=xm1_sq_hess,
        hessp=xm1_sq_hvp,
        options=options_all
        )

    g = nlp_info.x
    wh_opt = wh * g
    targets_opt = np.dot(xmat.T, wh_opt)

    b1 = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'wh_opt',
              'targets_opt',
              'g',
              'opts',
              'nlp_info')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b1 - a1,
                 wh_opt=wh_opt,
                 targets_opt=targets_opt,
                 g=g,
                 opts=opts,
                 nlp_info=nlp_info)

    return res


# %% helper functions
def xm1_sq(x):
    """Calculate objective function."""
    return np.sum((x - 1)**2)


def xm1_sq_grad(x):
    """Calculate gradient of objective function."""
    return 2 * x - 2


def xm1_sq_hess(x):
    H = identity(x.size) * 2.0
    return H


def xm1_sq_hvp(x, p):
    # hessian vector product
    # dot product of hessian and an arbitray vector p
    # for our objective function this is just p * 2.0
    return p * 2.0



