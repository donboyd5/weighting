# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

@author: donbo
"""

# %% imports

import importlib

import numpy as np
import jax
import jax.numpy as jnp
# from jax.scipy.optimize import minimize
from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian


# import numpy as jnp
# from scipy.optimize import minimize

# this next line is CRUCIAL or we will lose precision
from jax.config import config; config.update("jax_enable_x64", True)

from timeit import default_timer as timer
from collections import namedtuple
import src.utilities as ut
import src.functions_geoweight_poisson as fgp

# %% reimports
importlib.reload(fgp)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton_krylov.html

# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.0,
    'maxiter': 100,
    # 'method': 'newton-krylov',  # BFGS L-BFGS-B Newton-CG trust-krylov, trust-ncg
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}


# %% poisson - the primary function

def poisson(wh, xmat, geotargets, options=None):
    a = timer()

    options_all = options_defaults.copy()
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    # TODO: input checking

    if opts.scaling:
        xmat, geotargets, scale_factors = fgp.scale_problem(xmat, geotargets, opts.scale_goal)

    betavec0 = jnp.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    dw = fgp.jax_get_diff_weights(geotargets)

    l_diffs = lambda betavec0: fgp.jax_targets_diff(betavec0, wh, xmat, geotargets, dw)

    def ljax_sspd(bvec):
        sspd = fgp.jax_sspd(bvec, wh, xmat, geotargets, dw) # * opts.objscale   # jax_sspd = jax.jit(jax_sspd)
        return jnp.asarray(sspd)

    def get_jac(bvec, wh, xmat, geotargets, dw):
        jacfn = jax.jacfwd(fgp.jax_targets_diff)
        jacfn = jax.jit(jacfn)
        jacmat = jacfn(bvec, wh, xmat, geotargets, dw)
        jacmat = np.array(jacmat).reshape((bvec.size, bvec.size))
        return jacmat
    l_jac = lambda bvec: get_jac(bvec, wh, xmat, geotargets, dw)

    if opts.inner_M:
        # jac = BroydenFirst(alpha=opts.bfalpha)
        kjac = KrylovJacobian(inner_M=InverseJacobian(l_jac))
        inner_M = kjac
    else:
        inner_M = None

    result = newton_krylov(
        F=l_diffs,
        xin=betavec0,
        iter=opts.iter,
        rdiff=opts.rdiff,
        method='lgmres',
        inner_maxiter=opts.inner_maxiter,
        inner_M=inner_M,
        outer_k=opts.outer_k,
        verbose=opts.verbose,
        maxiter=opts.maxiter,
        f_tol=opts.f_tol,
        f_rtol=opts.f_rtol,
        x_tol=opts.x_tol,
        x_rtol=opts.x_rtol,
        tol_norm=opts.tol_norm,
        line_search=opts.line_search,
        callback=opts.callback)

    # get return values
    # print(result)
    beta_opt = result.reshape(geotargets.shape)
    whs_opt = fgp.get_whs_logs(beta_opt, wh, xmat, geotargets)
    geotargets_opt = jnp.dot(whs_opt.T, xmat)

    if opts.scaling:
        geotargets_opt = jnp.multiply(geotargets_opt, scale_factors)

    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets_opt',
              'beta_opt',
              'result')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 whs_opt=whs_opt,
                 geotargets_opt=geotargets_opt,
                 beta_opt=beta_opt,
                 result=result)

    return res
