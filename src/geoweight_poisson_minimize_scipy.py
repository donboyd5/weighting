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
from scipy.optimize import minimize

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


# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'maxiter': 100,
    'tol': 1e-6,
    'gtol': 1e-6,
    'ftol': 1e-7,
    'method': 'BFGS',  # BFGS L-BFGS-B Newton-CG trust-krylov, trust-ncg
    'hesstype': None,  # None, hessian, or hvp
    'disp': True,
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

    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f), primals, tangents)[1]

    def ljax_sspd(bvec):
        sspd = fgp.jax_sspd(bvec, wh, xmat, geotargets, dw) # * opts.objscale   # jax_sspd = jax.jit(jax_sspd)
        return jnp.asarray(sspd)

    lhvp = lambda x, p: hvp(ljax_sspd, (x, ), (p, ))  # GOOD

    lhessian = lambda x: jax.hessian(ljax_sspd)(x)

    hess = None
    hessp = None
    if opts.hesstype=='hessian':
        hess = lhessian
    elif opts.hesstype=='hvp':
        hessp = lhvp

    result = minimize(fun=ljax_sspd,
        x0=betavec0,
        method=opts.method,  # BFGS L-BFGS-B Newton-CG trust-krylov, trust-ncg
        jac=jax.jacfwd(ljax_sspd),
        hess=hess,
        hessp=hessp,
        tol=opts.tol,
        options={'maxiter': opts.maxiter,
                 'disp': opts.disp})

    # get return values
    beta_opt = result.x.reshape(geotargets.shape)
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
