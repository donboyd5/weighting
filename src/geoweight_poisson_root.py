# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

@author: donbo
"""

# %% imports
import scipy
import scipy.optimize as spo
import gc
import numpy as np

import jax
import jax.numpy as jnp
# this next line is CRUCIAL or we will lose precision
from jax.config import config; config.update("jax_enable_x64", True)

from timeit import default_timer as timer
from collections import namedtuple
import src.utilities as ut

import src.functions_geoweight_poisson as fgp

from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian
# jac = BroydenFirst()
# kjac = KrylovJacobian(inner_M=InverseJacobian(jac))


# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'stepmethod': 'jvp',  # vjp, jvp, full, finite-diff
    'max_nfev': 100,
    'ftol': 1e-7,
    'x_scale': 'jac',
    'quiet': True,
    'solver_opts': None}

# options_defaults = {**solver_defaults, **user_defaults}


# %% poisson - the primary function

def poisson(wh, xmat, geotargets, options=None):
    a = timer()

    # jac = BroydenFirst()
    # kjac = KrylovJacobian(inner_M=InverseJacobian(jac))
    # kjac = kjac = KrylovJacobian(inner_M=jac.inverse)
    # jac = BroydenFirst()
    # kjac = KrylovJacobian(inner_M=jac.inverse)

    options_all = options_defaults
    options_all.update(options)

    # tmp1 = options_all['solver_opts']
    # tmp2 = tmp1['jac_options']
    # tmp2['inner_M'] = kjac
    # tmp1['jac_options'] = tmp2
    # options_all['solver_opts'] = tmp1

    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = fgp.scale_problem(xmat, geotargets, opts.scale_goal)

    # betavec0 = np.zeros(geotargets.size)
    betavec0 = np.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    dw = fgp.jax_get_diff_weights(geotargets)  # get_diff_weights(geotargets)

    # determine which jacobian method to use
    if opts.stepmethod == 'jvp':
        jax_jacobian_basic = jax.jit(fgp.jac_jvp(fgp.jax_targets_diff, wh, xmat, geotargets, dw))  # jax_jacobian_basic is a function -- the jax jacobian
    elif opts.stepmethod == 'vjp':
        jax_jacobian_basic = jax.jit(fgp.jac_vjp(fgp.jax_targets_diff, wh, xmat, geotargets, dw))
    elif opts.stepmethod == 'jac':
        jax_jacobian_basic = jax.jit(jax.jacfwd(fgp.jax_targets_diff))  # jit definitely faster
    else:
        jax_jacobian_basic = None

    def jax_jacobian(beta, wh, xmat, geotargets, dw):
        jac_values = jax_jacobian_basic(beta, wh, xmat, geotargets, dw)
        jac_values = np.array(jac_values).reshape((dw.size, dw.size))
        return jac_values

    if opts.jac == 'jac':
        jac = jax_jacobian
    else:
        jac = None




    # CAUTION: linear operator approach does NOT work well because scipy least_squares does not allow the option x_scale='jac' when using a linear operator
    # This is fast and COULD be very good if a good scaling vector is developed but without that it iterates quickly but reduces
    # cost very slowly.

    # jax_jacobian_basic = jax.jit(jac_jvp(jax_targets_diff))  # jax_jacobian_basic is a function -- the jax jacobian
    # if opts.stepmethod == 'findiff':
    #     stepmethod = '2-point'
    # elif opts.stepmethod == 'jvp-linop':
    #     stepmethod = fgp.jvp_linop  # CAUTION: this method does not allow x_scale='jac' and reduces costs slowly
    # else:
    #     stepmethod = jax_jacobian

    # spo_result = spo.least_squares(
    #     fun=fgp.jax_targets_diff_copy,  # targets_diff,
    #     x0=betavec0,
    #     method='trf', jac=stepmethod, verbose=2,
    #     ftol=opts.ftol, xtol=1e-7,
    #     x_scale=opts.x_scale,
    #     loss='soft_l1',  # linear, soft_l1, huber, cauchy, arctan,
    #     max_nfev=opts.max_nfev,
    #     args=(wh, xmat, geotargets, dw))

    spo_result = spo.root(
        fun=fgp.jax_targets_diff_copy,  # targets_diff,
        x0=betavec0,
        args=(wh, xmat, geotargets, dw),
        method=opts.solver,
        jac=jac,
        tol=None,
        callback=None,
        options=opts.solver_opts)

    # get return values
    beta_opt = spo_result.x.reshape(geotargets.shape)
    whs_opt = fgp.get_whs_logs(beta_opt, wh, xmat, geotargets) # jax_get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = jnp.dot(whs_opt.T, xmat)

    if opts.scaling:
        geotargets_opt = np.multiply(geotargets_opt, scale_factors)

    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets_opt',
              'beta_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 whs_opt=whs_opt,
                 geotargets_opt=geotargets_opt,
                 beta_opt=beta_opt)

    return res
