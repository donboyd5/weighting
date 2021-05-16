# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/optimizer/lbfgs_minimize

@author: donbo
"""

# %% imports
import importlib

import jax
import jax.numpy as jnp
# from jax.scipy.optimize import minimize
from tensorflow_probability.substrates import jax as tfp

# import numpy as jnp
# from scipy.optimize import minimize

# this next line is CRUCIAL or we will lose precision
from jax.config import config; config.update("jax_enable_x64", True)

from timeit import default_timer as timer
from collections import namedtuple
import src.utilities as ut
import src.functions_geoweight_poisson as fgp

# tfp.substrates.jax.optimizer.bfgs_minimize(
#     value_and_gradients_function, initial_position, tolerance=1e-08, x_tolerance=0,
#     f_relative_tolerance=0, initial_inverse_hessian_estimate=None,
#     max_iterations=50, parallel_iterations=1, stopping_condition=None,
#     validate_args=True, max_line_search_iterations=50, name=None
# )


# %% import reloads
importlib.reload(fgp)


# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'objscale': 1.0,
    'method': 'BFGS',  # BFGS or LBFGS
    'max_iterations': 50,
    'max_line_search_iterations': 50,
    'num_correction_pairs': 10,
    'parallel_iterations': 1,
    'tolerance': 1e-8,
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}


# %% poisson - the primary function

def poisson(wh, xmat, geotargets, options=None):
    a = timer()

    options_all = options_defaults.copy()
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = fgp.scale_problem(xmat, geotargets, opts.scale_goal)

    betavec0 = jnp.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    dw = fgp.jax_get_diff_weights(geotargets)

    ljax_sspd = lambda bvec: fgp.jax_sspd(bvec, wh, xmat, geotargets, dw) * opts.objscale


    def loss_and_gradient(x):
        return tfp.math.value_and_gradient(lambda x: ljax_sspd(x), x)

    if opts.method == 'BFGS':
        result = tfp.optimizer.bfgs_minimize(
            loss_and_gradient,
            initial_position=betavec0,
            tolerance=opts.tolerance,
            max_iterations=opts.max_iterations,
            max_line_search_iterations=opts.max_line_search_iterations)
    elif opts.method == 'LBFGS':
        result = tfp.optimizer.lbfgs_minimize(
            loss_and_gradient,
            initial_position=betavec0,
            tolerance=opts.tolerance,
            num_correction_pairs=opts.num_correction_pairs,
            max_line_search_iterations=opts.max_line_search_iterations,
            max_iterations=opts.max_iterations)

    beta_opt = result.position.reshape(geotargets.shape)

    # get additional return values
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
