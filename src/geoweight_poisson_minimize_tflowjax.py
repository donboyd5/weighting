# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/optimizer/lbfgs_minimize

@author: donbo
"""

# %% imports

import gc

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


# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'objscale': 1.0,
    'maxiter': 100,
    'tolerance': 1e-8,
    'num_correction_pairs': 10,
    'max_line_search_iterations': 50,
    'max_iterations': 100,
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}


# %% poisson - the primary function

def poisson(wh, xmat, geotargets, options=None):
    # TODO: implement options
    a = timer()

    options_all = options_defaults.copy()
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = scale_problem(xmat, geotargets, opts.scale_goal)

    betavec0 = jnp.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    dw = jax_get_diff_weights(geotargets)

    ljax_sspd = lambda bvec: jax_sspd(bvec, wh, xmat, geotargets, dw) * opts.objscale

    result = tfp.optimizer.lbfgs_minimize(
        jax.value_and_grad(ljax_sspd),
        initial_position=betavec0,
        tolerance=opts.tolerance,
        num_correction_pairs=opts.num_correction_pairs,
        max_line_search_iterations=opts.max_line_search_iterations,
        max_iterations=opts.max_iterations)

    # result = tfp.optimizer.bfgs_minimize(
    #     jax.value_and_grad(ljax_sspd),
    #     initial_position=betavec0,
    #     max_iterations=opts.max_iterations,
    #     max_line_search_iterations=opts.max_line_search_iterations,
    #     tolerance=opts.tolerance)

    beta_opt = result.position.reshape(geotargets.shape)

    # result = tfp.optimizer.nelder_mead_minimize(
    #   sqrt_quadratic, initial_vertex=start, func_tolerance=1e-8,
    #   batch_evaluate_objective=True)

    # get additional return values
    delta_opt = 'Not reported'
    whs_opt = get_whs_logs(beta_opt, wh, xmat, geotargets)
    geotargets_opt = jnp.dot(whs_opt.T, xmat)

    if opts.scaling:
        geotargets_opt = jnp.multiply(geotargets_opt, scale_factors)

    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets_opt',
              'beta_opt',
              'delta_opt',
              'result')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 whs_opt=whs_opt,
                 geotargets_opt=geotargets_opt,
                 beta_opt=beta_opt,
                 delta_opt=delta_opt,
                 result=result)

    return res



# %% jax functions
# these functions are used by jax to compute the jacobian
# I have not yet figured out how to avoid having two versions of the functions
# def jax_get_delta(wh, beta, xmat):
#     beta_x = jnp.exp(jnp.dot(beta, xmat.T))
#     delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
#     return delta

def jax_get_diff_weights(geotargets, goal=100):
    goalmat = jnp.full(geotargets.shape, goal)
    # djb note there is no jnp.errstate so I use np.errstate
    # with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
    diff_weights = jnp.where(geotargets != 0, goalmat / geotargets, 1)
    return diff_weights

# def jax_get_geoweights(beta, delta, xmat):
#     """
#     Calculate state-specific weights for each household.

#     Definitions:
#     h: number of households
#     k: number of characteristics each household has
#     s: number of states or geographic areas

#     See (Khitatrakun, Mermin, Francis, 2016, p.4)

#     Parameters
#     ----------
#     beta : matrix
#         s x k matrix of coefficients for the poisson function that generates
#         state weights.
#     delta : vector
#         h-length vector of constants (one per household) for the poisson
#         function that generates state weights.
#     xmat : matrix
#         h x k matrix of characteristics (data) for households.

#     Returns
#     -------
#     matrix of dimension h x s.

#     """
#     # begin by calculating beta_x, an s x h matrix:
#     #   each row has the sum over k of beta[s_i, k] * x[h_j, k]
#     #     for each household where s_i is the state in row i
#     #   each column is a specific household
#     beta_x = jnp.dot(beta, xmat.T)

#     # add the delta vector of household constants to every row
#     # of beta_x and transpose
#     # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
#     beta_xd = (beta_x + delta).T

#     weights = jnp.exp(beta_xd)

#     return weights


# def jax_get_geotargets(beta, wh, xmat):
#     """
#     Calculate matrix of target values by state and characteristic.

#     Returns
#     -------
#     targets_mat : matrix
#         s x k matrix of target values.

#     """
#     delta = jax_get_delta(wh, beta, xmat)
#     whs = jax_get_geoweights(beta, delta, xmat)
#     targets_mat = jnp.dot(whs.T, xmat)
#     return targets_mat


# def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
#     '''
#     Calculate difference between calculated targets and desired targets.

#     Parameters
#     ----------
#     beta_obj: vector or matrix
#         if vector it will have length s x k and we will create s x k matrix
#         if matrix it will be dimension s x k
#         s x k matrix of coefficients for the poisson function that generates
#         state weights.
#     wh: array-like
#         DESCRIPTION.
#     xmat: TYPE
#         DESCRIPTION.
#     geotargets: TYPE
#         DESCRIPTION.
#     diff_weights: TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     matrix of dimension s x k.

#     '''
#     # beta must be a matrix so if beta_object is a vector, reshape it
#     if beta_object.ndim == 1:
#         beta = beta_object.reshape(geotargets.shape)
#     elif beta_object.ndim == 2:
#         beta = beta_object

#     geotargets_calc = jax_get_geotargets(beta, wh, xmat)
#     diffs = geotargets_calc - geotargets
#     # diffs = diffs * diff_weights
#     diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

#     # return a matrix or vector, depending on the shape of beta_object
#     if beta_object.ndim == 1:
#         diffs = diffs.flatten()

#     return diffs

# def jax_sspd(beta_object, wh, xmat, geotargets, diff_weights):
#     diffs = jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights)
#     sspd = jnp.square(diffs).sum()
#     return sspd


# %% scaling
def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = jnp.divide(xmat, scale_factors)
    geotargets = jnp.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors


# %% new functions
def get_whs_logs(beta_object, wh, xmat, geotargets):
    # note beta is an s x k matrix
    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    betax = beta.dot(xmat.T)
    # adjust betax to make exponentiation more stable numerically
    # subtract column-specific constant (the max) from each column of betax
    const = betax.max(axis=0)
    betax = jnp.subtract(betax, const)
    ebetax = jnp.exp(betax)
    # print(ebetax.min())
    # print(np.log(ebetax))
    logdiffs = betax - jnp.log(ebetax.sum(axis=0))
    shares = jnp.exp(logdiffs)
    whs = jnp.multiply(wh, shares).T
    return whs


def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    # geotargets_calc = jax_get_geotargets(beta, wh, xmat)
    whs = get_whs_logs(beta_object, wh, xmat, geotargets)
    geotargets_calc = jnp.dot(whs.T, xmat)
    diffs = geotargets_calc - geotargets
    # diffs = diffs * diff_weights
    diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs

def jax_sspd(beta_object, wh, xmat, geotargets, diff_weights):
    diffs = jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights)
    sspd = jnp.square(diffs).sum()
    return sspd


