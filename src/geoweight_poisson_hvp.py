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

# import numpy as jnp
from scipy.optimize import minimize

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

    ljax_sspd = lambda bvec: jax_sspd(bvec, wh, xmat, geotargets, dw) # * opts.objscale   # jax_sspd = jax.jit(jax_sspd)


    # forward-over-reverse
    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f), primals, tangents)[1]

    # hvp(f, (X,), (V,))

    result = minimize(ljax_sspd,
            betavec0,
            method='trust-ncg',
            jac=jax.grad(ljax_sspd),
            hessp=hvp(ljax_sspd),
            options={'gtol': 1e-8, 'disp': True})

    # get additional return values
    beta_opt = result.x.reshape(geotargets.shape)
    delta_opt = jax_get_delta(wh, beta_opt, xmat)
    whs_opt = jax_get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = jax_get_geotargets(beta_opt, wh, xmat)

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
def jax_get_delta(wh, beta, xmat):
    beta_x = jnp.exp(jnp.dot(beta, xmat.T))
    delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta

def jax_get_diff_weights(geotargets, goal=100):
    goalmat = jnp.full(geotargets.shape, goal)
    # djb note there is no jnp.errstate so I use np.errstate
    # with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
    diff_weights = jnp.where(geotargets != 0, goalmat / geotargets, 1)
    return diff_weights

def jax_get_geoweights(beta, delta, xmat):
    """
    Calculate state-specific weights for each household.

    Definitions:
    h: number of households
    k: number of characteristics each household has
    s: number of states or geographic areas

    See (Khitatrakun, Mermin, Francis, 2016, p.4)

    Parameters
    ----------
    beta : matrix
        s x k matrix of coefficients for the poisson function that generates
        state weights.
    delta : vector
        h-length vector of constants (one per household) for the poisson
        function that generates state weights.
    xmat : matrix
        h x k matrix of characteristics (data) for households.

    Returns
    -------
    matrix of dimension h x s.

    """
    # begin by calculating beta_x, an s x h matrix:
    #   each row has the sum over k of beta[s_i, k] * x[h_j, k]
    #     for each household where s_i is the state in row i
    #   each column is a specific household
    beta_x = jnp.dot(beta, xmat.T)

    # add the delta vector of household constants to every row
    # of beta_x and transpose
    # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
    beta_xd = (beta_x + delta).T

    weights = jnp.exp(beta_xd)

    return weights


def jax_get_geotargets(beta, wh, xmat):
    """
    Calculate matrix of target values by state and characteristic.

    Returns
    -------
    targets_mat : matrix
        s x k matrix of target values.

    """
    delta = jax_get_delta(wh, beta, xmat)
    whs = jax_get_geoweights(beta, delta, xmat)
    targets_mat = jnp.dot(whs.T, xmat)
    return targets_mat


def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
    '''
    Calculate difference between calculated targets and desired targets.

    Parameters
    ----------
    beta_obj: vector or matrix
        if vector it will have length s x k and we will create s x k matrix
        if matrix it will be dimension s x k
        s x k matrix of coefficients for the poisson function that generates
        state weights.
    wh: array-like
        DESCRIPTION.
    xmat: TYPE
        DESCRIPTION.
    geotargets: TYPE
        DESCRIPTION.
    diff_weights: TYPE
        DESCRIPTION.

    Returns
    -------
    matrix of dimension s x k.

    '''
    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    geotargets_calc = jax_get_geotargets(beta, wh, xmat)
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


# %% scaling
def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = jnp.divide(xmat, scale_factors)
    geotargets = jnp.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors

