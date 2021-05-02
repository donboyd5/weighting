
# This uses the Newton method with jvp and lsq




# %% imports
# import inspect
import scipy

import numpy as np
import jax
import jax.numpy as jnp
from timeit import default_timer as timer

from jax import jvp, vjp

# this next line is CRUCIAL or we will lose precision
from jax.config import config
config.update('jax_enable_x64', True)

from timeit import default_timer as timer

from collections import namedtuple

import utilities as ut # src.utilities
# import make_test_problems as mtp  # src.make_test_problems


# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 1e3,
    'init_beta': 0.5,
    'jacmethod': 'jvp',  # vjp, jvp, full, finite-diff
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}




# %% functions needed for residuals
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
    delta = jax_get_delta(wh, beta, xmat)
    whs = jax_get_geoweights(beta, delta, xmat)
    targets_mat = jnp.dot(whs.T, xmat)
    return targets_mat    


def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    geotargets_calc = jax_get_geotargets(beta, wh, xmat)
    diffs = geotargets_calc - geotargets
    diffs = diffs * diff_weights

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs # np.array(diffs)  # note that this is np, not jnp!


# %% utility functions
def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = np.divide(xmat, scale_factors)
    geotargets = np.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors

# %% main function
def poisson_newtjvplsq(wh, xmat, geotargets, options=None):
    # TODO: implement options
    a = timer()

    options_all = options_defaults.copy()
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = scale_problem(xmat, geotargets, opts.scale_goal)

    # betavec0 = np.zeros(geotargets.size)
    betavec0 = np.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    dw = get_diff_weights(geotargets)

    # define the different functions that can be used to construct the jacobian
    # these are alternative to each other - we'll only use one
    # I have not succeeded in moving them outside of the poisson function

    
    # if opts.jacmethod == 'jvp':
    #     jax_jacobian_basic = jax.jit(jac_jvp(jax_targets_diff))  # jax_jacobian_basic is a function -- the jax jacobian
    # elif opts.jacmethod == 'vjp':
    #     jax_jacobian_basic = jax.jit(jac_vjp(jax_targets_diff))
    # elif opts.jacmethod == 'full':
    #     jax_jacobian_basic = jax.jit(jax.jacfwd(jax_targets_diff))        
    # else:
    #     jax_jacobian_basic = None



    def jax_jacobian(beta, wh, xmat, geotargets, dw):
        jac_values = jax_jacobian_basic(beta, wh, xmat, geotargets, dw)
        jac_values = np.array(jac_values).reshape((dw.size, dw.size))
        return jac_values

    # jax_jacobian_basic = jax.jit(jac_jvp(jax_targets_diff))  # jax_jacobian_basic is a function -- the jax jacobian
    if opts.jacmethod == 'findiff':
        jacmethod = '2-point'
    else:
        jacmethod = jax_jacobian



    # get return values
    beta_opt = spo_result.x.reshape(geotargets.shape)
    delta_opt = get_delta(wh, beta_opt, xmat)
    whs_opt = get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = get_geotargets(beta_opt, wh, xmat)

    if opts.scaling:
        geotargets_opt = np.multiply(geotargets_opt, scale_factors)

    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets_opt',
              'beta_opt',
              'delta_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 whs_opt=whs_opt,
                 geotargets_opt=geotargets_opt,
                 beta_opt=beta_opt,
                 delta_opt=delta_opt)

    return res

