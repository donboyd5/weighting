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


# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'stepmethod': 'jvp',  # vjp, jvp, full, finite-diff
    'max_nfev': 100,
    'ftol': 1e-7,
    'x_scale': 'jac',
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}


# %% poisson - the primary function

def poisson(wh, xmat, geotargets, options=None):
    print('test 16')
    a = timer()

    options_all = options_defaults
    options_all.update(options)
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

    # CAUTION: linear operator approach does NOT work well because scipy least_squares does not allow the option x_scale='jac' when using a linear operator
    # This is fast and COULD be very good if a good scaling vector is developed but without that it iterates quickly but reduces
    # cost very slowly.

    # jax_jacobian_basic = jax.jit(jac_jvp(jax_targets_diff))  # jax_jacobian_basic is a function -- the jax jacobian
    if opts.stepmethod == 'findiff':
        stepmethod = '2-point'
    elif opts.stepmethod == 'jvp-linop':
        stepmethod = fgp.jvp_linop  # CAUTION: this method does not allow x_scale='jac' and reduces costs slowly
    else:
        stepmethod = jax_jacobian

    spo_result = spo.least_squares(
        fun=targets_diff,
        x0=betavec0,
        method='trf', jac=stepmethod, verbose=2,
        ftol=opts.ftol, xtol=1e-7,
        x_scale=opts.x_scale,
        loss='soft_l1',  # linear, soft_l1, huber, cauchy, arctan,
        max_nfev=opts.max_nfev,
        args=(wh, xmat, geotargets, dw))

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


# %% functions
# def get_delta(wh, beta, xmat):
#     """Get vector of constants, 1 per household.

#     See (Khitatrakun, Mermin, Francis, 2016, p.5)

#     Note: beta %*% xmat can get very large!! in which case or exp will be Inf.
#     It will get large when a beta element times an xmat element is large,
#     so either beta or xmat can be the problem.

#     In R the problem will bomb but with numpy it appears to recover
#     gracefully.

#     According to https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
#       For most practical purposes, you can probably approximate
#         1 / (1 + <a large number>) to zero. That is to say, just ignore the
#       warning and move on. Numpy takes care of the approximation for
#       you (when using np.float64).

#     This will generate runtime warnings of overflow or divide by zero.
#     """
#     # print("before betax")
#     # print(np.quantile(beta, [0, .1, .25, .5, .75, .9, 1]))

#     # import pickle
#     # save_list = [beta, xmat]
#     # save_name = '/home/donboyd/Documents/beta_xmat.pkl'
#     # open_file = open(save_name, "wb")
#     # pickle.dump(save_list, open_file)
#     # open_file.close()

#     beta_x = np.exp(np.dot(beta, xmat.T))
#     # print("after betax")

#     # beta_x[beta_x == 0] = 0.1  # experimental
#     # beta_x[np.isnan(beta_x)] = 0.1

#     delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
#     # print(delta)
#     # delta[delta == 0] = 0.1  # experimental
#     # delta[np.isnan(delta)] = 0.1
#     return delta


# def get_diff_weights(geotargets, goal=100):
#     """
#     difference weights - a weight to be applied to each target in the
#       difference function so that it hits its goal
#       set the weight to 1 if the target value is zero

#     do this in a vectorized way
#     """

#     # avoid divide by zero or other problems

#     # numerator = np.full(geotargets.shape, goal)
#     # with np.errstate(divide='ignore'):
#     #     dw = numerator / geotargets
#     #     dw[geotargets == 0] = 1

#     goalmat = np.full(geotargets.shape, goal)
#     with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
#         diff_weights = np.where(geotargets != 0, goalmat / geotargets, 1)

#     return diff_weights


# def get_geotargets(beta, wh, xmat):
#     """
#     Calculate matrix of target values by state and characteristic.

#     Returns
#     -------
#     targets_mat : matrix
#         s x k matrix of target values.

#     """
#     delta = get_delta(wh, beta, xmat)
#     whs = get_geoweights(beta, delta, xmat)
#     targets_mat = np.dot(whs.T, xmat)
#     return targets_mat


# def get_geoweights(beta, delta, xmat):
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
#     beta_x = np.dot(beta, xmat.T)

#     # add the delta vector of household constants to every row
#     # of beta_x and transpose
#     # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
#     beta_xd = (beta_x + delta).T

#     weights = np.exp(beta_xd)

#     return weights


# def targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
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

#     geotargets_calc = get_geotargets(beta, wh, xmat)
#     diffs = geotargets_calc - geotargets
#     # diffs = diffs * diff_weights
#     diffs = np.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

#     # return a matrix or vector, depending on the shape of beta_object
#     if beta_object.ndim == 1:
#         diffs = diffs.flatten()
#     return diffs


def targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
    # beta must be a matrix so if beta_object is a vector, reshape it
    # if beta_object.ndim == 1:
    #     beta = beta_object.reshape(geotargets.shape)
    # elif beta_object.ndim == 2:
    #     beta = beta_object

    # geotargets_calc = jax_get_geotargets(beta, wh, xmat)
    whs = fgp.get_whs_logs(beta_object, wh, xmat, geotargets)
    geotargets_calc = np.dot(whs.T, xmat)
    diffs = geotargets_calc - geotargets
    # diffs = diffs * diff_weights
    diffs = np.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs


# %% jax functions
# these functions are used by jax to compute the jacobian
# I have not yet figured out how to avoid having two versions of the functions
# def jax_get_delta(wh, beta, xmat):
#     beta_x = jnp.exp(jnp.dot(beta, xmat.T))
#     delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
#     return delta

# def jax_get_diff_weights(geotargets, goal=100):
#     goalmat = jnp.full(geotargets.shape, goal)
#     # djb note there is no jnp.errstate so I use np.errstate
#     with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
#         diff_weights = jnp.where(geotargets != 0, goalmat / geotargets, 1)
#     return diff_weights

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


# %% scaling
# def scale_problem(xmat, geotargets, scale_goal):
#     scale_factors = xmat.sum(axis=0) / scale_goal
#     xmat = jnp.divide(xmat, scale_factors)
#     geotargets = jnp.divide(geotargets, scale_factors)
#     return xmat, geotargets, scale_factors


# # %% new functions
# def get_whs_logs(beta_object, wh, xmat, geotargets):
#     # note beta is an s x k matrix
#     # beta must be a matrix so if beta_object is a vector, reshape it
#     if beta_object.ndim == 1:
#         beta = beta_object.reshape(geotargets.shape)
#     elif beta_object.ndim == 2:
#         beta = beta_object

#     betax = beta.dot(xmat.T)
#     # adjust betax to make exponentiation more stable numerically
#     # subtract column-specific constant (the max) from each column of betax
#     const = betax.max(axis=0)
#     betax = jnp.subtract(betax, const)
#     ebetax = jnp.exp(betax)
#     # print(ebetax.min())
#     # print(np.log(ebetax))
#     logdiffs = betax - jnp.log(ebetax.sum(axis=0))
#     shares = jnp.exp(logdiffs)
#     whs = jnp.multiply(wh, shares).T
#     return whs


# def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
#     # print(np.quantile(jnp.asarray(beta_object), q=np.array([0., .1, .5, .9, 1.])))
#     # beta must be a matrix so if beta_object is a vector, reshape it
#     if beta_object.ndim == 1:
#         beta = beta_object.reshape(geotargets.shape)
#     elif beta_object.ndim == 2:
#         beta = beta_object

#     # geotargets_calc = jax_get_geotargets(beta, wh, xmat)
#     whs = get_whs_logs(beta_object, wh, xmat, geotargets)
#     geotargets_calc = jnp.dot(whs.T, xmat)
#     diffs = geotargets_calc - geotargets
#     # diffs = diffs * diff_weights
#     diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

#     # return a matrix or vector, depending on the shape of beta_object
#     if beta_object.ndim == 1:
#         diffs = diffs.flatten()

#     return diffs

# # %% jax jacobian functions


# # define the different functions that can be used to construct the jacobian
# # these are alternative to each other - we'll only use one
# def jac_vjp(g, wh, xmat, geotargets, dw):
#     # build jacobian row by row to conserve memory use
#     f = lambda x: g(x, wh, xmat, geotargets, dw)
#     def jacfun(x, wh, xmat, geotargets, dw):
#         y, _vjp = jax.vjp(f, x)
#         Jt, = jax.vmap(_vjp, in_axes=0)(jnp.eye(len(y)))
#         return jnp.transpose(Jt)
#     return jacfun

# def jac_jvp(g, wh, xmat, geotargets, dw):
#     # build jacobian column by column to conserve memory use
#     f = lambda x: g(x, wh, xmat, geotargets, dw)
#     def jacfun(x, wh, xmat, geotargets, dw):
#         _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
#         gc.collect()
#         Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
#         return jnp.transpose(Jt)
#     return jacfun


# def jvp_linop(beta, wh, xmat, geotargets, dw):
#     # linear operator approach
#     # CAUTION: This does NOT work well because scipy least_squares does not allow the option x_scale='jac' when using a linear operator
#     # This is fast and COULD be very good if a good scaling vector is developed but without that it iterates quickly but reduces
#     # cost very slowly.
#     l_diffs = lambda beta: jax_targets_diff(beta, wh, xmat, geotargets, dw)
#     # l_diffs = jax.jit(l_diffs)  # jit is slower
#     # l_jvp = lambda diffs: jax.jvp(l_diffs, (beta,), (diffs,))[1]  # need to reshape
#     l_vjp = lambda diffs: jax.vjp(l_diffs, beta)[1](diffs)

#     def f_jvp(diffs):
#         diffs = diffs.reshape(diffs.size)
#         return jax.jvp(l_diffs, (beta,), (diffs,))[1]
#     # f_jvp = jax.jit(f_jvp)  # jit is slower

#     linop = scipy.sparse.linalg.LinearOperator((beta.size, beta.size),
#         matvec=f_jvp, rmatvec=l_vjp)
#     return linop


# # def jac_jvpgc(g):
# #     # build jacobian column by column to conserve memory use
# #     # with garbage collection after each column
# #     f = lambda x: g(x, wh, xmat, geotargets, dw)
# #     def jacfun(x, wh, xmat, geotargets, dw):
# #         def _jvp(s):
# #             gc.collect()
# #             return jax.jvp(f, (x,), (s,))[1]
# #         Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
# #         return jnp.transpose(Jt)
# #     return jacfun
