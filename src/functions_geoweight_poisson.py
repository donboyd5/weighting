# %% imports
# import numpy as np
# import jax
import jax.numpy as jnp
# import scipy
# from jax import jvp, vjp

# # this next line is CRUCIAL or we will lose precision
from jax.config import config
config.update('jax_enable_x64', True)


# %% functions needed for residuals
def jax_get_delta(wh, beta, xmat):
    beta_x = jnp.exp(jnp.dot(beta, xmat.T))
    delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta


def jax_get_diff_weights(geotargets, goal=100):
    # establish a weight for each target that, prior to application of any
    # other weights, will give each target equal priority
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
    # diffs = diffs * diff_weights
    diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs # np.array(diffs)  # note that this is np, not jnp!


# %% utility functions
def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = jnp.divide(xmat, scale_factors)
    geotargets = jnp.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors


# %% functions related to the jacobian and to Newton steps

# define lambda functions that each will take only one argument, and that
# will otherwise use items in the environment

# l_diffs: lambda function that gets differences from targets, as a vector
#   parameter: a beta vector
#     (In addition, it assumes existence of, and uses, values in the
#      environment: wh, xmat, geotargets, and dw. These do not change
#      within the loop.)
#   returns: vector of differences from targets
# l_diffs = lambda bvec: jax_targets_diff(bvec, wh, xmat, geotargets, dw)

# l_jvp: lambda function that evaluates the following jacobian vector product
#    (the dot product of a jacobian matrix and a vector)
#     matrix:
#      jacobian of l_diffs evaluated at the current bvec values,
#     vector:
#       differences from targets when l_diffs is evaluated at bvec
#   returns: a jacobian-vector-product vector
# This is used, in conjunction with l_vjp, to compute the step vector in the
# Newton method. It allows us to avoid computing the full jacobian, thereby
# saving considerable memory use and computation.
# l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]

# l_vjp: lambda function that evaluates the following vector jacobian product
#    (the dot product of the transpose of a jacobian matrix and a vector)
#     matrix:
#      transpose of jacobian of l_diffs evaluated at the current bvec values,
#     vector:
#       differences from targets when l_diffs is evaluated at bvec
#   returns: a vector-jacobian-product vector
# Used with l_jvp - see above.
# l_vjp = lambda diffs: vjp(l_diffs, bvec)[1](diffs)


# %% linear operator
# In concept, we need the inverse of the jacobian to compute a Newton step.
# However, we are avoiding creating the jacobian because we don't want to use
# that much memory or do such extensive calculations. 
# 
# To calculate the step, we need to solve the system:
#   Ax = b
#
# where:
#   A is the jacobian
#   b is the vector of differences from targets, evaluated at current bvec
#   x, to be solved for, is the next step we will take.
# 
# If we were to create the jacobian (A in this example), we could invert it
# (if invertible). However, we avoid creating the jacobian through use of
# the l_jvp and l_vjp functions. Furthermore, this jacobian often seems 
# extremely ill-conditioned, and so we don't use one of the methods that
# can solve for x iteratively, such as conjugate gradient 
# (jax.scipy.sparse.linalg.cg) or gmres (scipy.sparse.linalg.gmres). I have
# tried these methods and they generally either fail because the jacobian is
# so ill-conditioned, or cannot be used because it is not positive semidefinite.
# 
# Thus, as an alternative to solving for x directly (by constructing a large,
# difficult to calculate, ill-conditioned jacobian) or solving for x iteratively
# (by using an interative method such as cg or gmres), instead we solve for x
# approximately using least squares.
#
# Furthermore, we can solve approximately for x this way without creating the
# jacobian (matrix A in the system notation) by using the two vector-product
# functions l_jvp and l_vjp, and wrapping them in a linear operator. That's
# what this next line does. This lets us solve for approximate x quickly
# and robustly (without numerical problems), with very little memory usage.
# lsq_linop = scipy.sparse.linalg.LinearOperator((betavec0.size, betavec0.size),
#     matvec=l_jvp, rmatvec=l_vjp)

# we use a function to return the linear operator
def get_lsq_linop(bvsize, l_jvp, l_vjp):
    lsq_linop = scipy.sparse.linalg.LinearOperator((bvsize, bvsize),
        matvec=l_jvp, rmatvec=l_vjp)

def get_lsq_linop2(bvec, wh, xmat, geotargets, dw):
    l_diffs = lambda bvec: jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]
    l_vjp = lambda diffs: vjp(l_diffs, bvec)[1](diffs)
    lsq_linop = scipy.sparse.linalg.LinearOperator((bvec.size, bvec.size),
        matvec=l_jvp, rmatvec=l_vjp)




