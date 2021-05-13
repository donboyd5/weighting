# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

@author: donbo
"""

# %% imports

import importlib
import gc

import jax
import jax.numpy as jnp

import cyipopt as cy
from cyipopt import minimize_ipopt

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
user_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'objgoal': 100,
    'quiet': True}

ipopts = {
    'print_level': 0,
    'file_print_level': 5,
    'max_iter': 100,
    'linear_solver': 'ma86',
    'print_user_options': 'yes'
}

options_defaults = {**ipopts, **user_defaults}


# options_defaults = {**solver_defaults, **user_defaults}

# %% problem class
class ipprob:
    def __init__(self, f, g, h):
        self.f = f
        self.g = g
        self.h = h

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return self.f(x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return self.g(x)

    def hessian(self, x, lagrange, obj_factor):
        H = self.h(x)
        return obj_factor*H


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
    dw = fgp.jax_get_diff_weights(geotargets)
    # jax_sspd = jax.jit(jax_sspd)
    ljax_sspd = lambda bvec: fgp.jax_sspd(bvec, wh, xmat, geotargets, dw)
    ljax_sspd = jax.jit(ljax_sspd)

    g = jax.grad(ljax_sspd)
    g = jax.jit(g)
    h = jax.hessian(ljax_sspd)
    h = jax.jit(h)

    nlp = cy.Problem(
        n=len(betavec0),
        m=0,
        problem_obj=ipprob(ljax_sspd, g, h))

    for option, value in opts.ipopts.items():
        nlp.add_option(option, value)

    x, result = nlp.solve(betavec0)

    # cyipopt.Problem.jacobian() and cyipopt.Problem.hessian() methods should return the non-zero values
    # of the respective matrices as flattened arrays. The hessian should return a flattened lower
    # triangular matrix. The Jacobian and Hessian can be dense or sparse
    # cyipopt.minimize_ipopt(fun, x0, args=(), kwargs=None, method=None, jac=None,
    #   hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)[source]¶

    # result = minimize_ipopt(ljax_sspd, betavec0, jac=g, options=opts.ipopts)

    # get return values
    beta_opt = x.reshape(geotargets.shape)
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
        s x k matrix of coefficients for the poisson fun.ction that generates
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

# %% reweight class
class RW:

    def __init__(self, cc, n, quiet):
        self.cc = cc  # is this making an unnecessary copy??
        self.jstruct = np.nonzero(cc.T)
        # consider sps.find as possibly faster than np.nonzero, not sure
        self.jnz = cc.T[self.jstruct]
        # self.jnz = sps.find(cc)[2]

        hidx = np.arange(0, n, dtype='int64')
        self.hstruct = (hidx, hidx)
        self.hnz = np.full(n, 2)

        self.quiet = quiet

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return np.sum((x - 1)**2)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return 2*x - 2

    def constraints(self, x):
        """Returns the constraints."""
        # np.dot(x, self.cc)  # dense calculation
        # self.cc.T.dot(x)  # sparse calculation
        return np.dot(x, self.cc)

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return self.jnz

    def jacobianstructure(self):
        """ Define sparse structure of Jacobian. """
        return self.jstruct

    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        return obj_factor * self.hnz

    def hessianstructure(self):
        """ Define sparse structure of Hessian. """
        return self.hstruct

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        if(not self.quiet):
            print(f'{"":10} {iter_count:5d} {"":15} {obj_value:13.7e} {"":15} {inf_pr:13.7e}')

