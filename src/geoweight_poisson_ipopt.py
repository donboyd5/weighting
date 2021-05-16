# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

@author: donbo
"""

# %% imports

import importlib

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
    def __init__(self, f, g, h, quiet=True):
        self.f = f
        self.g = g
        self.h = h
        self.quiet = quiet

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return self.f(x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return self.g(x)

    def hessian(self, x, lagrange, obj_factor):
        H = self.h(x)
        return obj_factor*H

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
            if iter_count <= 10 or (iter_count % 10) == 0:
                print(f'{"":5} {iter_count:5d} {"":10} {obj_value:8.4e} {"":10} {inf_pr:8.4e}')


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
        problem_obj=ipprob(ljax_sspd, g, h, opts.quiet))

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
