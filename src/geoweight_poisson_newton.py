
# This uses the Newton method with jac, or jvp and linear operator


# %% imports
# import inspect
import importlib

import math
import scipy
from scipy.optimize import line_search
from scipy.optimize import minimize_scalar

import numpy as np
from numpy.linalg import norm # jax??

import jax
import jax.numpy as jnp
from jax import jvp, vjp
# this next line is CRUCIAL or we will lose precision
from jax.config import config
config.update('jax_enable_x64', True)

from timeit import default_timer as timer
from collections import namedtuple

import src.utilities as ut
import src.functions_geoweight_poisson as fgp
# import make_test_problems as mtp  # src.make_test_problems


# %% reimports
importlib.reload(fgp)


# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!!
    'init_beta': 0.5,
    'max_iter': 20,
    'maxp_tol': .01,  # .01 is 1/100 of 1% for the max % difference from target

    'base_stepmethod': 'jac',  # jvp or jac, jac seems to work better
    'startup_period': 8,  # # of iterations in startup period (0 means no startup period)
    'startup_stepmethod': 'jvp',  # jac or jvp
    'search_iter': 10,
    'jvp_reset_steps': 5,
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}


# %% main function
def poisson(wh, xmat, geotargets, options=None):
    a = timer()

    # override default options with user options, where appropriate
    options_all = options_defaults
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = fgp.scale_problem(xmat, geotargets, opts.scale_goal)

    if np.size(opts.init_beta)==1:
        betavec0 = np.full(geotargets.size, opts.init_beta)
    else:
        betavec0 = opts.init_beta

    if opts.base_stepmethod == 'jvp':
        base_step = jvp_step
    elif opts.base_stepmethod == 'jac':
        base_step = jac_step

    if opts.startup_stepmethod == 'jvp':
        startup_step = jvp_step
    elif opts.startup_stepmethod == 'jac':
        startup_step = jac_step


    # prepare for Newton iterations
    bvec = betavec0
    dw = fgp.jax_get_diff_weights(geotargets)

    count = 0
    no_improvement_count = 0
    no_improvement_proportion = 1e-3
    jvpcount = 0
    step_reset = False

    # construct initial values, pre-iteration
    diffs = fgp.jax_targets_diff_copy(bvec, wh, xmat, geotargets, dw)
    l2norm = norm(diffs, 2)
    maxpdiff = jnp.max(jnp.abs(diffs))
    rmse = math.sqrt((l2norm**2 - maxpdiff**2) / (bvec.size - 1))

    # define initial best values
    iter_best = 0
    l2norm_prior = np.float64(1e99)
    l2norm_best = l2norm_prior.copy()
    bvec_best = bvec.copy()
    l2norm_best = l2norm_prior.copy()

    # set all stopping conditions to False
    max_iter = False
    low_error = False
    no_improvement = False
    NO_IMPROVEMENT_MAX = 3
    ready_to_stop = False

    print('Starting Newton iterations...')
    print('                   rmse   max abs   step     step     step    search    total')
    print(' iter   l2norm    ex-max  % error  method  size (p)  seconds  seconds  seconds\n')

    # print stats at start
    print(f"{0: 4} {l2norm: 10.2f} {rmse: 8.2f} {maxpdiff: 8.2f}")

    while not ready_to_stop:
        count += 1
        iter_start = timer()

        # define stepmethod and default stepsize based on whether we are in startup period
        if count == (opts.startup_period + 1):
            print(f'Startup period ended. New stepmethod is {opts.base_stepmethod}')

        if count <= opts.startup_period:
            step_method = opts.startup_stepmethod
            get_step = startup_step
        else:
            if opts.base_stepmethod == 'jvp':
                step_method = 'jvp'
                get_step = jvp_step
            elif not step_reset:
                step_method = opts.base_stepmethod
                get_step = base_step
            elif step_reset and jvpcount <= opts.jvp_reset_steps:
                if(jvpcount ==0):
                    print(f'l2norm was worse than best prior so resetting stepmethod to jvp for {opts.jvp_reset_steps: 2}')
                    no_improvement_count = 0
                step_method = 'jvp'
                get_step = jvp_step
                jvpcount += 1
            elif step_reset and jvpcount >= jvp_reset_steps:
                # we should only be here for one iteration
                print("resetting step method to base stepmethod from jvp")
                step_method = opts.base_stepmethod
                get_step = base_step
                step_reset = False
                jvpcount = 0
                # no_improvement_count = 0
            else:
                print("WE SHOULD NOT BE HERE!")

        # get step direction and step size
        step_start = timer()
        step_dir = get_step(bvec, wh, xmat, geotargets, dw, diffs)
        step_end = timer()

        search_start = timer()
        p = getp_min(bvec, step_dir, wh, xmat, geotargets, dw, opts.search_iter)
        search_end = timer()

        bvec = bvec - step_dir * p
        diffs = fgp.jax_targets_diff_copy(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)
        maxpdiff = jnp.max(jnp.abs(diffs))
        rmse = math.sqrt((l2norm**2 - maxpdiff**2) / (bvec.size - 1))

        iter_end = timer()

        step_time = step_end - step_start
        search_time = search_end - search_start
        itime = iter_end - iter_start

        print(f'{count: 4} {l2norm: 10.2f} {rmse: 8.2f} {maxpdiff: 8.2f}    {step_method}    {p: 6.3f}  {step_time: 6.2f}   {search_time: 6.2f}    {itime: 6.2f}')

        if l2norm >= l2norm_prior * (1.0 - no_improvement_proportion):
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if l2norm < l2norm_best:
            iter_best = count
            bvec_best = bvec.copy()
            l2norm_best = l2norm.copy()
        else:
            # if this isn't the best iteration, reset the jvp counter
            bvec = bvec_best.copy()
            l2norm = l2norm_best.copy()
            step_reset = True

        l2norm_prior = l2norm

        # check stopping conditions
        message = ''

        if maxpdiff <= opts.maxp_tol:
            low_error = True
            message = message + '  Maximum absolute percent error is sufficiently low.\n'

        if no_improvement_count >= NO_IMPROVEMENT_MAX:
            no_improvement = True
            message = message + '  l2norm no longer improving.\n'

        if count > opts.max_iter:
            max_iter = True
            message = message + '  Maximum number of iterations exceeded.\n'

        ready_to_stop = max_iter or low_error or no_improvement


    print(f'\nDone with Newton iterations:')
    print(message)

    # get return values
    beta_opt = bvec_best.reshape(geotargets.shape)
    whs_opt = fgp.get_whs_logs(beta_opt, wh, xmat, geotargets) # jax_get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = jnp.dot(whs_opt.T, xmat)

    if opts.scaling:
        geotargets_opt = np.multiply(geotargets_opt, scale_factors)

    b = timer()

    print(f'  Elapsed seconds: {b - a: 9.2f}')
    print(f'  Using results from iteration # {iter_best}, with best l2norm: {l2norm_best: 12.2f}')

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets_opt',
              'beta_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    result = Result(elapsed_seconds=b - a,
                    whs_opt=whs_opt,
                    geotargets_opt=geotargets_opt,
                    beta_opt=beta_opt)

    return result


# %% functions for step computation
# %% ..jvp linear operator approach

def get_linop(bvec, wh, xmat, geotargets, dw, diffs):
    l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    l_diffs = jax.jit(l_diffs)
    l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]
    l_vjp = lambda diffs: vjp(l_diffs, bvec)[1](diffs)
    l_jvp = jax.jit(l_jvp)
    l_vjp = jax.jit(l_vjp)
    linop = scipy.sparse.linalg.LinearOperator((bvec.size, bvec.size),
        matvec=l_jvp, rmatvec=l_vjp)
    return linop

def jvp_step(bvec, wh, xmat, geotargets, dw, diffs):
    Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)
    step_results = scipy.optimize.lsq_linear(Jsolver, diffs)
    if not step_results.success: print("Failure in getting step!! Check results carefully.")
    step = step_results.x
    return step

# def jvp_step(bvec, wh, xmat, geotargets, dw, diffs):
#      ALL at once approach to jvp, has not worked
#     a = timer()
#     l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
#     l_diffs = jax.jit(l_diffs)
#     l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]
#     # Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)
#     b = timer()
#     print("get linop time: ", b - a)
#     a = timer()
#     # step_results = scipy.optimize.lsq_linear(l_jvp, diffs)
#     step_results, info = jax.scipy.sparse.linalg.cg(l_jvp, diffs)
#     # print(type(step_results))
#     # max_iter
#     b = timer()
#     print("solve linop time", b - a)
#     # if not step_results.success: print("Failure in getting step!! Check results carefully.")
#     step = step_results
#     # print("nit: ", step_results.nit)
#     # print("step: ", step)
#     return step


# %% ..full jacobian approach
def get_jac(bvec, wh, xmat, geotargets, dw):
    jacfn = jax.jacfwd(fgp.jax_targets_diff)
    jacfn = jax.jit(jacfn)
    jacmat = jacfn(bvec, wh, xmat, geotargets, dw)
    jacmat = np.array(jacmat).reshape((bvec.size, bvec.size))
    return jacmat

def jac_step(bvec, wh, xmat, geotargets, dw, diffs):
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]
    return step

# %% functions related to line search

def getp_min(bvec, step_dir, wh, xmat, geotargets, dw, search_iter):

    def get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw):
        bvec = bvec - step_dir * p
        diffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)
        return l2norm

    res = minimize_scalar(get_norm, bounds=(0, 1), args=(bvec, step_dir, wh, xmat, geotargets, dw),
        method='bounded', options={'maxiter': search_iter})

    return res.x
