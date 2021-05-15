
# This uses the Newton method with jvp and lsq


# %% imports
# import inspect
import importlib

import scipy
from scipy.optimize import line_search
import pickle

import numpy as np
from numpy.linalg import norm # jax??

import jax
import jax.numpy as jnp
from jax import jvp, vjp, jacfwd
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
    'stepmethod': 'jac',  # jvp or jac, jac seems to work better
    'max_iter': 20,
    'linesearch': True,
    'init_p': 0.75,  # less than 1 seems important
    'maxp_tol': .01,  # .01 is 1/100 of 1% for the max % difference from target
    'startup_period': True,  # should we have a separate startup period?
    'startup_imaxpdiff': 1e6,  # if initial maxpdiff is greater than this go into startup mode
    'startup_iter': 8,  # number of iterations for the startup period
    'startup_p': .25,  # p, the step multiplier, in the startup period
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}


# %% main function
def poisson(wh, xmat, geotargets, options=None):
    a = timer()

    options_all = options_defaults.copy()
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = fgp.scale_problem(xmat, geotargets, opts.scale_goal)

    if np.size(opts.init_beta)==1:
        betavec0 = np.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    else:
        betavec0 = opts.init_beta

    bvec = betavec0.copy()
    dw = fgp.jax_get_diff_weights(geotargets)

    # define functions for getting the step: linear operator approach, or full jacobian

    # linear operator approach
    def get_linop(bvec, wh, xmat, geotargets, dw, diffs):
        l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
        l_diffs = jax.jit(l_diffs)
        l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]
        l_vjp = lambda diffs: vjp(l_diffs, bvec)[1](diffs)
        linop = scipy.sparse.linalg.LinearOperator((bvec.size, bvec.size),
            matvec=l_jvp, rmatvec=l_vjp)
        return linop

    def jvp_step(bvec, wh, xmat, geotargets, dw, diffs):
        Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)
        step_results = scipy.optimize.lsq_linear(Jsolver, diffs)
        if not step_results.success: print("Failure in getting step!! Check results carefully.")
        step = step_results.x
        return step

    # full jacobian approach
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

    if opts.stepmethod == 'jvp':
        get_step = jvp_step
    elif opts.stepmethod == 'jac':
        get_step = jac_step

    def getp_one(l2norm, l2norm_prior, step_dir, init_p):
        return init_p

    def getp_simple(l2norm, l2norm_prior, step_dir, init_p):
        # simple halving approach to getting step length
        p = init_p
        max_search = 5
        search = 0
        if l2norm > l2norm_prior:
            print(f'starting line search at count: {count:4}')
            l2n = l2norm.copy()
            l2np = l2norm_prior.copy()
            # ?? start from the best prior result, not the latest
            # bvec_temp = bvec_best.copy()
            # step_dir_temp = step_dir_best
            bvec_temp = bvec.copy()
            step_dir_temp = step_dir
            while (l2n > l2np * 1.0001 or l2n == 1e99) and search <= max_search:
                search += 1
                p = p / 2.0
                bvec_temp2 = bvec_temp - step_dir_temp * p
                diffs_temp = fgp.jax_targets_diff(bvec_temp2, wh, xmat, geotargets, dw)
                l2np = l2n.copy()
                l2n = norm(diffs_temp, 2)
                print(f'...trying new p: {p:10.4f}  l2norm: {l2n: 12.2f}')
                if np.isnan(l2n):
                    l2n = np.float64(1e99)
        return p

    def getp_simple2(l2norm, l2norm_prior, step_dir, init_p):
        # simple halving approach to getting step length
        # use half of the average of this step and prior step????
        p = init_p
        max_search = 5
        search = 0
        if l2norm > l2norm_prior:
            print(f'starting line search at count: {count:4}')
            l2n = l2norm.copy()
            l2np = l2norm_prior.copy()
            # ?? start from the best prior result, not the latest
            # bvec_temp = bvec_best.copy()
            # step_dir_temp = step_dir_best
            bvec_temp = bvec.copy()
            step_dir_temp = (step_dir + step_dir_prior) / 2.0
            while (l2n > l2np or l2n == 1e99) and search <= max_search:
                search += 1
                p = p / 2.0
                bvec_temp2 = bvec_temp - step_dir_temp * p
                diffs_temp = fgp.jax_targets_diff(bvec_temp2, wh, xmat, geotargets, dw)
                l2np = l2n.copy()
                l2n = norm(diffs_temp, 2)
                print(f'...trying new p: {p:10.4f}  l2norm: {l2n: 12.2f}')
                if np.isnan(l2n):
                    l2n = np.float64(1e99)
        return p

    def getp_wolfe(l2norm, l2norm_prior, step_dir, init_p):
      print("obj: ", objfn(diffs))
      (alpha, fc, *all) = line_search(objfn, gradfn, bvec, step_dir)
      print('# of function calls: ', fc)
      print('alpha: ', alpha)
      print('type: ', type(alpha))
      if alpha is None:
          p = 1.0
      else:
          p = alpha
      return p

    objfn = lambda x: fgp.jax_sspd(x, wh, xmat, geotargets, dw)
    gradfn = jax.grad(objfn)


    # determine whether and how to do line searches
    getp = getp_one
    if opts.linesearch:
        getp = getp_simple
        # getp = getp_simple2
        # getp = getp_wolfe


    # begin Newton iterations

    # initial values
    count = 0
    no_improvement_count = 0
    no_improvement_proportion = 1e-3
    maxpdiff = 1e99

    # diffs_prior = 1e99
    l2norm_prior = np.float64(1e99)
    bvec_best = bvec.copy()
    l2norm_best = l2norm_prior.copy()
    step_dir = np.zeros(bvec.shape)

    # before we start, check the initial error
    idiffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    imaxpdiff = jnp.max(jnp.abs(idiffs))

    if opts.startup_period:
        if imaxpdiff > opts.startup_imaxpdiff:
            startup_iter = opts.startup_iter
            startup_p = opts.startup_p
        else:
            startup_iter = 2
            startup_p = .5
    else:
        startup_iter = 0

    # set all stopping conditions to False
    max_iter = False
    low_error = False
    no_improvement = False
    ready_to_stop = False

    print('iteration      l2norm      maxabs_error')
    while not ready_to_stop:
        count += 1

        diffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)
        if l2norm >= l2norm_prior * (1.0 - no_improvement_proportion):
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if l2norm < l2norm_best:
            bvec_best = bvec.copy()
            l2norm_best = l2norm.copy()
            step_dir_best = step_dir.copy()

        maxpdiff = jnp.max(jnp.abs(diffs))
        print(f'{count: 6}   {l2norm: 12.2f}      {maxpdiff: 12.2f}')

        step_dir = get_step(bvec, wh, xmat, geotargets, dw, diffs)

        # make the solver more robust against large initial errors
        if opts.startup_period and count <= startup_iter:
            p = startup_p
        else:
            p = getp(l2norm, l2norm_prior, step_dir, opts.init_p)

        bvec = bvec - step_dir * p
        l2norm_prior = l2norm
        step_dir_prior = step_dir

        # check stopping condition
        message = ''

        if maxpdiff <= opts.maxp_tol:
            low_error = True
            message = message + '  Maximum absolute percent error is sufficiently low.\n'

        if no_improvement_count >= 3:
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
    delta_opt = 'Not reported'  # get_delta(wh, beta_opt, xmat)
    whs_opt = get_whs_logs(beta_opt, wh, xmat, geotargets) # jax_get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = jnp.dot(whs_opt.T, xmat)

    if opts.scaling:
        geotargets_opt = np.multiply(geotargets_opt, scale_factors)

    b = timer()

    print(f'  Elapsed seconds: {b - a: 9.2f}')

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets_opt',
              'beta_opt',
              'delta_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    result = Result(elapsed_seconds=b - a,
                    whs_opt=whs_opt,
                    geotargets_opt=geotargets_opt,
                    beta_opt=beta_opt,
                    delta_opt=delta_opt)

    return result



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
    # if beta_object.ndim == 1:
    #     beta = beta_object.reshape(geotargets.shape)
    # elif beta_object.ndim == 2:
    #     beta = beta_object

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


