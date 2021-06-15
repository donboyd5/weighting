
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
    'init_beta': 0.0,
    'max_iter': 20,
    'maxp_tol': .01,  # .01 is 1/100 of 1% for the max % difference from target
    'no_improvement_proportion': 1e-3,
    'jac_min_improvement': 0.10,

    'stepmethod': 'auto',
    'jac_threshold': 5,  # try to use jac when rmse is below this
    'base_stepmethod': 'jac',  # jvp or jac, jac seems to work better
    'startup_period': 8,  # # of iterations in startup period (0 means no startup period)
    'startup_stepmethod': 'jvp',  # jac or jvp
    'step_fixed': False,  # False, or a fixed number
    'search_iter': 20,
    'jvp_reset_steps': 5,
    'lgmres_maxiter': 20,
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

    # prepare for Newton iterations
    bvec = betavec0
    dw = fgp.jax_get_diff_weights(geotargets)

    count = 0
    no_improvement_count = 0
    jvpcount = 1e9
    step_method = 'jvp'
    get_step = jvp_step

    # construct initial values, pre-iteration
    diffs = fgp.jax_targets_diff_copy(bvec, wh, xmat, geotargets, dw)
    l2norm = norm(diffs, 2)
    maxpdiff = jnp.max(jnp.abs(diffs))
    rmse = math.sqrt(l2norm**2 / bvec.size)
    rmsexmax = math.sqrt((l2norm**2 - maxpdiff**2) / (bvec.size - 1))

    # define initial best values
    iter_best = 0
    l2norm_prior = l2norm.copy()
    l2norm_best = l2norm.copy()
    bvec_best = bvec.copy()

    # set all stopping conditions to False
    max_iter = False
    low_error = False
    no_improvement = False
    NO_IMPROVEMENT_MAX = 2
    ready_to_stop = False

    print('Starting Newton iterations...\n')
    print('                 max abs                            ')
    print('                    %             --  step --   --  # seconds  --')
    print(' iter   l2norm    error    rmse   method size   step search  total\n')

    # print stats at start
    print(f"{0: 4} {l2norm: 9.2f} {maxpdiff: 8.2f} {rmse: 7.2f}")

    while not ready_to_stop:
        count += 1
        iter_start = timer()

        if opts.stepmethod == 'auto':
            # to use jac, we must be under the threshold and either
            #    lower than the previous step, or
            #    there have been 5 prior jac steps
            # at this point l2norm is the value from the prior iteration
            if rmse < opts.jac_threshold:
                if step_method == 'jvp' and \
                    jvpcount >= (opts.jvp_reset_steps - 1):
                    step_method = 'jac'
                    get_step = jac_step
                    jvpcount = 0
                elif step_method == 'jvp' and \
                    jvpcount < (opts.jvp_reset_steps - 1):
                    step_method = 'jvp'
                    get_step = jvp_step
                    jvpcount += 1
                elif step_method == 'jac' and \
                    l2norm < (l2norm_prior * (1 - opts.jac_min_improvement)):
                    step_method = 'jac'
                    get_step = jac_step
                elif step_method == 'jac' and\
                    l2norm >= (l2norm_prior * (1 - opts.jac_min_improvement)):
                    step_method = 'jvp'
                    get_step = jvp_step
                    jvpcount = 0
                else:
                    print("WE SHOULD NOT GET HERE!")
        else:
            step_method = opts.base_stepmethod
            if opts.base_stepmethod == 'jvp':
                get_step = jvp_step
            elif opts.base_stepmethod == 'jac':
                get_step = jac_step

        # NOW we can set l2norm_prior and calculate a new l2norm
        l2norm_prior = l2norm

        # get step direction and step size
        step_start = timer()
        step_dir = get_step(bvec, wh, xmat, geotargets, dw, diffs, opts)
        step_end = timer()

        search_start = timer()
        if opts.step_fixed is False:
            p = getp_min(bvec, step_dir, wh, xmat, geotargets, dw, opts.search_iter, l2norm_prior)
            # p = getp_reduce(bvec, step_dir, wh, xmat, geotargets, dw, opts.search_iter, l2norm_prior)
        else: p = opts.step_fixed
        search_end = timer()

        bvec = bvec - step_dir * p
        diffs = fgp.jax_targets_diff_copy(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)
        maxpdiff = jnp.max(jnp.abs(diffs))
        rmse = math.sqrt(l2norm**2 / bvec.size)
        rmsexmax = math.sqrt((l2norm**2 - maxpdiff**2) / (bvec.size - 1))

        iter_end = timer()

        step_time = step_end - step_start
        search_time = search_end - search_start
        itime = iter_end - iter_start

        print(f'{count: 4} {l2norm: 9.2f} {maxpdiff: 8.2f} {rmse: 7.2f}   {step_method}  {p: 6.3f} {step_time: 6.2f} {search_time: 6.2f} {itime: 6.2f}')

        if l2norm >= l2norm_prior * (1.0 - opts.no_improvement_proportion) and step_method == 'jvp':
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if l2norm <= l2norm_best:
            iter_best = count
            bvec_best = bvec.copy()
            l2norm_best = l2norm.copy()
        else:
            # if this isn't the best iteration, prepare to reset
            bvec = bvec_best.copy()

        # check stopping conditions
        message = ''

        if maxpdiff <= opts.maxp_tol:
            low_error = True
            message = message + '  Maximum absolute percent error is sufficiently low.\n'

        if no_improvement_count >= NO_IMPROVEMENT_MAX:
            no_improvement = True
            message = message + '  l2norm no longer improving.\n'

        if count >= opts.max_iter:
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

def jvp_step_good(bvec, wh, xmat, geotargets, dw, diffs, opts):
    Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)

    # step, info = scipy.sparse.linalg.cg(Jsolver, diffs, maxiter=25)
    step_results = scipy.optimize.lsq_linear(Jsolver, diffs, max_iter=5) # max_iter None default
    if not step_results.success: print("Failure in getting step!! Check results carefully.")
    step = step_results.x
    return step

def jvp_step(bvec, wh, xmat, geotargets, dw, diffs, opts):
    Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html
    # Construct a linear operator that computes P^-1 * x.
    # M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
    # scipy.sparse.linalg = spla.LinearOperator((bvec.size, bvec.size), M_x)

    step, info = scipy.sparse.linalg.lgmres(Jsolver, diffs, maxiter=opts.lgmres_maxiter)
    # print("info (0 is good): ", info)
    # step_results = scipy.optimize.lsq_linear(Jsolver, diffs, max_iter=5) # max_iter None default
    # if not step_results.success: print("Failure in getting step!! Check results carefully.")
    # step = step_results.x
    return step

def jvp_step2a(bvec, wh, xmat, geotargets, dw, diffs, opts):
    # jax.scipy.sparse.linalg.gmres(A, b, x0=None, *, tol=1e-05, atol=0.0,
    #   restart=20, maxiter=None, M=None, solve_method='batched')

    l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    l_diffs = jax.jit(l_diffs)
    l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]
    l_jvp = jax.jit(l_jvp)

    # step, info = scipy.sparse.linalg.cg(Jsolver, diffs, maxiter=25)
    # step, info = jax.scipy.sparse.linalg.gmres(l_jvp, diffs, solve_method='incremental')
    step, info = jax.scipy.sparse.linalg.cg(l_jvp, diffs)
    print(step)
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
    # print(np.round(jacmat[0:10,0:5], 4))
    return jacmat

def jac_step_M(bvec, wh, xmat, geotargets, dw, diffs, options):
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    # step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]
    M = np.diagflat(np.diag(jacmat))
    # print("using M")
    # M = 1 / M

    step, info = scipy.sparse.linalg.lgmres(jacmat, diffs, M = M, maxiter=options.lgmres_maxiter)

    # jinv = jnp.linalg.pinv(jacmat)
    # jinv = scipy.linalg.pinv(jacmat)
    # step = jnp.dot(jinv, diffs)

    return step

def jac_step(bvec, wh, xmat, geotargets, dw, diffs, options):
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    # step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]

    jinv = jnp.linalg.pinv(jacmat)
    # jinv = scipy.linalg.pinv(jacmat)
    step = jnp.dot(jinv, diffs)

    return step


def jac_step_lu(bvec, wh, xmat, geotargets, dw, diffs, options):
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    # step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]
    # create preconditioner
    sJ = scipy.sparse.csc_matrix(jacmat)
    sJ_LU = scipy.sparse.linalg.splu(sJ)
    M = scipy.sparse.linalg.LinearOperator((bvec.size,bvec.size), sJ_LU.solve)
    step, info = scipy.sparse.linalg.lgmres(sJ, diffs, M=M)

    # jinv = jnp.linalg.pinv(jacmat)
    # jinv = scipy.linalg.pinv(jacmat)
    # step = jnp.dot(jinv, diffs)

    return step


# %% functions related to line search

def getp_min(bvec, step_dir, wh, xmat, geotargets, dw, search_iter, l2norm_prior):

    def get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw):
        bvec = bvec - step_dir * p
        diffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)
        return l2norm

    # p = 1.0
    # l2norm = get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw)
    # if l2norm < l2norm_prior:
    #     return p

    res = minimize_scalar(get_norm, bounds=(0, 1), args=(bvec, step_dir, wh, xmat, geotargets, dw),
        method='bounded', options={'maxiter': search_iter})

    p = res.x
    if get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw) > l2norm_prior:
        p = 0

    return p

def getp_reduce(bvec, step_dir, wh, xmat, geotargets, dw, search_iter, l2norm_prior):

    def get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw):
        bvec = bvec - step_dir * p
        diffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)
        return l2norm

    p = 1.0
    count = 0
    l2norm = get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw)
    while count < 20 and l2norm > l2norm_prior:
        count += 1
        p = p * .8

    if l2norm > l2norm_prior:
        p = 0

    return p



