
# This uses the Newton method with jac, or jvp and linear operator


# %% imports
# import inspect
import importlib

import sys
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
import src.scipy_nonlin_mod as snl
# import make_test_problems as mtp  # src.make_test_problems


# %% reimports
importlib.reload(fgp)
importlib.reload(snl)



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
    'jac_threshold': 1e9,  # try to use jac when rmse is below this
    'jac_precondition': False,
    'base_stepmethod': 'jac',  # jvp or jac, jac seems to work better
    'startup_period': 8,  # # of iterations in startup period (0 means no startup period)
    'startup_stepmethod': 'jvp',  # jac or jvp
    'step_fixed': False,  # False, or a fixed number
    'search_iter': 20,
    'jvp_reset_steps': 5,
    'jvp_precondition': False,
    'lgmres_maxiter': 20,
    'notes': True,
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}


# %% main function
def poisson(wh, xmat, geotargets, options=None, logfile=None):
    a = timer()

    if logfile is None:
        f = sys.stdout
    else:
        # maybe check if file is open and else open it??
        f = logfile

    # now that we have needed arguments, define lambda diffs
    l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    l_diffs = jax.jit(l_diffs)
    options['l_diffs'] = l_diffs

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

    # we need to define jac after dw is defined
    # def fjac(x):
    #     return fgp.jax_targets_diff(x, wh, xmat, geotargets, dw)

    fjac = l_diffs

    # l_diffsa = lambda bvec: jnp.array(fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw))
    # jacfn = jax.jacfwd(l_diffsa)
    # l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    # get_jac(bvec, wh, xmat, geotargets, dw)
    ljac = lambda bvec: get_jac(bvec, wh, xmat, geotargets, dw)

    jacdjb = snl.jac_setup_djb(fjac, betavec0, jacobian='krylov') # ljac)  # 'krylov')  # krylov, jvp_krylov
    print("created jacdjb")


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

    # rng = np.random.default_rng(12345)

    print('Starting Newton iterations...\n', file=f)
    print('                          max abs                         -------- # seconds --------', file=f)
    print('                    %        %              --  step --   --- iteration ----    cumul-', file=f)
    print(' iter   l2norm   change    error     rmse   method size   step search  total    ative\n', file=f)

    # print stats at start
    print(f"{0: 4} {l2norm: 9.2f}        {maxpdiff: 10.2f} {rmse: 8.2f}", file=f)

    newt_start = timer()
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
                    print("WE SHOULD NOT GET HERE!", file=f)
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
        # step_dir = get_step(bvec, wh, xmat, geotargets, dw, diffs, opts)
        step_dir = jacdjb.solve(diffs, tol=1e-3)
        step_end = timer()

        # print('step: ', step_dir)
        # step2 = jacdjb.solve(diffs, tol=1e-3)
        # print('step2: ', step2)

        search_start = timer()
        if opts.step_fixed is False:
            p = getp_min(bvec, step_dir, wh, xmat, geotargets, dw, l2norm_prior, opts)
            # p = getp_reduce(bvec, step_dir, wh, xmat, geotargets, dw, opts.search_iter, l2norm_prior)
        else: p = opts.step_fixed
        search_end = timer()

        # if p == 0:
        #     print("p==0 so perturbing step direction and trying 1 more time...")
        #     tol = 1e-8
        #     step_dir = step_dir * (1.0 + rng.uniform(low=-tol, high=tol, size=step_dir.size))
        #     p = getp_min(bvec, step_dir, wh, xmat, geotargets, dw, l2norm_prior, opts)

        bvec = bvec - step_dir * p

        diffs = fgp.jax_targets_diff_copy(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)

        jacdjb.update(bvec.copy(), diffs)

        pch = l2norm / l2norm_prior * 100 - 100
        maxpdiff = jnp.max(jnp.abs(diffs))
        rmse = math.sqrt(l2norm**2 / bvec.size)

        iter_end = timer()

        step_time = step_end - step_start
        search_time = search_end - search_start
        itime = iter_end - iter_start
        ctime = iter_end - newt_start

        print(f'{count: 4} {l2norm: 9.2f}   {pch: 6.2f} {maxpdiff: 8.2f} {rmse: 8.2f}   {step_method}  {p: 6.3f} {step_time: 6.2f} {search_time: 6.2f} {itime: 6.2f}{ctime: 9.2f}', file=f)

        if l2norm >= l2norm_prior * (1.0 - opts.no_improvement_proportion) and step_method == 'jvp':
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if l2norm < l2norm_best:
            iter_best = count
            bvec_best = bvec.copy()
            l2norm_best = l2norm.copy()
        else:
            # if this isn't the best iteration, prepare to reset
            bvec = bvec_best.copy()
            l2norm = l2norm_best.copy()

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


    print(f'\nDone with Newton iterations:', file=f)
    print(message, file=f)

    # get return values
    beta_opt = bvec_best.reshape(geotargets.shape)
    whs_opt = fgp.get_whs_logs(beta_opt, wh, xmat, geotargets) # jax_get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = jnp.dot(whs_opt.T, xmat)

    if opts.scaling:
        geotargets_opt = np.multiply(geotargets_opt, scale_factors)

    b = timer()

    print(f'  Elapsed seconds: {b - a: 9.2f}', file=f)
    print(f'  Using results from iteration # {iter_best}, with best l2norm: {l2norm_best:<12.2f}', file=f)

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

def jvp_step_test(bvec, wh, xmat, geotargets, dw, diffs, opts):
    Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html
    # Construct a linear operator that computes P^-1 * x.
    # M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
    # scipy.sparse.linalg = spla.LinearOperator((bvec.size, bvec.size), M_x)

    step, info = scipy.sparse.linalg.qmr(Jsolver, diffs, maxiter=opts.lgmres_maxiter)[0:2]
    # if info > 0:
    #     print('NOTE: lgmres did not converge after iterations: ', info)
    #     print('Increasing option lgmres_maxiter may lead to better step direction (but longer step calculation time).')
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html
    # scipy.sparse.linalg.lgmres(A, b, x0=None, tol=1e-05, maxiter=1000, M=None, callback=None,
    #   inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True, prepend_outer_v=False, atol=None)
    # print("info (0 is good): ", info)
    # step_results = scipy.optimize.lsq_linear(Jsolver, diffs, max_iter=5) # max_iter None default
    # if not step_results.success: print("Failure in getting step!! Check results carefully.")
    # step = step_results.x
    return step


def jvp_step(bvec, wh, xmat, geotargets, dw, diffs, opts):
    Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html
    # Construct a linear operator that computes P^-1 * x.
    # M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
    # scipy.sparse.linalg = spla.LinearOperator((bvec.size, bvec.size), M_x)
    if opts.jvp_precondition:
        invM = diag_jac_lowmem(opts.l_diffs, jax.device_put(jax.numpy.array(bvec)))
        invM = np.array(invM)
        spM = scipy.sparse.diags(1 / invM)
    else:
        spM = None

    # tol=1e-05  default
    step, info = scipy.sparse.linalg.lgmres(Jsolver, diffs, M = spM, maxiter=opts.lgmres_maxiter)

    # step, info = scipy.sparse.linalg.lgmres(Jsolver, diffs, maxiter=opts.lgmres_maxiter) #  outer_k=3
    if info > 0:
        # print('NOTE: lgmres did not converge after iterations: ', info, '. See option lgmres_maxiter.')
        if opts.notes:
            print(f'NOTE: lgmres jvp step did not converge after {info} iterations. See option lgmres_maxiter.')
        # print(f"{0: 4} {l2norm: 9.2f} {maxpdiff: 8.2f} {rmse: 7.2f}", file=f)
        # print('Increasing option lgmres_maxiter may lead to better step direction (but longer step calculation time).')
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html
    # scipy.sparse.linalg.lgmres(A, b, x0=None, tol=1e-05, maxiter=1000, M=None, callback=None,
    #   inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True, prepend_outer_v=False, atol=None)
    # print("info (0 is good): ", info)
    # step_results = scipy.optimize.lsq_linear(Jsolver, diffs, max_iter=5) # max_iter None default
    # if not step_results.success: print("Failure in getting step!! Check results carefully.")
    # step = step_results.x
    return step

def jvp_step_best(bvec, wh, xmat, geotargets, dw, diffs, opts):
    Jsolver = get_linop(bvec, wh, xmat, geotargets, dw, diffs)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html
    # Construct a linear operator that computes P^-1 * x.
    # M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
    # scipy.sparse.linalg = spla.LinearOperator((bvec.size, bvec.size), M_x)

    step, info = scipy.sparse.linalg.lgmres(Jsolver, diffs, maxiter=opts.lgmres_maxiter) #  outer_k=3
    if info > 0:
        # print('NOTE: lgmres did not converge after iterations: ', info, '. See option lgmres_maxiter.')
        if opts.notes:
            print(f'NOTE: lgmres jvp step did not converge after {info} iterations. See option lgmres_maxiter.')
        # print(f"{0: 4} {l2norm: 9.2f} {maxpdiff: 8.2f} {rmse: 7.2f}", file=f)
        # print('Increasing option lgmres_maxiter may lead to better step direction (but longer step calculation time).')
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html
    # scipy.sparse.linalg.lgmres(A, b, x0=None, tol=1e-05, maxiter=1000, M=None, callback=None,
    #   inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True, prepend_outer_v=False, atol=None)
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
    return jacmat

def jac_step_M(bvec, wh, xmat, geotargets, dw, diffs, options):
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    # step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]
    invMjd = np.diag(jacmat)
    # Mjd = np.diag(1 / invMjd)
    spMjd = scipy.sparse.diags(1 / invMjd)
    # print("using M")
    # M = 1 / M

    # tol=1e-05  default
    step, info = scipy.sparse.linalg.lgmres(jacmat, diffs, M = spMjd, tol=1e-08, maxiter=options.lgmres_maxiter)

    # jinv = jnp.linalg.pinv(jacmat)
    # jinv = scipy.linalg.pinv(jacmat)
    # step = jnp.dot(jinv, diffs)

    return step

def jac_step_lstsq(bvec, wh, xmat, geotargets, dw, diffs, options):
    # jac_step_lstsq
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)

    step, res, rnk, s = scipy.linalg.lstsq(
        jacmat, diffs,
        cond=1e-12,
        overwrite_a=False, overwrite_b=False,
        check_finite=True,
        lapack_driver='gelsd')  # gelsd, gelsy, gelss
    return step


def jac_step_solve(bvec, wh, xmat, geotargets, dw, diffs, options):
    # jac_step_solve
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    step = np.linalg.solve(jacmat, diffs)
    return step

def jac_step_lgmres(bvec, wh, xmat, geotargets, dw, diffs, options):
    # BEST
    # jac_step_lgmres
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)

    # tol=1e-05  default
    step, info = scipy.sparse.linalg.lgmres(jacmat, diffs, maxiter=options.lgmres_maxiter)

    return step

def jac_step_approx(bvec, wh, xmat, geotargets, dw, diffs, options):
    # from scipy.sparse.linalg import dsolve
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    # step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]

    # jinv = jnp.linalg.pinv(jacmat)
    # jacmat = jacmat.astype(np.float64)
    # jacmat = jacmat.astype(np.complex64)
    # np.complex64
    # step = scipy.sparse.linalg.dsolve.spsolve(jacmat, diffs, use_umfpack=False)
    step, info = scipy.sparse.linalg.qmr(jacmat, diffs)[:2]
    # step = scipy.sparse.linalg.dsolve.linsolve.spsolve(jacmat, diffs, use_umfpack=True)
    # step = jnp.dot(jinv, diffs)

    return step


def jac_step_best(bvec, wh, xmat, geotargets, dw, diffs, options):
    # jac_step_best
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    # step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]

    # jinv = jnp.linalg.pinv(jacmat)
    jinv = scipy.linalg.pinv(jacmat)  # , cond=1e-11
    # documentation for scipy 1.7.0 (caution: code currently uses 1.6.2)
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.linalg.pinv.html#scipy.linalg.pinv
    # scipy.linalg.pinv(a, atol=None, rtol=None, return_rank=False, check_finite=True, cond=None, rcond=None)
    # rcond in scipy 1.6.2 appears to be rtol in 1.7.0, and cond --> atol  DO NOT USE BOTH cond and rcond
    # rtol default value is max(M, N) * eps where eps is the machine precision value of the datatype of a
    step = jnp.dot(jinv, diffs)

    return step

def jac_step_qr(bvec, wh, xmat, geotargets, dw, diffs, options):
    # jac_step_qr
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    # Ax = b
    # A is jacmat, b is diffs, and x is step
    q, r = jax.scipy.linalg.qr(jacmat)
    p = jnp.dot(q.T, diffs)
    step = jnp.dot(jnp.linalg.inv(r), p)
    # step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]

    # jinv = jnp.linalg.pinv(jacmat)
    # jinv = scipy.linalg.pinv(jacmat)  # , cond=1e-11
    # documentation for scipy 1.7.0 (caution: code currently uses 1.6.2)
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.linalg.pinv.html#scipy.linalg.pinv
    # scipy.linalg.pinv(a, atol=None, rtol=None, return_rank=False, check_finite=True, cond=None, rcond=None)
    # rcond in scipy 1.6.2 appears to be rtol in 1.7.0, and cond --> atol  DO NOT USE BOTH cond and rcond
    # rtol default value is max(M, N) * eps where eps is the machine precision value of the datatype of a
    # step = jnp.dot(jinv, diffs)

    return step


# A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
# b = np.array([1, 1, 1, 1])
# lu, piv = lu_factor(A)
# x = lu_solve((lu, piv), b)

# A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
# c, low = cho_factor(A)
# x = cho_solve((c, low), [1, 1, 1, 1])

def jac_step_chol(bvec, wh, xmat, geotargets, dw, diffs, options):
    # jac_step_chol
    print("cholesky factorization...")
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    c, low = jax.scipy.linalg.cho_factor(jacmat)
    step = jax.scipy.linalg.cho_solve((c, low), diffs)
    return step

def jac_step(bvec, wh, xmat, geotargets, dw, diffs, options):
    # jac_step_lu1
    # print("lu factorization...")
    # which is better, pinv or lu??
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    lu, piv = jax.scipy.linalg.lu_factor(jacmat)
    step = jax.scipy.linalg.lu_solve((lu, piv), diffs)
    return step


def jac_step_lu2(bvec, wh, xmat, geotargets, dw, diffs, options):
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


# %% low memory approach to getting the diagonal of the Jacobian
def diag_jac_lowmem(f, x):
    # get the diagonal of the jacobian iteratively while using very little memory
    # do this so we can use the inverse of the diagaonal as a preconditioner
    # inspired by:
    #     https://github.com/google/jax/issues/1563 for general approach, and
    #     https://github.com/google/jax/issues/1923 for map as alternative to vmap
    #     map drastically reduces memory usage and actually is faster than vmap
    #     as the jacobian gets larger
    # call this as:
    #   diag_jac_lowmem(f, jax.device_put(jax.numpy.array(x)))
    # where f is a function of m variables that returns a vector of m values
    # and x is the input vector to the function f
    def partial_grad_f_index(i):
        def partial_grad_f_x(xi):
            return f(jax.ops.index_update(x, i, xi))[i]
        return jax.grad(partial_grad_f_x)(x[i])
    return jax.lax.map(partial_grad_f_index, jax.numpy.arange(x.shape[0]))


# %% functions related to line search

def getp_min(bvec, step_dir, wh, xmat, geotargets, dw, l2norm_prior, opts):

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
        method='bounded', options={'maxiter': opts.search_iter})
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
    # scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method='brent', tol=None, options=None)
    # options for method bounded are: options={'func': None, 'xatol': 1e-05, 'maxiter': 500, 'disp': 0})

    p = res.x
    new_norm = get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw)
    if new_norm > l2norm_prior:
        if opts.notes:
            print(f'NOTE: Did not find l2norm-reducing step size after {res.nfev} function evaluations, setting p to 0.')

        p = 0

    if not res.success and (new_norm < l2norm_prior) and (0.1 < res.x < 0.9):
        # print('NOTE: optimal step size not found after function evaluations: ', res.nfev, '. See option search_iter.')
        if opts.notes:
            print(f'NOTE: l2norm improved but optimal step size not found after {res.nfev} function evaluations. See option search_iter.')
        # print(f'NOTE: lgmres did not converge after {info} iterations. See option lgmres_maxiter.')
        # print('Increasing option search_iter may result in better step size (but longer calculation time).')
        # print('Solver message: ', res.message)

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



# def getp_min(bvec, step_dir, wh, xmat, geotargets, dw, search_iter, l2norm_prior):
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
#     #  scipy.optimize.line_search(f, myfprime, xk, pk, gfk=None, old_fval=None, old_old_fval=None,
#     #   args=(), c1=0.0001, c2=0.9, amax=None, extra_condition=None, maxiter=10)

#     def get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw):
#         bvec = bvec - step_dir * p
#         diffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
#         l2norm = norm(diffs, 2)
#         return l2norm

#     # p = 1.0
#     # l2norm = get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw)
#     # if l2norm < l2norm_prior:
#     #     return p

#     res = minimize_scalar(get_norm, bounds=(0, .9), args=(bvec, step_dir, wh, xmat, geotargets, dw),
#         method='bounded', options={'maxiter': search_iter})
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
#     # scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method='brent', tol=None, options=None)
#     # options for method bounded are: options={'func': None, 'xatol': 1e-05, 'maxiter': 500, 'disp': 0})
#     if not res.success:
#         print('NOTE: optimal step size not found after (option search_iter) function evaluations: ', res.nfev)
#         # print('Increasing option search_iter may result in better step size (but longer calculation time).')
#         # print('Solver message: ', res.message)

#     p = res.x
#     if get_norm(p, bvec, step_dir, wh, xmat, geotargets, dw) > l2norm_prior:
#         p = 0

#     return p
