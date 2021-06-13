
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
    'p': .9,
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}
# if np.linalg.cond(Ds) < 1 / sys.float_info.epsilon:
#     Dsinv = np.linalg.inv(Ds)
# else:
#     print("Ds is singular, cannot be inverted")

# %% main function
def poisson(wh, xmat, geotargets, options=None):
    a = timer()
    # NOTE: in this version we keep beta as a matrix throughout, not as a vector

    # override default options with user options, where appropriate
    options_all = options_defaults
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = fgp.scale_problem(xmat, geotargets, opts.scale_goal)

    if np.size(opts.init_beta)==1:
        beta0 = np.full(geotargets.shape, opts.init_beta)
    else:
        beta0 = opts.init_beta

    # prepare for Newton iterations
    # create the 3-dimensional array xxp:
    #   for each row in x, it has a k x k matrix that is the dot product of that row with its transpose
    #   thus its dimension is h x k x k
    xxp = np.einsum('ij,ik->ijk', xmat, xmat)

    beta = beta0
    dw = fgp.jax_get_diff_weights(geotargets)

    count = 0
    no_improvement_count = 0
    no_improvement_proportion = 1e-3

    # construct initial values, pre-iteration
    diffs = fgp.jax_targets_diff_copy(beta, wh, xmat, geotargets, dw)
    l2norm = norm(diffs, 2)
    maxpdiff = jnp.max(jnp.abs(diffs))
    rmse = math.sqrt((l2norm**2 - maxpdiff**2) / (beta.size - 1))

    # define initial best values
    iter_best = 0
    l2norm_prior = np.float64(1e99)
    l2norm_best = l2norm_prior.copy()
    beta_best = beta.copy()
    l2norm_best = l2norm_prior.copy()

    # set all stopping conditions to False
    max_iter = False
    low_error = False
    no_improvement = False
    NO_IMPROVEMENT_MAX = 3
    ready_to_stop = False

    print('Starting Newton iterations...')
    print('                   rmse   max abs   step   step size (p)      step    search    total')
    print(' iter   l2norm    ex-max  % error  method    min   max      seconds  seconds  seconds\n')

    # print stats at start
    print(f"{0: 4} {l2norm: 10.2f} {rmse: 8.2f} {maxpdiff: 8.2f}")

    step_method = 'sep'
    p = 1.0

    while not ready_to_stop:
        count += 1
        iter_start = timer()

        beta_x = jnp.dot(beta, xmat.T)
        exp_beta_x = jnp.exp(beta_x)
        delta = jnp.log(wh / exp_beta_x.sum(axis=0))
        beta_xd = (beta_x + delta).T  # this is the same as before but with new betax delta
        whs = jnp.exp(beta_xd)
        targs = jnp.dot(whs.T, xmat)  # s x k
        diffs = targs - geotargets  # s x k
        pdiffs = diffs / targs * 100.
        l2norm = norm(pdiffs, 2)
        maxpdiff = jnp.max(jnp.abs(pdiffs))
        rmse = math.sqrt((l2norm**2 - maxpdiff**2) / (beta.size - 1))

        step_start = timer()
        # D1 = np.einsum('il,ijk->ljk', whs, xxp)  # this is an array of jacobian
        # print("correct D: ", D1)

        D = loop_jac(beta, wh, xmat, geotargets, delta)
        # print("Jacobian (?): ", D)

        # what to do if we can't invert the D matrix?
        # invD1 = np.linalg.pinv(D1)  # invert all of them
        invD = np.linalg.pinv(D)  # pseudo inverse - an array of D inverses

        # print("correct inverse: ", invD1)
        # print("j inverse: ", invD)

        step_dir = np.einsum('ijk,ik->ij', invD, diffs)
        step_end = timer()
        step_time = step_end - step_start

        search_start = timer()
        # p = getp_min(bvec, step_dir, wh, xmat, geotargets, dw, opts.search_iter)
        # numpy.apply_along_axis(abs_sum, 1, arr)
        # p = .2
        p = getp(beta, step_dir, wh, xmat, geotargets, opts.search_iter)
        # p = opts.p
        # print(p)
        pmin = np.min(p)
        pmax = np.max(p)
        p = np.min(p)
        search_end = timer()

        # p = np.ones(beta.shape[0])
        beta = beta - step_dir * p # [:, np.newaxis]
        iter_end = timer()
        itime = iter_end - iter_start
        search_time = search_end - search_start

        print(f'{count: 4} {l2norm: 10.2f} {rmse: 8.2f} {maxpdiff: 8.2f}    {step_method}    {pmin: 6.3f}  {pmax: 6.3f}   {step_time: 6.2f}   {search_time: 6.2f}    {itime: 6.2f}')


        if l2norm >= l2norm_prior: # * (1.0 - no_improvement_proportion):
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        l2norm_prior = l2norm

        if l2norm < l2norm_best:
            iter_best = count
            beta_best = beta.copy()
            l2norm_best = l2norm.copy()
        else:
            # if this isn't the best iteration, reset the jvp counter
            beta = beta_best.copy()
            l2norm = l2norm_best.copy()
            # step_reset = True

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


    # beta_best = beta
    # iter_best = count
    # l2norm_best = l2norm

    print(f'\nDone with Newton iterations:')
    print(message)

    # get return values
    beta_opt = beta_best
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


# %% ..full jacobian approach

def dfn(beta, wh, xmat, geotargets, delta, dw):
    beta_x = jnp.dot(beta, xmat.T)
    # exp_beta_x = jnp.exp(beta_x)
    # delta = jnp.log(wh / exp_beta_x.sum(axis=0))
    beta_xd = (beta_x + delta).T  # this is the same as before but with new betax delta
    whs = jnp.exp(beta_xd)
    targs = jnp.dot(whs.T, xmat)  # s x k
    diffs = targs - geotargets  # s x k
    return diffs


def get_jac(bvec, wh, xmat, geotargets, delta, dw):
    jacfn = jax.jacfwd(dfn)
    # jacfn = jax.jit(jacfn)
    jacmat = jacfn(bvec, wh, xmat, geotargets, delta, dw)
    jacmat = np.array(jacmat).reshape((bvec.size, bvec.size))
    return jacmat

def loop_jac(beta, wh, xmat, geotargets, delta):
    # loop through each row of beta and get the jacobian
    dw = np.ones(beta.shape[1])
    jacarray = np.zeros((beta.shape[0], beta.shape[1], beta.shape[1]))
    # print("jacarray shape: ", jacarray.shape)
    # print("range: ", range(beta.shape[0]))
    for state in range(beta.shape[0]):
        # print("state: ", state)
        # print("beta: ", beta)
        bvec = beta[state]
        targets = geotargets[state]
        jacmat = get_jac(bvec, wh, xmat, targets, delta, dw)
        jacarray[state] = jacmat

    # print(jacarray.shape)
    return jacarray



def jac_step(bvec, wh, xmat, geotargets, dw, diffs):
    jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
    step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]
    return step

# %% functions related to line search

def getp(beta, step_dir, wh, xmat, geotargets, search_iter):

    def get_norm(p, bvec, stepvec, wh, xmat, targets):
        bvec = bvec - stepvec * p
        beta_x = jnp.dot(bvec, xmat.T)
        exp_beta_x = jnp.exp(beta_x)
        delta = jnp.log(wh / exp_beta_x.sum(axis=0))
        beta_xd = (beta_x + delta).T  # this is the same as before but with new betax delta
        whs = jnp.exp(beta_xd)
        etargs = jnp.dot(whs.T, xmat)  # s x k
        diffs = etargs - targets  # s x k
        pdiffs = diffs / etargs * 100.
        l2norm = norm(pdiffs, 2)
        return l2norm

    # loop through each row of beta and stepdir, and get the best p
    p = np.ones(beta.shape[0])
    for state in range(beta.shape[0]):
        bvec = beta[state]
        stepvec = step_dir[state]
        targets = geotargets[state]
        p[state] = minimize_scalar(get_norm, bounds=(0, 1),
            args=(bvec, stepvec, wh, xmat, targets),
            method='bounded', options={'maxiter': search_iter}).x
    return p



# def get_jac(bvec, wh, xmat, geotargets, dw):
#     jacfn = jax.jacfwd(fgp.jax_targets_diff)
#     jacfn = jax.jit(jacfn)
#     jacmat = jacfn(bvec, wh, xmat, geotargets, dw)
#     jacmat = np.array(jacmat).reshape((bvec.size, bvec.size))
#     return jacmat

# def jac_step(bvec, wh, xmat, geotargets, dw, diffs):
#     jacmat = get_jac(bvec, wh, xmat, geotargets, dw)
#     step = jnp.linalg.lstsq(jacmat, diffs, rcond=None)[0]
#     return step


