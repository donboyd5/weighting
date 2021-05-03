
# This uses the Newton method with jvp and lsq


# %% imports
# import inspect
import importlib

import scipy

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
    'scale_goal': 1e3,
    'init_beta': 0.5,
    'stepmethod': 'jvp',  # jvp or jac
    'max_iter': 5,
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

    betavec0 = np.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    bvec = betavec0
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

    # begin Newton iterations
    count = 0
    tol = 1e-4
    error = 1e99
    
    print("iteration        sspd        l2norm      maxabs_error")
    while count < opts.max_iter and error > tol:
        count += 1
        diffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
        l2norm = norm(diffs, 2)
        maxabs = norm(jnp.abs(diffs), jnp.inf)
        error = jnp.square(diffs).sum()
        print(f'{count: 6}   {error: 12.2f}  {l2norm: 12.2f}      {maxabs: 12.2f}')

        step = get_step(bvec, wh, xmat, geotargets, dw, diffs)
        bvec = bvec - step
    
    # get return values
    beta_opt = bvec.reshape(geotargets.shape)
    delta_opt = fgp.jax_get_delta(wh, beta_opt, xmat)
    whs_opt = fgp.jax_get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = fgp.jax_get_geotargets(beta_opt, wh, xmat)

    if opts.scaling:
        geotargets_opt = np.multiply(geotargets_opt, scale_factors)

    b = timer()

    print(f'\nDone with Newton iterations. Elapsed seconds: {b - a: 9.2f}')

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

