# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

@author: donbo
"""


# https://github.com/google/jax/issues/4753
# https://github.com/hanrach/p2d_solver/blob/main/run_ex.py


# %% imports

import numpy as np
from numpy.linalg import norm
import jax
import jax.numpy as jnp
from timeit import default_timer as timer

# from numpy.linalg import solve
# from numpy.linalg import norm
from jax import jacfwd
from jax import jvp

# this next line is CRUCIAL or we will lose precision
from jax.config import config; config.update('jax_enable_x64', True)

from collections import namedtuple

import utilities as ut # src.utilities
import make_test_problems as mtp  # src.make_test_problems


# %% functions needed for residuals
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
    diffs = diffs * diff_weights

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs # np.array(diffs)  # note that this is np, not jnp!

def jvp2(g):
    f = lambda x: g(x, wh, xmat, geotargets, dw)
    _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
    return _jvp(g)

# f = lambda betavec: jax_targets_diff(betavec, wh, xmat, geotargets, dw)
# f(bv2)
# # Push forward the vector `v` along `f` evaluated at `W`
# y, u = jax.jvp(f, (betavec,), (bv2,))
# u



# %% test on something small
p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=3, k=3, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=10, k=8, xsd=.1, ssd=.5, pctzero=.2)
p = mtp.Problem(h=20000, s=20, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=30000, s=30, k=20, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=35000, s=40, k=25, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4) 
p = mtp.Problem(h=50000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.2)

wh = p.wh
xmat = p.xmat
geotargets = p.geotargets
betavec0 = np.full(geotargets.size, 0.5, dtype='float64')  # 1e-13 or 1e-12 seems best

np.random.seed(1)
gnoise = np.random.normal(0, .01, p.k * p.s)
gnoise = gnoise.reshape(p.geotargets.shape)
ngtargets = p.geotargets * (1 + gnoise)

ngtargets = p.geotargets

dw = jax_get_diff_weights(ngtargets)

# tmp = jax_targets_diff(betavec0, wh, xmat, geotargets, dw)

# %% solve using jvp cg


# jac_x_prod = lambda x: jvp(jax_targets_diff, (betavec,), (x,))[1]
# jac_x_prod = lambda x: jvp2(jax_targets_diff, (betavec,), (x,))[1]
# jac_x_prod = lambda x: jvp(jax_targets_diff, (betavec, wh, xmat, geotargets, dw), (x,))[1]
# jac_x_prod = lambda x: jvp(jax_targets_diff, (betavec, wh, xmat, geotargets, dw), (x, wh, xmat, geotargets, dw))[1]




# f(bv2)
# # Push forward the vector `v` along `f` evaluated at `W`
# jax.jvp(f, (betavec,), (bv2,))


# djb this works
f = lambda betavec: jax_targets_diff(betavec, wh, xmat, ngtargets, dw)
jac_x_prod = lambda x: jvp(f, (betavec,), (x, ))[1]

f(betavec0)
jac_fn = lambda x: jacfwd(f, (betavec,), (x, ))
jac_fn(betavec0)  # djb come back to this and get syntax right

jax_jacobian_basic = jax.jacfwd(jax_targets_diff)
J = jax_jacobian_basic(beta, wh, xmat, geotargets, dw)
# J = np.array(J.reshape((dw.size, dw.size))


# https://github.com/google/jax/tree/master/examples
jvpf = lambda x: jax_targets_diff(x, wh, xmat, ngtargets, dw)
jvpf2 = lambda s: jax.jvp(jvpf, (x, ), (s,))[1]

jvpf = lambda betavec: jax_targets_diff(betavec, wh, xmat, ngtargets, dw)
jvpf2 = lambda s: jax.jvp(jvpf, (betavec,), (s,))[1]
jpbasic = jax.jvp(jvpf2)

import inspect
print(inspect.signature(jvpf))
print(inspect.signature(jvpf2))
jvpf2(y)
jvpf2(betavec)


lambda x: jvp(fn, (U2,), (x,))[1]

jvpf(betavec) # targets_diff at this point
jvpf2(betavec)
jax.jvp(jvpf, (betavec,), (y, ))
jax.jvp(jvpf, (betavec,), (y, ))
jax.jvp(jvpf, (betavec,), (wh, xmat, geotargets, dw))
jax.jvp(jvpf, (wh, xmat, geotargets, dw), (betavec,))

jax.scipy.sparse.linalg.cg(jvpf2, y)[0]

# jac_x_prod = jax.jit(jac_x_prod)
# djb come back to this
def f(x):
    return jnp.square(x + 2)

x = 2.
f(x)  # 4.
jax.jvp(f, (x,), (1.,))    

x = jnp.array([2., 3.])
s = jnp.array([7., 9])
f(x)
jfns = jax.jacfwd(f)
jvals = jfns(x)
jvals
jfns(s)

jvals.dot(s) # same as lf(s)
jvals.dot(x)  # same as lf(x)

yvals = f(x)
step1 = jnp.linalg.lstsq(jvals, yvals, rcond=None)[0]
step1

# same result with jvp??
lf = lambda z: jvp(f, (x,), (z,))[1]
lf(x)
lf(s)
step2 = jax.scipy.sparse.linalg.cg(lf, yvals)[0]  # yes, same as linalg.lstsq
step2

jax.jvp(f, (x,), (s, ))

tmp = jax.jvp(f, (x,), (s, ))[1]

# bv2 = np.full(geotargets.size, 0.1) 
# jac_x_prod(bv2)

maxit = 10
tol = 1e-4

count = 0
res = 1e9
betavec = betavec0

start = timer()
while(count < maxit and  res > tol):
    # print("\n", count)
    print("betavec", betavec)
    # print(jac_x_prod(betavec))
    y = jax_targets_diff(betavec, wh, xmat, ngtargets, dw)

    print("y", y)
    # res = jnp.linalg.norm(y / jnp.linalg.norm(betavec, np.inf), np.inf)
    res = jnp.square(y).sum()
    print("res", res)

    step = jax.scipy.sparse.linalg.cg(jac_x_prod, y)[0]

    J = jax_jacobian_basic(betavec, wh, xmat, geotargets, dw)
    # J = np.array(J.reshape((dw.size, dw.size))
    step = jnp.linalg.lstsq(J, y, rcond=None)[0]

    print("step", step)

    betavec = betavec - step
    count = count + 1
    print(count, res)

end = timer()

time = end - start
time


# U2
# np.round(fn(U2), 2)



# %% define jax functions
# jax.jit(jax_targets_diff)
jac_fn = jax.jit(jacfwd(fn))
fn = jax.jit(fn)

# %% retest as jit on something small


# %% loop

# %% option defaults
options_defaults = {
    'scaling': True,
    'scale_goal': 1e3,
    'init_beta': 0.5,
    'quiet': True}

# options_defaults = {**solver_defaults, **user_defaults}

# %% newton
def Newton_system(F, J, x, wh, xmat, geotargets, dw, eps=.01, maxpd=.01):
    """    
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F. Both F and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    http://hplgit.github.io/prog4comp/doc/pub/._p4c-solarized-Python031.html
    """
    F_value = F(x, wh, xmat, geotargets, dw)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector    
    max_abs_pdiff = np.abs(F_value / geotargets.flatten() * 100.0).max()
    print('iteration  F_norm  max_abs_pdiff')
    iteration_counter = 0        
    while abs(F_norm) > eps and max_abs_pdiff > maxpd and iteration_counter < 10:
        jval = J(x, wh, xmat, geotargets, dw)
        delta = jnp.linalg.lstsq(jval, -F_value, rcond=None)[0] # , rcond=None)[0]
        # hplgit's original line was the following but it is not numerically stable
        # delta = np.linalg.solve(J(x), -F_value)  so I use lstsq
        x = x + delta
        F_value = F(x, wh, xmat, geotargets, dw)
        print(iteration_counter, F_norm, max_abs_pdiff)
        F_norm = jnp.linalg.norm(F_value, ord=2)
        max_abs_pdiff = np.abs(F_value / geotargets.flatten() * 100.0).max()
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
    
    print("all done!")
    return x, iteration_counter, F_value

# jnewt = jax.jit(Newton_system)

# %% problem
# p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
# p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)    
# p = mtp.Problem(h=1000, s=8, k=4, xsd=.1, ssd=.5, pctzero=.4)    
# p = mtp.Problem(h=10000, s=15, k=10, xsd=.1, ssd=.5, pctzero=.4)    

# p = mtp.Problem(h=20000, s=25, k=20, xsd=.1, ssd=.5, pctzero=.4)    
# p = mtp.Problem(h=30000, s=35, k=30, xsd=.1, ssd=.5, pctzero=.4)    
# p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4)    

# %% continue setup
# wh = p.wh
# xmat = p.xmat
# # geotargets = p.geotargets

# # now add noise to geotargets
# np.random.seed(1)
# gnoise = np.random.normal(0, .05, p.k * p.s)
# gnoise = gnoise.reshape(p.geotargets.shape)
# ngtargets = p.geotargets * (1 + gnoise)

# geotargets = ngtargets

# np.round(geotargets / p.geotargets * 100 - 100, 2)



# %% poisson - the primary function

def poisson(wh, xmat, geotargets, options=None):
    # TODO: implement options
    a = timer()

    options_all = options_defaults.copy()
    options_all.update(options)
    opts = ut.dict_nt(options_all)  # convert dict to named tuple for ease of use

    if opts.scaling:
        xmat, geotargets, scale_factors = scale_problem(xmat, geotargets, scale_goal = 1e3)

    # betavec0 = np.zeros(geotargets.size)
    betavec0 = np.full(geotargets.size, opts.init_beta)  # 1e-13 or 1e-12 seems best
    dw = get_diff_weights(geotargets)
    
    jax_jacobian_basic = jax.jit(jax.jacfwd(jax_targets_diff))

    def jax_jacobian(beta, wh, xmat, geotargets, dw):
        jac_values = jax_jacobian_basic(beta, wh, xmat, geotargets, dw)
        jac_values = np.array(jac_values).reshape((dw.size, dw.size))
        return jac_values

    x, n, fval = Newton_system(F=targets_diff,
             J=jax_jacobian, x=betavec0,
             wh=wh, xmat=xmat, geotargets=geotargets, dw=dw,
             eps=0.01,
             maxpd=.01)

    # x, n, fval = jnewt(F=jax_targets_diff,
    #          J=jacmethod, x=betavec0,
    #          wh=wh, xmat=xmat, geotargets=geotargets, dw=dw,
    #          eps=0.01,
    #          maxpd=.01)             

    # get return values
    beta_opt = x.reshape(geotargets.shape)
    delta_opt = get_delta(wh, beta_opt, xmat)
    whs_opt = get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = get_geotargets(beta_opt, wh, xmat)

    if opts.scaling:
        geotargets_opt = np.multiply(geotargets_opt, scale_factors)

    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets_opt',
              'beta_opt',
              'delta_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 whs_opt=whs_opt,
                 geotargets_opt=geotargets_opt,
                 beta_opt=beta_opt,
                 delta_opt=delta_opt)

    return res


# %% functions
def get_delta(wh, beta, xmat):
    """Get vector of constants, 1 per household.

    See (Khitatrakun, Mermin, Francis, 2016, p.5)

    Note: beta %*% xmat can get very large!! in which case or exp will be Inf.
    It will get large when a beta element times an xmat element is large,
    so either beta or xmat can be the problem.

    In R the problem will bomb but with numpy it appears to recover
    gracefully.

    According to https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
      For most practical purposes, you can probably approximate
        1 / (1 + <a large number>) to zero. That is to say, just ignore the
      warning and move on. Numpy takes care of the approximation for
      you (when using np.float64).

    This will generate runtime warnings of overflow or divide by zero.
    """
    beta_x = np.exp(np.dot(beta, xmat.T))

    # beta_x[beta_x == 0] = 0.1  # experimental
    # beta_x[np.isnan(beta_x)] = 0.1

    delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    # print(delta)
    # delta[delta == 0] = 0.1  # experimental
    # delta[np.isnan(delta)] = 0.1
    return delta


def get_diff_weights(geotargets, goal=100):
    """
    difference weights - a weight to be applied to each target in the
      difference function so that it hits its goal
      set the weight to 1 if the target value is zero

    do this in a vectorized way
    """

    # avoid divide by zero or other problems

    # numerator = np.full(geotargets.shape, goal)
    # with np.errstate(divide='ignore'):
    #     dw = numerator / geotargets
    #     dw[geotargets == 0] = 1

    goalmat = np.full(geotargets.shape, goal)
    with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
        diff_weights = np.where(geotargets != 0, goalmat / geotargets, 1)

    return diff_weights


def get_geotargets(beta, wh, xmat):
    """
    Calculate matrix of target values by state and characteristic.

    Returns
    -------
    targets_mat : matrix
        s x k matrix of target values.

    """
    delta = get_delta(wh, beta, xmat)
    whs = get_geoweights(beta, delta, xmat)
    targets_mat = np.dot(whs.T, xmat)
    return targets_mat


def get_geoweights(beta, delta, xmat):
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
        s x k matrix of coefficients for the poisson function that generates
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
    beta_x = np.dot(beta, xmat.T)

    # add the delta vector of household constants to every row
    # of beta_x and transpose
    # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
    beta_xd = (beta_x + delta).T

    weights = np.exp(beta_xd)

    return weights


def targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
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

    geotargets_calc = get_geotargets(beta, wh, xmat)
    diffs = geotargets_calc - geotargets
    diffs = diffs * diff_weights

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs


# %% jax functions
# these functions are used by jax to compute the jacobian
# I have not yet figured out how to avoid having two versions of the functions



# %% scaling
def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = np.divide(xmat, scale_factors)
    geotargets = np.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors

