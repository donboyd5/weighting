

# %% imports
import numpy as np
from timeit import default_timer as timer
from collections import namedtuple
import scipy.optimize as spo

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev
from jax import vjp
from jax import jacobian

import autograd as ag

import src.make_test_problems as mtp


# %% regular functions
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


# %% jx functions
def jxget_delta(wh, beta, xmat):
    beta_x = jnp.exp(jnp.dot(beta, xmat.T))
    delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta

def jxget_diff_weights(geotargets, goal=100):
    goalmat = jnp.full(geotargets.shape, goal)
    # djb note there is no jnp.errstate so I use np.errstate  
    with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
        diff_weights = jnp.where(geotargets != 0, goalmat / geotargets, 1)
    return diff_weights

def jxget_geoweights(beta, delta, xmat):
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
    beta_x = jnp.dot(beta, xmat.T)

    # add the delta vector of household constants to every row
    # of beta_x and transpose
    # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
    beta_xd = (beta_x + delta).T

    weights = jnp.exp(beta_xd)

    return weights


def jxget_geotargets(beta, wh, xmat):
    """
    Calculate matrix of target values by state and characteristic.

    Returns
    -------
    targets_mat : matrix
        s x k matrix of target values.

    """
    delta = jxget_delta(wh, beta, xmat)
    whs = jxget_geoweights(beta, delta, xmat)
    targets_mat = jnp.dot(whs.T, xmat)
    return targets_mat    


def jxtargets_diff(beta_object, wh, xmat, geotargets, diff_weights):
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

    geotargets_calc = jxget_geotargets(beta, wh, xmat)
    diffs = geotargets_calc - geotargets
    diffs = diffs * diff_weights

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs


# %% jax jacobian functions
jfn = jax.jacobian(jxtargets_diff)

def jfn2(beta0, wh, xmat, geotargets, dw):
   jacvals = jfn(beta0, wh, xmat, geotargets, dw)
   jacvals = np.array(jacvals).reshape((dw.size, dw.size))
   return jacvals


# %% testbed
import numpy as np
import jax.numpy as jnp
# this next item is CRUCIAL or we will lose precision
from jax.config import config; config.update("jax_enable_x64", True)

p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=100, s=8, k=4, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=8, k=4, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=15, k=10, xsd=.1, ssd=.5, pctzero=.4) # bad jtdiffs
p = mtp.Problem(h=10000, s=30, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=20000, s=50, k=20, xsd=.1, ssd=.5, pctzero=.4)

wh = p.wh
xmat = p.xmat
geotargets = p.geotargets

betavec0 = np.full(geotargets.size, 0.1) 
beta0 = betavec0.reshape(geotargets.shape)

delta = get_delta(wh, beta0, xmat)
dw = get_diff_weights(geotargets)

# next few items not needed to run the optimization
gweights = get_geoweights(beta0, delta, xmat)

whs = get_geoweights(beta0, delta, xmat)
whs.sum(axis=1)
wh

# test what went wrong
get_delta(wh, beta0, xmat)
jxget_delta(wh, beta0, xmat) # this is bad

# why?? look into delta
np.exp(np.dot(beta0, xmat.T))
jnp.exp(jnp.dot(beta0, xmat.T)) # infs

tmp = np.exp(np.dot(beta0, xmat.T))
dir(tmp)
type(tmp)
tmp.dtype # float64

jtmp = jnp.exp(jnp.dot(beta0, xmat.T)) # infs
jtmp.dtype # float32

# is it dot??
np.dot(beta0, xmat.T)
jnp.dot(beta0, xmat.T) # this looks ok

get_geotargets(beta0, wh, xmat)
jxget_geotargets(beta0, wh, xmat)

tdiffs = targets_diff(beta0, wh, xmat, geotargets, dw)
jtdiffs = jxtargets_diff(beta0, wh, xmat, geotargets, dw)
tdiffs
jtdiffs

jfn2(beta0, wh, xmat, geotargets, dw)   

# %% optimize

spo_result = spo.least_squares(
    fun=targets_diff,
    x0=betavec0,
    method='trf', jac='2-point', verbose=2,
    ftol=1e-7, xtol=1e-7,
    x_scale='jac',
    loss='soft_l1',  # linear, soft_l1, huber, cauchy, arctan,
    max_nfev=100,
    args=(wh, xmat, geotargets, dw))


spo_result = spo.least_squares(
    fun=targets_diff,
    x0=betavec0,
    method='trf', jac=jfn2, verbose=2,
    ftol=1e-7, xtol=1e-7,
    # x_scale='jac',
    loss='soft_l1',  # linear, soft_l1, huber, cauchy, arctan,
    max_nfev=100,
    args=(wh, xmat, geotargets, dw))


# %% autograd
import autograd.numpy as np
jagfn = ag.jacobian(targets_diff)

jtest = jfn(beta0, wh, xmat, geotargets, dw)
jtest.shape
jtest.reshape((6, 6))

jagtest = jagfn(beta0, wh, xmat, geotargets, dw)
jagtest.shape
jagtest.reshape((6, 6))




# %% poisson - the primary function

def poisson(wh, xmat, geotargets, options=None):
    # TODO: implement options
    a = timer()
    # betavec0 = np.zeros(geotargets.size)
    betavec0 = np.full(geotargets.size, 0.1)  # 1e-13 or 1e-12 seems best
    dw = get_diff_weights(geotargets)
    spo_result = spo.least_squares(
        fun=targets_diff,
        x0=betavec0,
        method='trf', jac='2-point', verbose=2,
        ftol=1e-7, xtol=1e-7,
        x_scale='jac',
        loss='soft_l1',  # linear, soft_l1, huber, cauchy, arctan,
        max_nfev=100,
        args=(wh, xmat, geotargets, dw))

    # get return values
    beta_opt = spo_result.x.reshape(geotargets.shape)
    delta_opt = get_delta(wh, beta_opt, xmat)
    whs_opt = get_geoweights(beta_opt, delta_opt, xmat)
    geotargets_opt = get_geotargets(beta_opt, wh, xmat)

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




# %% older stuff
def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
        return outputs


def logprob_fun(params, inputs, targets):
    preds = predict(params, inputs)
    return jnp.sum((preds - targets)**2)


grad_fun = jit(grad(logprob_fun))  # compiled gradient evaluation function
# fast per-example grads
perex_grads = jit(vmap(grad_fun, in_axes=(None, 0, 0)))



def objective(X):
    x, y, z = X
    return x**2 + y**2 + z**2


def eq(X):
    x, y, z = X
    return 2 * x - y + z - 3


sol = minimize(objective, [1, -0.5, 0.5],
               constraints={'type': 'eq', 'fun': eq})
sol


import autograd.numpy as np
from autograd import grad

def F(L):
    'Augmented Lagrange function'
    x, y, z, _lambda = L
    return objective([x, y, z]) - _lambda * eq([x, y, z])

# Gradients of the Lagrange function
dfdL = grad(F, 0)

x1 = [1.0, 2.0, 3.0, 4.0]
x1 = [1.0, -0.5, 0.5, 0.0]
dfdL(x1)

# Find L that returns all zeros in this function.
def obj(L):
    x, y, z, _lambda = L
    dFdx, dFdy, dFdz, dFdlam = dfdL(L)
    return [dFdx, dFdy, dFdz, eq([x, y, z])]

from scipy.optimize import fsolve
x, y, z, _lam = fsolve(obj, [0.0, 0.0, 0.0, 1.0])
print(f'The answer is at {x, y, z}')

from autograd import hessian
h = hessian(objective, 0)
h(np.array([x, y, z]))

np.linalg.eig(h(np.array([x, y, z])))[0]





# import autograd.numpy as np
from autograd import grad
from autograd import hessian_vector_product as hvp
from autograd import elementwise_grad as egrad
from autograd import make_hvp as make_hvp
from autograd import jacobian
from autograd import hessian
from scipy.optimize import minimize

import inspect
from inspect import signature

f_hvp = hvp(f)
def f_hvp_wrap(x, p, xmat, targets, objscale, diff_weights):
    return f_hvp(x, p, xmat=xmat, targets=targets, objscale=objscale, diff_weights=diff_weights)

p = mtp.Problem(h=6, s=3, k=2)
p = mtp.Problem(h=10, s=3, k=2)
p = mtp.Problem(h=100, s=5, k=3)
p = mtp.Problem(h=1000, s=10, k=6)
p = mtp.Problem(h=10000, s=30, k=10)
p = mtp.Problem(h=20000, s=50, k=30)
p = mtp.Problem(h=30000, s=50, k=30)

xmat = p.xmat
wh = p.wh
targets = p.targets
h = xmat.shape[0]
s = targets.shape[0]
k = targets.shape[1]

diff_weights = get_diff_weights(targets)

# A = lil_matrix((h, h * s))
# for i in range(0, h):
#     A[i, range(i*s, i*s + s)] = 1
# A
# b = A.todense()  # ok to look at dense version if small

# fast way to fill A
# get i and j indexes of nonzero values, and data
inz = np.arange(0, h).repeat(s)
jnz = np.arange(0, s * h)
A = sp.sparse.coo_matrix((np.ones(h*s), (inz, jnz)), shape=(h, h * s))
# A2.todense() - A.todense()

A = A.tocsr()  # csr format is faster for our calculations
# A = A.tocsc()
lincon = sp.optimize.LinearConstraint(A, wh, wh)

wsmean = np.mean(wh) / targets.shape[0]
wsmin = np.min(wh) / targets.shape[0]
wsmax = np.max(wh)  # no state can get more than all of the national weight

objscale = 1

bnds = sp.optimize.Bounds(wsmin / 10, wsmax)

# starting values (initial weights), as an array
# x0 = np.full(h * s, 1)
# x0 = np.full(p.h * p.s, wsmean)
# initial weights that satisfy constraints
x0 = np.ones((h, s)) / s
x0 = np.multiply(x0, wh.reshape(x0.shape[0], 1)).flatten()

# verify that starting values satisfy adding-up constraint
np.square(np.round(x0.reshape((h, s)).sum(axis=1) - wh, 2)).sum()

# pv = x0 * 2
# pvr = pv[::-1]
# vec = f_hvp_wrap(x0, pvr, xmat, targets, objscale, diff_weights)

resapprox2 = minimize(f, x0,
               method='trust-constr',
               bounds=bnds,
               constraints=lincon,  # lincon lincon_feas
               jac=gfun,
               hess='2-point',
               # hessp=f_hvp_wrap,
               args=(xmat, targets, objscale, diff_weights),
               options={'maxiter': 100, 'verbose': 2,
                        'gtol': 1e-4, 'xtol': 1e-4,
                        'initial_tr_radius': 1,  # default 1
                        'factorization_method': 'AugmentedSystem'})

reshvp = minimize(f, x0,
               method='trust-constr',
               bounds=bnds,
               constraints=lincon,  # lincon lincon_feas
               jac=gfun,
               # hess='2-point',
               hessp=f_hvp_wrap,  # f_hvp_wrap wrap2
               args=(xmat, targets, objscale, diff_weights),
               options={'maxiter': 100, 'verbose': 2,
                        'gtol': 1e-4, 'xtol': 1e-4,
                        'initial_tr_radius': 1,  # default 1
                        'factorization_method': 'AugmentedSystem'})



# %% jac1

def f(x):
    return jnp.square(x)

def g(x):
    return jnp.square(x).sum()    

x = np.arange(10) * 3.7
f(x)
g(x)
jax.grad(g)(x)

j = ag.jacobian(f)
j(x)
2 * x

grad(f)(x)
j = jacobian(f)(x)
j.shape
np.diag(j)

np.nonzero(j)
nz = np.nonzero(j)
j[nz]

def hessian(f):
    return jacfwd(jacrev(f))

H = hessian(f)(x)

jacfwd(f)(x)
jacrev(f)(x)

print("hessian, with shape", H.shape)
print(H)

from jax import jit, jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))

# %% jac3
def f(x):
    try:
        if x < 3:
            return 2 * x ** 3
        else:
            raise ValueError
    except ValueError:
        return jnp.pi * x

y, f_vjp = vjp(f, 4.)
print(jit(f_vjp)(1.))



# %% jac
def f(x):
    return np.array([x[0]**2,x[1]**2])

x = np.array([[3.,11.],[5.,13.],[7.,17.]])

jac = jax.jacobian(f)
vmap_jac = jax.vmap(jac)
result = np.linalg.det(vmap_jac(x))
print(result)


jac = jax.jacobian(f)
jac(x)

jvmap = jax.vmap(jac)
jvmap(x)


# %% jac2


# Initialize random model coefficients
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (3,))
b = random.normal(b_key, ())

W = random.normal(100, (3,))

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

J = jacfwd(f)(W)
print("jacfwd result, with shape", J.shape)
print(J)

J = jacrev(f)(W)
print("jacrev result, with shape", J.shape)
print(J)
