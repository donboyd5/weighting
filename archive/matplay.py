
# %% imports
from importlib import reload

# choose carefully
# import numpy as np
import numpy as onp
import jax.numpy as jnp
# import jax.numpy as np

import pandas as pd
import pickle
import jax

import scipy.optimize as spo

import src.make_test_problems as mtp
import src.microweight as mw


# %% reimports
# reload(pc)
# reload(rwp)
# reload(gwp)
reload(mw)



# %% constants
IGNOREDIR = '/media/don/ignore/'
WEIGHTDIR = IGNOREDIR + 'puf_versions/weights/'

qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)
opts = {'scaling': False}

# %% create problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10, s=3, k=2, xsd=.1, ssd=.5)

wh = p.wh
xmat = p.xmat
geotargets = p.geotargets
targs = p.targets
iwhs = p.whs

# %% alternatively get pickled problem
pkl_name = IGNOREDIR + 'pickle.pkl'
open_file = open(pkl_name, "rb")
pkl = pickle.load(open_file)
open_file.close()

targvars, ht2wide, pufsub, dropsdf_wide = pkl
wfname_national = WEIGHTDIR + 'weights2017_georwt1.csv'
wfname_national
final_national_weights = pd.read_csv(wfname_national)
pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])


# %% get stub and make problem from alternative data
stub = 6
qx = '(ht2_stub == @stub)'

pufstub = pufsub.query(qx)[['pid', 'ht2_stub'] + targvars]
# pufstub.replace({False: 0.0, True: 1.0}, inplace=True)
pufstub[targvars] = pufstub[targvars].astype(float)

# get targets and national weights
targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
whdf = pd.merge(pufstub[['pid']], final_national_weights[['pid', 'weight']], how='left', on='pid')

wh = whdf.weight.to_numpy()
onp.quantile(wh, qtiles)

xmat = pufstub[targvars].astype(float).to_numpy()
xmat
xmat[:, 0:7]
xmat.sum(axis=0)

geotargets = targetsdf[targvars].to_numpy()
(geotargets==0).sum()
geotargets = onp.where(geotargets==0, 1e3, geotargets)


# %% explore beta.dot(xmat)
beta0 = np.full(geotargets.shape, 0.0)
beta05 = np.full(geotargets.shape, 0.5)
beta1 = np.full(geotargets.shape, 1.0)

np.quantile(xmat, qtiles)  # negatives?
np.quantile(xmat.sum(axis=0), qtiles)

beta0.dot(xmat.T)
beta05.dot(xmat.T)
beta1.dot(xmat.T) # bigger

# try scaling
scale = 1 / xmat.sum(axis=0)
scale = 1 / np.abs(xmat).sum(axis=0)
xmat.sum(axis=0)
np.abs(xmat).sum(axis=0)
np.quantile(scale, qtiles)
xmat2 = np.multiply(xmat, scale)
xmat2
np.quantile(xmat2, qtiles)  # negatives?
np.quantile(xmat2.sum(axis=0), qtiles)

beta0.dot(xmat2.T)
beta05.dot(xmat2.T)
beta1.dot(xmat2.T) # bigger

np.quantile(beta0.dot(xmat2.T), qtiles)
np.quantile(beta05.dot(xmat2.T), qtiles)
np.quantile(beta1.dot(xmat2.T), qtiles)


wh.dot(xmat2)
geotargets2 = np.multiply(geotargets, scale)

# %% solve directly

wh
xmat
geotargets
dw = np.ones(geotargets.size)
bvstart = np.full(geotargets.size, 1e-6)

dw2 = np.ones(geotargets2.size)

lsspd = lambda x: sspd(x, wh, xmat, geotargets, dw)
lsspd = lambda x: sspd(x, wh, xmat2, geotargets2, dw2)
lsspd = lambda x: jsspd(x, wh, xmat2, geotargets2, dw2)
g = jax.grad(lsspd)
tmp = lsspd(bvstart)
tmp.dtype
tmp

g(bvstart)

result = spo.minimize(fun=lsspd,
    x0=bvstart,
    jac=g,
    method='BFGS')

result = spo.minimize(fun=lsspd,
    x0=bvstart,
    jac=g,
    method='L-BFGS-B')

dir(result)
result.success
result.x
bopt = result.x
onp.quantile(bopt, qtiles)

ltargs = lambda x: targets_diff(x, wh, xmat, geotargets, dw)
ltargs = lambda x: targets_diff(x, wh, xmat2, geotargets2, dw2)
jac = jax.jacfwd(ltargs)

result = spo.least_squares(
    fun=ltargs,
    x0=bvstart,
    method='trf', jac=jac, verbose=2,
    # ftol=opts.ftol, xtol=1e-7,
    # x_scale='jac',
    loss='soft_l1',  # linear, soft_l1, huber, cauchy, arctan,
    max_nfev=50)

prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=geotargets)
prob2 = mw.Microweight(wh=wh, xmat=xmat2, geotargets=geotargets2)
poisson_opts = {
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.5,
    'stepmethod': 'jac'}
poisson_opts.update({'x_scale': 'jac'})
poisson_opts.update({'max_nfev': 200})
poisson_opts.update({'x_scale': 1e-6})
poisson_opts.update({'x_scale': 1.0 / onp.abs(bvopt).flatten()})

gwp1 = prob2.geoweight(method='poisson-lsq', options=poisson_opts)
gwp1 = prob.geoweight(method='poisson-lsq', options=poisson_opts)
dir(gwp1)
gwp1.sspd
dir(gwp1.method_result)
bvopt = gwp1.method_result.beta_opt
onp.quantile(bvopt, qtiles)
np.round(1.0 / onp.abs(bvopt).flatten(), 1)

gx = prob2.geoweight(method='poisson-lbfgs', options=poisson_opts)
dir(gx)
gx.elapsed_seconds
gx.sspd

gi = prob2.geoweight(method='poisson-ipopt', options=poisson_opts)
dir(gi)
gi.elapsed_seconds
gi.sspd

# %% solve unscaled
prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=geotargets)
gw = prob.geoweight(method='poisson-lsq', options=opts)
gw.elapsed_seconds
gw.sspd
dir(gw.method_result)
beta = gw.method_result.beta_opt
delta = gw.method_result.delta_opt
whs_opt = gw.whs_opt

# all good
geotargets
iwhs.T.dot(xmat)
whs_opt.T.dot(xmat)

wh.shape # (h, )
xmat.shape # (h, k)
geotargets.shape  # (s, k)
targs.shape # (k, )

targs
wh.dot(xmat)  # (k, )

beta.dot(xmat.T)  # around 100, we want this to be small positive or small negative and give right results

# solve with hvp
# 'method': 'trust-ncg',  # trust-ncg, trust-krylov, or maybe Newton-CG
opts.update({'method': 'trust-krylov'})
opts.update({'method': 'trust-ncg'})
opts.update({'method': 'Newton-CG'})
opts
gwh = prob.geoweight(method='poisson-hvp', options=opts)
gwh.elapsed_seconds
gwh.sspd


opts = {'scaling': False}
opts = {'scaling': True}
gw = prob.geoweight(method='poisson-nelder', options=opts)
gw.elapsed_seconds
gw.sspd


# %% scale constant
# scale = 1 / xmat.sum(axis=0)
scale = 1 / np.abs(xmat).sum(axis=0)
xmat2 = np.multiply(xmat, scale)
xmat2
xmat2.shape
xmat2.sum(axis=0)

wh.dot(xmat2)
geotargets2 = np.multiply(geotargets, scale)

np.round(np.quantile(beta05.dot(xmat2.T), qtiles), 2)


prob2 = mw.Microweight(wh=wh, xmat=xmat2, geotargets=geotargets2)
gw2 = prob2.geoweight(method='poisson-lsq', options=opts)
gw2.sspd
gw2.elapsed_seconds
beta2 = gw2.method_result.beta_opt
beta2  # pretty close to 1
np.quantile(beta2.dot(xmat2.T), qtiles)  # ranges from -0.4 to + 0.3
delta2 = gw2.method_result.delta_opt
delta2  # close to 2
whs_opt2 = gw2.whs_opt
whs_opt2 - whs_opt
geotargets2_opt = whs_opt2.T.dot(xmat)  # use original xmat
# compare to original geo targets
p.geotargets
geotargets2_opt

# solve with hvp
# 'method': 'trust-ncg',  # trust-ncg, trust-krylov, or maybe Newton-CG
opts.update({'method': 'trust-krylov'})
opts.update({'method': 'trust-ncg'})
opts.update({'method': 'Newton-CG'})
opts
gw2h = prob2.geoweight(method='poisson-hvp', options=opts)
gw2h.elapsed_seconds
gw2h.sspd

# %% scale zscore
scale = 1 / xmat.sum(axis=0)
xmat2 = np.multiply(xmat, scale)
xmat2
xmat2.shape
xmat2.sum(axis=0)

wh.dot(xmat2)
geotargets2 = np.multiply(geotargets, scale)

np.round(np.quantile(beta05.dot(xmat2.T), qtiles), 2)


prob2 = mw.Microweight(wh=wh, xmat=xmat2, geotargets=gt2)
gw2 = prob2.geoweight(method='poisson-lsq', options=opts)
gw2.sspd
beta2 = gw2.method_result.beta_opt
beta2  # pretty close to 1
np.quantile(beta2.dot(xmat2.T), qtiles)  # ranges from -0.4 to + 0.3
delta2 = gw2.method_result.delta_opt
delta2  # close to 2
whs_opt2 = gw2.whs_opt
whs_opt2 - whs_opt
geotargets2_opt = whs_opt2.T.dot(xmat)  # use original xmat
# compare to original geo targets
geotargets
geotargets2_opt


# %% notes

# https://stackoverflow.com/questions/24767191/scipy-is-not-optimizing-and-returns-desired-error-not-necessarily-achieved-due
# Watch out for negative values of the log() function, resolve them and tell the optimizer that they are bad, by adding a penalty:

# #!/usr/bin/python
# import math
# import random
# import numpy as np
# from scipy.optimize import minimize

# def loglikelihood(params, data):
#     (mu, alpha, beta) = params
#     tlist = np.array(data)
#     r = np.zeros(len(tlist))
#     for i in xrange(1,len(tlist)):
#         r[i] = math.exp(-beta*(tlist[i]-tlist[i-1]))*(1+r[i-1])
#     loglik = -tlist[-1]*mu
#     loglik += alpha/beta*sum(np.exp(-beta*(tlist[-1]-tlist))-1)
#     argument = mu + alpha * r
#     limit = 1e-6
#     if np.min(argument) < limit:
#         # add a penalty for too small argument of log
#         loglik += np.sum(np.minimum(0.0, argument - limit)) / limit
#         # keep argument of log above the limit
#         argument = np.maximum(argument, limit)
#     loglik += np.sum(np.log(argument))
#     return -loglik

# atimes = [ 148.98894201,  149.70253172,  151.13717804,  160.35968355,
#         160.98322609,  161.21331798,  163.60755544,  163.68994973,
#         164.26131871,  228.79436067]
# a= 0.01
# alpha = 0.5
# beta = 0.6
# print loglikelihood((a, alpha, beta), atimes)

# res = minimize(loglikelihood, (0.01, 0.1,0.1), method = 'BFGS',args = (atimes,))
# print res


# I have the following code which attempts to minimize a log likelihood function.

# #!/usr/bin/python
# import math
# import random
# import numpy as np
# from scipy.optimize import minimize

# def loglikelihood(params, data):
#     (mu, alpha, beta) = params
#     tlist = np.array(data)
#     r = np.zeros(len(tlist))
#     for i in xrange(1,len(tlist)):
#         r[i] = math.exp(-beta*(tlist[i]-tlist[i-1]))*(1+r[i-1])
#     loglik  = -tlist[-1]*mu
#     loglik = loglik+alpha/beta*sum(np.exp(-beta*(tlist[-1]-tlist))-1)
#     loglik = loglik+np.sum(np.log(mu+alpha*r))
#     return -loglik

# atimes = [ 148.98894201,  149.70253172,  151.13717804,  160.35968355,
#         160.98322609,  161.21331798,  163.60755544,  163.68994973,
#         164.26131871,  228.79436067]
# a= 0.01
# alpha = 0.5
# beta = 0.6
# print loglikelihood((a, alpha, beta), atimes)

# res = minimize(loglikelihood, (0.01, 0.1,0.1), method = 'BFGS',args = (atimes,))
# print res
# It gives me

# 28.3136498357
# ./test.py:17: RuntimeWarning: invalid value encountered in log
#   loglik = loglik+np.sum(np.log(mu+alpha*r))
#    status: 2
#   success: False
#      njev: 14
#      nfev: 72
#  hess_inv: array([[1, 0, 0],
#        [0, 1, 0],
#        [0, 0, 1]])
#       fun: 32.131359359964378
#         x: array([ 0.01,  0.1 ,  0.1 ])
#   message: 'Desired error not necessarily achieved due to precision loss.'
#       jac: array([ -2.8051672 ,  13.06962156, -48.97879982])
# Notice that it hasn't managed to optimize the parameters at all and the minimized value 32 is bigger than 28 which is what you get with a= 0.01, alpha = 0.5, beta = 0.6 . It's possible this problem could be avoided by choosing better initial guesses but if so, how can I do this automatically?


# I copied your example and tried a little bit. Looks like if you stick with BFGS solver, after a few iteration the mu+ alpha * r will have some negative numbers, and that's how you get the RuntimeWarning.

import math
import random
import numpy as np
from scipy.optimize import minimize


def loglikelihood1(params, data):
    (mu, alpha, beta) = params
    tlist = np.array(data)
    r = np.zeros(len(tlist))
    for i in range(1,len(tlist)):
        r[i] = math.exp(-beta*(tlist[i]-tlist[i-1]))*(1+r[i-1])
    loglik  = -tlist[-1]*mu
    loglik = loglik+alpha/beta*sum(np.exp(-beta*(tlist[-1]-tlist))-1)
    loglik = loglik+np.sum(np.log(mu+alpha*r))
    return -loglik

def loglikelihood2(params, data):
    (mu, alpha, beta) = params
    tlist = np.array(data)
    r = np.zeros(len(tlist))
    for i in range(1,len(tlist)):
        r[i] = math.exp(-beta*(tlist[i]-tlist[i-1]))*(1+r[i-1])
    loglik = -tlist[-1]*mu
    loglik += alpha/beta*sum(np.exp(-beta*(tlist[-1]-tlist))-1)
    argument = mu + alpha * r
    limit = 1e-6
    if np.min(argument) < limit:
        # add a penalty for too small argument of log
        # print("/n")
        print('\n', argument)
        print(np.min(argument))
        print("LL before", loglik)
        print('presum', np.minimum(0.0, argument - limit))
        print('sum', np.sum(np.minimum(0.0, argument - limit)))
        print('penalty', np.sum(np.minimum(0.0, argument - limit)) / limit)
        loglik += np.sum(np.minimum(0.0, argument - limit)) / limit
        print("LL after", loglik)
        # keep argument of log above the limit
        argument = np.maximum(argument, limit)
    loglik += np.sum(np.log(argument))
    return -loglik

def ll3(params, data):
    (mu, alpha, beta) = params
    tlist = np.array(data)
    r = np.zeros(len(tlist))
    for i in range(1,len(tlist)):
        r[i] = math.exp(-beta*(tlist[i]-tlist[i-1]))*(1+r[i-1])
    loglik  = -tlist[-1]*mu
    loglik = loglik+alpha/beta*sum(np.exp(-beta*(tlist[-1]-tlist))-1)
    loglik = loglik+np.sum(np.log(mu+alpha*r))
    loglik = loglik / 30.0
    return -loglik


atimes = [ 148.98894201,  149.70253172,  151.13717804,  160.35968355,
        160.98322609,  161.21331798,  163.60755544,  163.68994973,
        164.26131871,  228.79436067]
a= 0.01
alpha = 0.5
beta = 0.6
loglikelihood1((a, alpha, beta), atimes)
loglikelihood2((a, alpha, beta), atimes)

res1 = minimize(loglikelihood1, (0.01, 0.1,0.1), method = 'BFGS',args = (atimes,))
res1

res2 = minimize(loglikelihood2, (0.01, 0.1,0.1), method = 'BFGS',args = (atimes,))
res2

res3 = minimize(loglikelihood1, (0.01, 0.1,0.1), method = 'Nelder-Mead',args = (atimes,))
res3

# Two solutions:
# 1) Scale your log-likelihood and gradients by a factor, like 1/n where n is the number of samples.
# 2) Scale your gtol: for example "gtol": 1e-7 * n

res4_1 = minimize(loglikelihood1, (0.01, 0.1,0.1), method = 'BFGS',args = (atimes,))
res4_1
res4_2 = minimize(loglikelihood1, (0.01, 0.1,0.1), method = 'BFGS',args = (atimes,), options={'gtol': 1e-7*10.0})
res4_2


# Facing the same warning, I solved it by rewriting the log-likelihood function to get log(params) and log(data) as arguments, instead of params and data.

# Thus, I avoid using np.log() in the likelihood function or Jacobian, if possible.
lparams = np.log((0.01, 0.1, 0.1))
ldata = np.log(atimes)
ldata[-1]
(lparams, )
def ll5(lparams, ldata):
    (mu, alpha, beta) = lparams
    tlist = np.array(ldata)
    r = np.zeros(len(tlist))
    for i in range(1,len(tlist)):
        r[i] = math.exp(-beta*(tlist[i]-tlist[i-1]))*(1+r[i-1])
    loglik  = -tlist[-1]*mu
    loglik = loglik+alpha/beta*sum(np.exp(-beta*(tlist[-1]-tlist))-1)
    loglik = loglik+np.sum(np.log(mu+alpha*r))
    return -loglik



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
    # print("before betax")
    # print(np.quantile(beta, [0, .1, .25, .5, .75, .9, 1]))

    # import pickle
    # save_list = [beta, xmat]
    # save_name = '/home/donboyd/Documents/beta_xmat.pkl'
    # open_file = open(save_name, "wb")
    # pickle.dump(save_list, open_file)
    # open_file.close()

    beta_x = np.exp(np.dot(beta, xmat.T))
    # print("after betax")

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
    # diffs = diffs * diff_weights
    diffs = np.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs

def sspd(beta_object, wh, xmat, geotargets, diff_weights):
    sspd = np.square(targets_diff(beta_object, wh, xmat, geotargets, diff_weights)).sum()
    return sspd

# %% jax functions
def get_delta(wh, beta, xmat):
    beta_x = jnp.exp(jnp.dot(beta, xmat.T))
    delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta

def get_diff_weights(geotargets, goal=100):
    goalmat = jnp.full(geotargets.shape, goal)
    diff_weights = jnp.where(geotargets != 0, goalmat / geotargets, 1)
    return diff_weights

def get_geotargets(beta, wh, xmat):
    delta = get_delta(wh, beta, xmat)
    whs = get_geoweights(beta, delta, xmat)
    targets_mat = jnp.dot(whs.T, xmat)
    return targets_mat


def get_geoweights(beta, delta, xmat):
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

    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    geotargets_calc = get_geotargets(beta, wh, xmat)
    diffs = geotargets_calc - geotargets
    # diffs = diffs * diff_weights
    diffs = np.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs

def sspd(beta_object, wh, xmat, geotargets, diff_weights):
    sspd = np.square(targets_diff(beta_object, wh, xmat, geotargets, diff_weights)).sum()
    return sspd


# %% more
