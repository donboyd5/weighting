

# %% imports
# for checking:
# import sys; print(sys.executable)
# print(sys.path)
import importlib

import numpy as np
import gc  # gc.collect()

import scipy as sp
from scipy.optimize import lsq_linear
from numpy.random import seed
from timeit import default_timer as timer

import src.make_test_problems as mtp
import src.microweight as mw


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


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
    delta = np.log(wh / beta_x.sum(axis=0))
    return delta


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

def tpc_iter(wh, xmat, geotargets, beta_init, maxiter):
    scale_goal = 1e3
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = np.divide(xmat, scale_factors)
    geotargets = np.divide(geotargets, scale_factors)

    xpx = np.matmul(xmat.T, xmat)
    invxpx = np.linalg.inv(xpx) 

    beta0 = np.full(geotargets.shape, beta_init)
    delta0 = get_delta(wh, beta0, xmat)

    ebeta = beta0 # initial value
    edelta = delta0 # tpc uses initial delta based on initial beta 

    sse_vec = np.full(maxiter, np.nan)

    for iter in np.arange(maxiter):        
        edelta = get_delta(wh, ebeta, xmat)
        # save values for exit
        ebeta_prior = ebeta.copy()
        edelta_prior = edelta.copy()

        ewhs = get_geoweights(ebeta, edelta, xmat)
        ews = ewhs.sum(axis=0)

        etargets = np.matmul(ewhs.T, xmat)
        d = geotargets - etargets

        sse = np.square(d / geotargets * 100.0).sum()
        sse_vec[iter] = sse

        # step
        lhs_mutiplier = (-(1 / ews)).reshape((ews.size, 1))  
        step_tpc = np.multiply(lhs_mutiplier, np.matmul(d, invxpx))
        # step_tpc <- step_tpc * sscale
        
        ebeta = ebeta - step_tpc  # djb + or - ???

    # end of loop
    geotargets_opt = np.multiply(etargets, scale_factors)
    return ebeta_prior, edelta_prior, ewhs, geotargets_opt, d, sse_vec


# %% jax functions


# %% get problem 
p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=8, k=5, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=20, k=10, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=20000, s=30, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=30000, s=40, k=30, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4)


# %% now add noise to geotargets
np.random.seed(1)
gnoise = np.random.normal(0, .02, p.k * p.s)
gnoise = gnoise.reshape(p.geotargets.shape)
ngtargets = p.geotargets * (1 + gnoise)

# %% run the function
beta_opt, delta_opt, whs_opt, targets_opt, d_opt, sses = tpc_iter(p.wh, p.xmat, ngtargets, 0.5, 100)

sses
pdiff = d_opt / ngtargets * 100
np.round(pdiff, 2)
np.round(np.quantile(pdiff, q=qtiles), 2)

targets_opt.shape

# %% try optimize
sp.optimize.newton(func, x0, fprime=None, args=(), tol=1.48e-08, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True)


# %% prep data
geotargets = p.geotargets
wh = p.wh
xmat = p.xmat

scale_goal = 1e3
scale_factors = xmat.sum(axis=0) / scale_goal
xmat = np.divide(p.xmat, scale_factors)
geotargets = np.divide(p.geotargets, scale_factors)


# %% misc
xpx = np.matmul(xmat.T, xmat)
invxpx = np.linalg.inv(xpx) 
invxpx

beta_init = 0
beta0 = np.full(geotargets.shape, beta_init)
delta0 = get_delta(wh, beta0, xmat)

ebeta = beta0 # initial value
edelta = delta0 # tpc uses initial delta based on initial beta 

maxiter = 15
sse_vec = np.full(maxiter, np.nan)


# %% begin approach

iter = 0
iter += 1
for iter in np.arange(maxiter):
  edelta = get_delta(wh, ebeta, xmat)
  ewhs = get_geoweights(ebeta, edelta, xmat)
  ews = ewhs.sum(axis=0)
  ewh = ewhs.sum(axis=1)

  etargets = np.matmul(ewhs.T, xmat)
  d = geotargets - etargets

  sse = np.square(d).sum()
  
  # print(sprintf("iter: %i, sse: %.4e", iter, sse))
  sse_vec[iter] = sse
  sse_vec

  # ad hoc step
  lhs_mutiplier = (-(1 / ews)).reshape((ews.size, 1))  
  step_tpc = np.multiply(lhs_mutiplier, np.matmul(d, invxpx))
  # step_tpc <- -(1 / ews) * d %*% invxpx
  # step_tpc <- step_tpc * sscale
  
  # gradient step
  ebeta = ebeta + step_tpc # djb + or - ??


sse_vec
d / geotargets * 100



# %% r code below here
# unbundle the problem list and create additional variables needed
targets <- p$targets
wh <- p$wh
xmat <- p$xmat

xpx <- t(xmat) %*% xmat
invxpx <- solve(xpx) # TODO: add error check and exit if not invertible

beta0 <- matrix(0, nrow=nrow(targets), ncol=ncol(targets))
delta0 <- get_delta(wh, beta0, xmat) # tpc uses initial delta based on initial beta 

# scale notes ----
# h=100, s=20, k=8  100  4 steps jac, 6 steps tpc * 100
# h=1000, s=20, k=8  5 steps tpc * 1000
# h=2000, s=25, k=10 scale 2000  5 iter # findiff 23 secs, serial 42, par8 42; 
# h=4000, s=30, k=10 scale 4000
# h=6000, s=50, k=20

# start here ----
ebeta <- beta0 # tpc uses 0 as beta starting point
edelta <- delta0 # tpc uses initial delta based on initial beta 

maxiter <- 27
sse_vec <- rep(NA_real_, maxiter)
# iter <- 0
steps <- array(dim=c(2, length(ebeta), maxiter))
sscale <- nrow(xmat)
sscale <- nrow(xmat) * .4

# ACS hks 1000 4 5
# 1 1.7606e-01
# .9 1.5264e-04 1.1 6.3403e+03
# .95 1.9640e-04 .85 4.4871e-03

# ACS hks 10k 4 20
# 1 NAN at 7
# 1 NAN 6 scale 1000
# scale 1000 .75 sscale is good 2.0895e-06
# sscale .85 5.5032e-09
# sscale .90 6.5301e-05
# .95 9.0354e-08


for(iter in 1:maxiter){
  # iter <- iter + 1
  edelta <- get_delta(wh, ebeta, xmat)
  ewhs <- get_weights(ebeta, edelta, xmat)
  ews <- colSums(ewhs)
  ewh <- rowSums(ewhs)
  
  etargets <- t(ewhs) %*% xmat
  d <- targets - etargets
  
  sse <- sum(d^2)
  print(sprintf("iter: %i, sse: %.4e", iter, sse))
  sse_vec[iter] <- sse

  # ad hoc step
  step_tpc <- -(1 / ews) * d %*% invxpx
  step_tpc <- step_tpc * sscale
  
  # gradient step
  # step_gr1 <- numDeriv::grad(sse_fn, as.vector(ebeta), wh=wh, xmat=xmat, targets=targets)
  # step_grad <- vtom(step_gr1, nrow(ebeta)) * .01
  
  # jac <- jacobian(diff_vec, x=as.vector(ebeta), wh=wh, xmat=xmat, targets=targets)
  # ijac <- solve(jac)
  # step_jac <- vtom(as.vector(d) %*% ijac, nrow(ebeta))
  step_jac <- step_tpc

  steps[1, , iter] <- step_tpc
  steps[2, , iter] <- step_jac
  
  ebeta <- ebeta - step_tpc
}

# end of loop ----
