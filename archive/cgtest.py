# test conjugate gradient

# %% urls
# syntax https://jax.readthedocs.io/en/latest/jax.html
# https://github.com/google/jax/issues/4753
# https://github.com/hanrach/p2d_solver/blob/main/run_ex.py

# https://sajay.online/posts/fun_with_jax/

# https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/slides/lec02.pdf

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.matvec.html

# %% solving Ax=b with conjugate gradient

# %% roger grosse
# https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/
# see https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/slides/lec02.pdf
# ~ p.40
# A must be PSD
# https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/readings/L02_Taylor_approximations.pdf


# def approx_solve(A_mvp, b, niter):
#   dim = b.size
#   A_linop = scipy.sparse.linalg.LinearOperator((dim,dim), matvec=A_mvp)
#   res = scipy.sparse.linalg.cg(A_linop, b, maxiter=niter)
#   return res[0]


# %% preconditioning
# https://scicomp.stackexchange.com/questions/27450/correct-use-of-scipys-sparse-linalg-spilu
# https://stackoverflow.com/questions/46876951/sparse-matrix-solver-with-preconditioner

# from scipy.sparse.linalg import LinearOperator, spilu
# ilu = spilu(A)
# Mx = lambda x: ilu.solve(x)
# M = LinearOperator((N, N), Mx)


# %% jacfwd documentation
# syntax https://jax.readthedocs.io/en/latest/jax.html
# jax.eval_shape(f, A, x)  # get shape with no FLOPs
# jax.jacfwd(fun, argnums=0, holomorphic=False)
#   Jacobian of fun evaluated column-by-column using forward-mode AD.
#   Parameters
#       fun (Callable) – Function whose Jacobian is to be computed.
#       argnums (Union[int, Sequence[int]]) – Optional, integer or sequence
#           of integers. Specifies which positional argument(s) to differentiate with respect
#           to (default 0).
#       holomorphic (bool) – Optional, bool. Indicates whether fun is promised to be holomorphic. Default False.
#   Return type Callable
#   Returns
#       A function with the same arguments as fun, that evaluates the Jacobian 
#       of fun using forward-mode automatic differentiation.

# %% jax jvp documentation
# syntax https://jax.readthedocs.io/en/latest/jax.html
# jax.jvp(fun, primals, tangents)
# Computes a (forward-mode) Jacobian-vector product of fun.
# Parameters
#   fun (Callable) – Function to be differentiated. Its arguments should be arrays,
#       scalars, or standard Python containers of arrays or scalars. It should
#       return an array, scalar, or standard Python container of arrays or scalars.
#   primals – The primal values at which the Jacobian of fun should be evaluated.
#       Should be either a tuple or a list of arguments, and its length should
#       equal to the number of positional parameters of fun.
#   tangents – The tangent vector for which the Jacobian-vector product should
#       be evaluated. Should be either a tuple or a list of tangents, with
#       the same tree structure and array shapes as primals.
# Return type   Tuple[Any, Any]
# Returns
#   A (primals_out, tangents_out) pair, where primals_out is fun(*primals),
#   and tangents_out is the Jacobian-vector product of function evaluated 
#   at primals with tangents. The tangents_out value has the same Python
#   tree structure and shapes as primals_out.

# %% jax vjp documentation
# https://jax.readthedocs.io/en/latest/jax.html#jax.vjp
# source https://jax.readthedocs.io/en/latest/_modules/jax/_src/api.html#vjp
# jax.vjp(fun: Callable[[…], Any], *primals: Any, has_aux: bool) → Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]
# Compute a (reverse-mode) vector-Jacobian product of fun.

# grad() is implemented as a special case of vjp().

# Parameters
#   fun (Callable) – Function to be differentiated. Its arguments should be arrays, scalars, or 
#       standard Python containers of arrays or scalars. It should return an array, scalar, or standard
#        Python container of arrays or scalars.

#   primals – A sequence of primal values at which the Jacobian of fun should be evaluated. The length 
#       of primals should be equal to the number of positional parameters to fun. Each primal value 
#       should be a tuple of arrays, scalar, or standard Python containers thereof.

#   has_aux (bool) – Optional, bool. Indicates whether fun returns a pair where the first element is 
#       considered the output of the mathematical function to be differentiated and the second element 
#       is auxiliary data. Default False.

# Return type  Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]

# Returns
#   If has_aux is False, returns a (primals_out, vjpfun) pair, where primals_out is fun(*primals). 
#   vjpfun is a function from a cotangent vector with the same shape as primals_out to a tuple of 
#   cotangent vectors with the same shape as primals, representing the vector-Jacobian product of 
#   fun evaluated at primals. If has_aux is True, returns a (primals_out, vjpfun, aux) tuple where 
#   aux is the auxiliary data returned by fun.

# >>> import jax
# >>>
# >>> def f(x, y):
# ...   return jax.numpy.sin(x), jax.numpy.cos(y)
# ...
# >>> primals, f_vjp = jax.vjp(f, 0.5, 1.0)
# >>> xbar, ybar = f_vjp((-0.7, 0.3))
# >>> print(xbar)
# -0.61430776
# >>> print(ybar)
# -0.2524413

# def f(x, y):
#     return jax.numpy.sin(x), jax.numpy.cos(y)

# jax.numpy.sin(0.5)
# f(0.5, 1.0)
# primals, f_vjp = jax.vjp(f, 0.5, 1.0)
# xbar, ybar = f_vjp((-0.7, 0.3))

a = 0.5
b = 1.0
a = -0.7
b = 0.3
c = 2.0
d = 3.0
vp = jax.vjp(f, a, b)
vp((c, d))




# %% scipy cg documentation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
# scipy.sparse.linalg.cg(A, b, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)
# Use Conjugate Gradient iteration to solve Ax = b.

# Parameters
# A{sparse matrix, dense matrix, LinearOperator}
# The real or complex N-by-N matrix of the linear system. A must represent a hermitian, 
# positive definite matrix. Alternatively, A can be a linear operator which can produce Ax 
# using, e.g., scipy.sparse.linalg.LinearOperator.

# b{array, matrix}
# Right hand side of the linear system. Has shape (N,) or (N,1).

# Returns
# x{array, matrix}
# The converged solution.

# infointeger
# Provides convergence information:
# 0 : successful exit >0 : convergence to tolerance not achieved, number of iterations <0 : illegal input or breakdown

# Other Parameters
# x0{array, matrix}
# Starting guess for the solution.

# tol, atolfloat, optional
# Tolerances for convergence, norm(residual) <= max(tol*norm(b), atol). The default for atol is 'legacy', which emulates a different legacy behavior.

# Warning
# The default value for atol will be changed in a future release. For future compatibility, specify atol explicitly.

# maxiterinteger
# Maximum number of iterations. Iteration will stop after maxiter steps even if the specified tolerance has not been achieved.

# M{sparse matrix, dense matrix, LinearOperator}
# Preconditioner for A. The preconditioner should approximate the inverse of A. Effective 
# preconditioning dramatically improves the rate of convergence, which implies that fewer 
# iterations are needed to reach a given error tolerance.

# callbackfunction
# User-supplied function to call after each iteration. It is called as callback(xk), where xk is the current solution vector.


# %% jax cg documentation

# https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html
# source: https://jax.readthedocs.io/en/latest/_modules/jax/_src/scipy/sparse/linalg.html#cg
# jax.scipy.sparse.linalg.cg(A, b, x0=None, *, tol=1e-05, atol=0.0, maxiter=None, M=None)
# djb note the * after x0

# you need to supply the linear operator A as a function instead of a sparse
# matrix or LinearOperator
# A is a 2D array or function that calculates the linear map (matrix-vector product)
# Ax when called like A(x)
# A must represent a hermitian, positive definite matrix, and must return array(s) 
# with the same structure and shape as its argument.
# djb A is really a function

# b (array or tree of arrays) – Right hand side of the linear system representing 
# a single vector. Can be stored as an array or Python container of array(s) with any shape.

# 


# %% notes to self
# possibly investigate preconditioning for conjugate gradient


# %% imports
import inspect
import scipy

import numpy as np
from numpy.linalg import norm
import jax
import jax.numpy as jnp
from timeit import default_timer as timer

# from numpy.linalg import solve
# from numpy.linalg import norm
from jax import jacfwd
from jax import jvp, vjp

# this next line is CRUCIAL or we will lose precision
from jax.config import config
config.update('jax_enable_x64', True)

from timeit import default_timer as timer

from collections import namedtuple

import utilities as ut # src.utilities
import make_test_problems as mtp  # src.make_test_problems


# %% test without parameters
# a 1-vector objective function
obj =lambda x: x**3
obj(5)
f = lambda z: obj(z)
inspect.getsource(f)
f(2)
f(3)

# define the jacobian
j = jacfwd(obj)
j(2.)  # 3x**2 so j(2.) is 12
j(3.)

# vector input
x = np.array([2., 3., 4.])
f(x) # x**3
j(x)  # 3x**2  -- 12, 27, 48

# direct calculation of jacobian vector product
y = np.array([3., 4., 5.])
j(x).dot(y)  # 12*3, 27*4, 48*5

# jax jvp
jvp(f, (x,), (y,))[1]  # same result

# solve for step directly
z = f(x)
# jvals is j evaluated at x
jvals = j(x)
np.linalg.solve(jvals, z)
np.linalg.lstsq(jvals, z, rcond=None)[0]
# [0.66666667, 1.        , 1.33333333]

# use scipy cg to solve for step
scipy.sparse.linalg.cg(np.array(jvals), z)[0]  # same answer

# use jax scipy cg to solve for step
# first with matrix
jax.scipy.sparse.linalg.cg(jvals, z)[0]  # same answer

# jvp function using scipy
jvpfn = lambda y: jvp(f, (x,), (y,))[1]
jvpfn(z)
jvp_linop = scipy.sparse.linalg.LinearOperator((z.size, z.size), matvec=jvpfn)
scipy.sparse.linalg.cg(jvp_linop, z)

# jvp function using jax
jvpfn = lambda y: jvp(f, (x,), (y,))[1]
jvpfn(z)
jax.scipy.sparse.linalg.cg(jvpfn, z)[0]  # same answer



# %% repeat the test with, parameters
# a 1-vector objective function
obj2 =lambda xa, xb: xa**3 + xb**2
obj2(5, 3)
f2 = lambda za: obj2(za, xb)
inspect.getsource(f2)
xb = 3.
f2(2.)
f2(3.)

# define the jacobian
j2 = jacfwd(obj2)
inspect.getsource(j2)
# deriv wrt x??
j2(2., 3.)  # 3x**2, 2y  so j2(2., 3.) is 12
j2(3., 2.)

# vector input
xa = np.array([2., 3., 4.])
xb = np.array([3., 4., 5.])
obj2(xa, xb) # good
f2(xa) # good
f2(xb) # good
j2(xa, xb)

# direct calculation of jacobian vector product
y2 = np.array([4., 5., 6])
j2(xa, xb).dot(y2)

# jax jvp
jvp(f2, (xa,), (y2,))[1]  # same result, uses xb in the environment

# solve for step directly
z2 = f2(xa)
# jvals2 is j2 evaluated at xa, xb
jvals2 = j2(xa, xb)
jvals2
np.linalg.solve(jvals2, z2)
np.linalg.lstsq(jvals2, z2, rcond=None)[0]  # same answer
# array([1.41666667, 1.59259259, 1.85416667])

# use scipy cg to solve for step
scipy.sparse.linalg.cg(np.array(jvals2), z2)[0]  # same answer

# use jax scipy cg to solve for step
# first with matrix
jax.scipy.sparse.linalg.cg(jvals2, z2)[0]  # same answer

# jvp function using scipy  (xb in the environment)
jvpfn2 = lambda y2: jvp(f2, (xa,), (y2,))[1]  # xb in environment
jvpfn2(z2)
jvpfn2(y2)  # same as j2(xa, xb).dot(y2)
jvp_linop2 = scipy.sparse.linalg.LinearOperator((z2.size, z2.size), matvec=jvpfn2)
scipy.sparse.linalg.cg(jvp_linop2, z2) # good!
# (array([1.41666667, 1.59259259, 1.85416667]), 0)

# jvp function using jax
jax.scipy.sparse.linalg.cg(jvpfn2, z2)[0]  # same answer

# djb here

# %% use a more complex function

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

def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = np.divide(xmat, scale_factors)
    geotargets = np.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors


# %% create problem
p = mtp.Problem(h=30, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)    
p = mtp.Problem(h=30, s=3, k=2, xsd=.1, ssd=.5, pctzero=.0)    
p = mtp.Problem(h=1000, s=10, k=4, xsd=.1, ssd=.5, pctzero=.4)    
p = mtp.Problem(h=10000, s=20, k=8, xsd=.1, ssd=.5, pctzero=.4)   
p = mtp.Problem(h=20000, s=25, k=20, xsd=.1, ssd=.5, pctzero=.4)    
p = mtp.Problem(h=30000, s=35, k=30, xsd=.1, ssd=.5, pctzero=.4)    
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4) 
p = mtp.Problem(h=50000, s=50, k=40, xsd=.1, ssd=.5, pctzero=.3) 
p = mtp.Problem(h=60000, s=80, k=50, xsd=.1, ssd=.5, pctzero=.3) 
p = mtp.Problem(h=100000, s=100, k=50, xsd=.1, ssd=.5, pctzero=.3) 

p.h
p.s
p.k

xmat, geotargets, scale_factors = scale_problem(p.xmat, p.geotargets, 1000.0)

# xmat = p.xmat
# geotargets = p.geotargets

wh = p.wh

np.random.seed(1)
gnoise = np.random.normal(0, .01, p.k * p.s)
gnoise = gnoise.reshape(geotargets.shape)
ngtargets = geotargets * (1 + gnoise)
ngtargets

# ngtargets = geotargets

dw = jax_get_diff_weights(ngtargets)

betavec0 = np.full(geotargets.size, 0.5, dtype='float64')  # 1e-13 or 1e-12 seems best
betavec0 = np.full(geotargets.size, 0.0, dtype='float64')  # 1e-13 or 1e-12 seems best
betavec0 = np.full(geotargets.size, 1.0, dtype='float64')  # 1e-13 or 1e-12 seems best
betavec0 = np.full(geotargets.size, 1e-10, dtype='float64')  # 1e-13 or 1e-12 seems best



# %% Newton step

# do this first as functions depend upon it
bvec = betavec0.copy()

# create lambda function of xbvec, which will vary within the loop
# wh, xmat, ngtargets, dw do not change within the loop so we can define outside loop
ldiffs = lambda xbvec: jax_targets_diff(xbvec, wh, xmat, ngtargets, dw)

# define jacobian
jdiffs = jacfwd(jax_targets_diff)

# jvp lambda
jvpfn3 = lambda yvar: jvp(ldiffs, (bvec,), (yvar,))[1]
# vjb lambda note different signature and syntax
vjpfn3 = lambda yvar: vjp(ldiffs, bvec)[1](yvar)


ldiffs(bvec)
jvpfn3(bvec)
vjpfn3(bvec)

jvp(ldiffs, bvec)(bvec)
vjp(ldiffs, bvec)

jvp_linop3a = scipy.sparse.linalg.LinearOperator((bvec.size, bvec.size), matvec=jvpfn3, rmatvec=jvpfn3)
jvp_linop3b = scipy.sparse.linalg.LinearOperator((bvec.size, bvec.size), matvec=jvpfn3, rmatvec=vjpfn3)

# vjp lambda
# J2T_op = lambda v: jax.vjp(f, x2)[1](v)[0]



# start loop
y3 = jax_targets_diff(bvec, wh, xmat, ngtargets, dw)  # resids
# ldiffs(bvec)  # good, same

# res = jnp.linalg.norm(y / jnp.linalg.norm(betavec, np.inf), np.inf)
res = jnp.square(y3).sum()
res

# solve for step directly

# jvals3 is jdiffs (jacobian) evaluated here
jvals3 = jdiffs(bvec, wh, xmat, ngtargets, dw)
jvals3
# np.linalg.cholesky(jvals3) # error if not positive definite

np.linalg.solve(jvals3, y3)
# np.linalg.inv(jvals3) # what does the inverse look like?
# array([-0.05050206, -0.07945673, -0.04930957, -0.08029163, -0.0492151 , -0.07998717])
np.linalg.lstsq(jvals3, y3, rcond=None)[0]  # NOT!! same answer
jnp.linalg.lstsq(jvals3, y3, rcond=None)[0]  # NOT!! same answer
# array([-8.26486139e-04,  4.55111163e-04,  3.66004531e-04, -3.79782024e-04, 4.60481608e-04, -7.53291394e-05])
np.linalg.cond(jvals3)  # condition number very high!!

# use scipy cg to solve for step
scipy.sparse.linalg.cg(np.array(jvals3), y3)[0] # similar to lstsq, not solve
# array([-8.06123019e-04,  4.57304524e-04,  3.86365601e-04, -3.77587343e-04, 4.80842989e-04, -7.31356235e-05])

# from scipy.sparse.linalg import LinearOperator, spilu
# ilu = scipy.sparse.linalg.spilu(jvals3)
# Mx = lambda x: ilu.solve(x)
# M = scipy.sparse.linalg.LinearOperator((y3.size, y3.size), Mx)
# scipy.sparse.linalg.cg(np.array(jvals3), y3, M=M)
# M = np.linalg.inv(jvals3)
# scipy.sparse.linalg.gmres(np.array(jvals3), y3)[0] 

# use jax scipy cg to solve for step
# first with matrix
jax.scipy.sparse.linalg.cg(jvals3, y3)[0]  # similar to lstsq
# DeviceArray([-8.06123019e-04,  4.57304524e-04,  3.86365601e-04, -3.77587343e-04,  4.80842989e-04, -7.31356235e-05],            dtype=float64)

# jvp function using scipy (vars in the environment)
# jvpfn3 = lambda yvar: jvp(ldiffs, (bvec,), (yvar,))[1]  # vars in environment
# jvpfn3(y3)  # should be same as j2(xa, xb).dot(y2)
jvp_linop3 = scipy.sparse.linalg.LinearOperator((y3.size, y3.size), matvec=jvpfn3)
scipy.sparse.linalg.cg(jvp_linop3, y3) # like lstsq
# (array([-8.06123019e-04,  4.57304524e-04,  3.86365601e-04, -3.77587343e-04,    4.80842989e-04, -7.31356235e-05]),0)

# jvp_linop3a = scipy.sparse.linalg.LinearOperator((y3.size, y3.size), matvec=jvpfn3, rmatvec=jvpfn3)
# %timeit scipy.sparse.linalg.lsqr(jvp_linop3a, y3)
# %timeit scipy.optimize.lsq_linear(jvp_linop3a, y3)

# jnp.linalg.lstsq(jvals3, y3, rcond=None)[0]

start = timer()
# %timeit step_res = scipy.optimize.lsq_linear(jvp_linop3a, y3)
# %timeit step_resb = scipy.optimize.lsq_linear(jvp_linop3b, y3)
# step_res = scipy.optimize.lsq_linear(jvp_linop3a, y3)
step_resb = scipy.optimize.lsq_linear(jvp_linop3b, y3)
end = timer()
end - start
# dir(step_res)
# step_res.x
step_res.success
step_resb.success
step_resb.x

# scipy.sparse.linalg.gmres(jvp_linop3, y3)

# jvp function using jax
# jax.scipy.sparse.linalg.cg(jvpfn3, y3)[0]

# step = jax.scipy.sparse.linalg.cg(jvpfn3, y3)[0]

# step = jnp.linalg.lstsq(jvals3, y3, rcond=None)[0]

step = step_res.x
step = step_resb.x

step
bvec = bvec - step
bvec

