
# https://github.com/google/jax/issues/4753
# https://github.com/hanrach/p2d_solver/blob/main/run_ex.py

import jax
import numpy as np
import jax.numpy as np
import timeit

from numpy.linalg import solve
from numpy.linalg import norm
from jax import jacfwd
from jax import jvp

import jax
import jax.numpy as np
from jax import jacfwd
from jax.config import config
config.update('jax_enable_x64', True)
from jax.numpy.linalg import norm
from jax.scipy.linalg import solve
from p2d_param import get_battery_sections  # no
import timeit
from settings import Tref  # no 
from unpack import unpack, unpack_fast  # no
from scipy.sparse import csr_matrix, csc_matrix
from unpack import unpack  # no

import jax
from jax.scipy.linalg import solve
import jax.numpy as np
from jax.numpy.linalg import norm
from jax import jacrev
from jax.config import config
config.update('jax_enable_x64', True)
from p2d_main_fn import p2d_fn
#from res_fn_order2 import fn
from residual import ResidualFunction

def fn(x):
    return np.square(x - 1)

jac_fn = jax.jit(jacfwd(fn))
fn = jax.jit(fn)
# where fn takes a vector of size ~5000. Inside newton's method,

maxit = 20
tol = 1e-6
U0 = np.array([0., 1., 2., 3., 4])

# djb
U = U0
count = 0
res = 1e9
while(count < maxit and  res > tol):
    start1 =timeit.default_timer() 
    J =  jac_fn(U)
    y = fn(U)
    res = norm(y / norm(U, np.inf), np.inf)
    # delta = solve(J, y)
    delta = np.linalg.lstsq(J, y, rcond=None)[0]
    U = U - delta
    count = count + 1
    end1 =timeit.default_timer() 
    print(count, res)

U

# djb2 this works
# So in summary, if you replace delta = solve(J,y) with

# jac_x_prod = lambda x: jvp(fn, y, x)
# delta = jax.scipy.sparse.linalg.cg(jax_x_prod, y)[0]
U = U0
count = 0
res = 1e9
while(count < maxit and  res > tol):
    start1 =timeit.default_timer() 
    # J =  jac_fn(U)
    y = fn(U)
    res = norm(y / norm(U, np.inf), np.inf)
    # delta = solve(J, y)
    # delta = np.linalg.lstsq(J, y, rcond=None)[0]
    # djb jax.jvp(f, (x,), (s,))[1]
    # jac_x_prod = lambda x: jvp(fn, y, x)
    # delta = jax.scipy.sparse.linalg.cg(jax_x_prod, y)[0]
    # x = U
    # jac_x_prod = lambda x: jvp(fn, (y,), (x,))
    jac_x_prod = lambda x: jvp(fn, (U,), (x,))[1]  # djb
    # jac_x_prod(U)
    # jac_x_prod = lambda x: jvp(fn, y, x) # what jax said
    # jac_x_prod(y)[1] # returns 2 arrays of size 5
    delta = jax.scipy.sparse.linalg.cg(jac_x_prod, y)[0]
    U = U - delta
    count = count + 1
    end1 =timeit.default_timer() 
    print(count, res)

jac_x_prod(U0)
jac_x_prod(U)

# https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html
# jax.scipy.sparse.linalg.cg(A, b, x0=None, *, tol=1e-05, atol=0.0, maxiter=None, M=None)

# you need to supply the linear operator A as a function instead of a sparse
# matrix or LinearOperator
# A is a 2D array or function that calculates the linear map (matrix-vector product)
# Ax when called like A(x)
# A must represent a hermitian, positive definite matrix, and must return array(s) 
# with the same structure and shape as its argument.
# djb A is really a function

# b (array or tree of arrays) – Right hand side of the linear system representing 
# a single vector. Can be stored as an array or Python container of array(s) with any shape.

# FilteredStackTrace: ValueError: 
# matvec() output shapes must match b, got [(5,), (5,)] and [(5,)]

# primals – The primal values at which the Jacobian of fun should be evaluated. Should be
#  either a tuple or a list of arguments, and its length should equal to the number of
#  positional parameters of fun.

# tangents – The tangent vector for which the Jacobian-vector product should be evaluated.
#  Should be either a tuple or a list of tangents, with the same tree structure
#  and array shapes as primals.

# Return type Tuple[Any, Any]

# Returns A (primals_out, tangents_out) pair, where primals_out is fun(*primals), 
# and tangents_out is the Jacobian-vector product of function evaluated at 
# primals with tangents. The tangents_out value has the same Python tree structure and shapes as primals_out.



# full jacobian approach
while(count < maxit and  res > tol):
    start1 =timeit.default_timer() 
    J =  jac_fn(U, Uold)
    y = fn(U, Uold)
    res = norm(y/norm(U,np.inf),np.inf)
    delta = solve(J, y)
    U = U - delta
    count = count + 1
    end1 =timeit.default_timer() 
    print(count, res)

# vector product approach
while(count < maxit and  res > tol):
    start1 =timeit.default_timer() 
    J =  jac_fn(U, Uold)
    y = fn(U,Uold)
    res = norm(y/norm(U,np.inf),np.inf)
    jac_x_prod = lambda x: jvp(fn, y, x)
    delta = jax.scipy.sparse.linalg.cg(jax_x_prod, y)[0]
    U = U - delta
    count = count + 1
    end1 =timeit.default_timer() 
    print(count, res)



# Something you could consider doing is using the JAX conjugate gradients solver.

# The idea is that you can solve a system Ax = b without needing to instantiate A, you just need a function that gives x computes Ax. You want to solve the system J @ delta = y for Newton's method.
# The function

# jac_x_prod = lambda x: jvp(fn, y, x)
# is exactly this function that compute J @ x given x, where J is the Jacobian of fn evaluated at y.
# I think you might be able to use linearize here too, but I'm not too familiar with that.

# So in summary, if you replace delta = solve(J,y) with

# jac_x_prod = lambda x: jvp(fn, y, x)
# delta = jax.scipy.sparse.linalg.cg(jax_x_prod, y)[0]
# This should drastically cut down your memory usage.
# As for whether it would be in general faster, I'm not too sure.        

def newton(fn, jac_fn, U):
    maxit=20
    tol = 1e-8
    count = 0
    res = 100
    fail = 0
    Uold = U
 
    start =timeit.default_timer()     
    J =  jac_fn(U, Uold)
    y = fn(U,Uold)
    res0 = norm(y/norm(U,np.inf),np.inf)
    delta = solve(J,y)
    U = U - delta
    count = count + 1
    end = timeit.default_timer()
    print("time elapsed in first loop", end-start)
    print(count, res0)
    while(count < maxit and  res > tol):
        start1 =timeit.default_timer() 
        J =  jac_fn(U, Uold)
        y = fn(U,Uold)
        res = norm(y/norm(U,np.inf),np.inf)
        delta = solve(J,y)
        U = U - delta
        count = count + 1
        end1 =timeit.default_timer() 
        print(count, res)
        print("time per loop", end1-start1)
        
    if fail ==0 and np.any(np.isnan(delta)):
        fail = 1
        print("nan solution")
        
    if fail == 0 and max(abs(np.imag(delta))) > 0:
            fail = 1
            print("solution complex")
    
    if fail == 0 and res > tol:
        fail = 1;
        print('Newton fail: no convergence')
    else:
        fail == 0 
        
    return U, fail


# Do one newton step        
jac_fn = jax.jit(jacfwd(fn))
fn = jax.jit(fn)

# djb full thing
[sol, fail] = newton(fn, jac_fn, U)
