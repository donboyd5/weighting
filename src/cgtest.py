# test conjugate gradient

# %% urls
# syntax https://jax.readthedocs.io/en/latest/jax.html
# https://github.com/google/jax/issues/4753
# https://github.com/hanrach/p2d_solver/blob/main/run_ex.py


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

# %% scipy cg documentation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
# scipy.sparse.linalg.cg(A, b, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)
# Use Conjugate Gradient iteration to solve Ax = b.

# Parameters
# A{sparse matrix, dense matrix, LinearOperator}
# The real or complex N-by-N matrix of the linear system. A must represent a hermitian, positive definite matrix. Alternatively, A can be a linear operator which can produce Ax using, e.g., scipy.sparse.linalg.LinearOperator.

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
# Preconditioner for A. The preconditioner should approximate the inverse of A. Effective preconditioning dramatically improves the rate of convergence, which implies that fewer iterations are needed to reach a given error tolerance.

# callbackfunction
# User-supplied function to call after each iteration. It is called as callback(xk), where xk is the current solution vector.


# %% jax cg documentation

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




# %% imports

# %% test without parameters


# %% test with parameters




