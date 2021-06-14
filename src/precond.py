
# https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781783984749/1/ch01lvl1sec12/matrix-factorizations-related-to-solving-matrix-equations
# For any other [not PSD] kind of generic square matrix A, the next best method
# to solve the basic system A ● x = b is pivoted LU factorization. This is
# equivalent to finding a permutation matrix P, and triangular matrices U
# (upper) and L (lower) so that P ● A = L ● U. In such a case, a permutation
# of the rows in the system according to P gives the equivalent equation
# (P ● A) ● x = P ● b. Set c = P ● b and y = U ● x, and solve for y in
# the system L ● y = c using forward substitution. Then, solve for x in
# the system U ● x = y with back substitution.

# The relevant functions to perform this operation are lu, lu_factor
# (for factorization), and lu_solve (for solution) in the module scipy.linalg. For sparse matrices we have splu, and spilu, in the module scipy.sparse.linalg.


# https://link.springer.com/chapter/10.1007/978-3-030-50417-5_8


import numpy as np
from scipy import sparse
from scipy.sparse import linalg

import scipy.linalg as spla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla

import scipy.sparse.linalg as spla

import pylops

# %% working example - DO NOT CHANGE
A = np.array([[ 0.4445,  0.4444, -0.2222],
              [ 0.4444,  0.4445, -0.2222],
              [-0.2222, -0.2222,  0.1112]])
sA = sparse.csc_matrix(A)

b = np.array([[ 0.6667],
              [ 0.6667],
              [-0.3332]])

sA_iLU = sparse.linalg.spilu(sA)
M = sparse.linalg.LinearOperator((3,3), sA_iLU.solve)


x = sparse.linalg.gmres(A,b,M=M)
# x = sparse
sparse.linalg.gmres(A,b)

sparse.linalg.lgmres(A,b,M=M)
sparse.linalg.lgmres(A,b)

%timeit sparse.linalg.gmres(A,b,M=M)
%timeit sparse.linalg.gmres(A,b)

# %% variants
# use A and M from above
from scipy.sparse.linalg import aslinearoperator

Alo = aslinearoperator(A)
Mlo = aslinearoperator(M)
sparse.linalg.gmres(A, b)
sparse.linalg.gmres(Alo, b)
sparse.linalg.gmres(A, b, M=M)
sparse.linalg.gmres(Alo, b, M=Mlo)

sparse.linalg.lgmres(A, b)
sparse.linalg.lgmres(Alo, b)
sparse.linalg.lgmres(A, b, M=M)
sparse.linalg.lgmres(Alo, b, M=Mlo)

%timeit sparse.linalg.gmres(A, b)
%timeit sparse.linalg.gmres(Alo, b)
%timeit sparse.linalg.gmres(A, b, M=M)
%timeit sparse.linalg.gmres(Alo, b, M=Mlo)

%timeit sparse.linalg.lgmres(A, b)
%timeit sparse.linalg.lgmres(Alo, b)
%timeit sparse.linalg.lgmres(A, b, M=M)
%timeit sparse.linalg.lgmres(Alo, b, M=Mlo)

sol, info = sparse.linalg.gmres(A, b)
sol, info = sparse.linalg.lgmres(Alo, b, M=Mlo)
Mlo(sol)  # returns a vector of multiliers

np.fill_diagonal(A,np.diag(A))
A
np.diagflat(np.diag(A))


pylops.Diagonal(A)
pylops.Diagonal(Alo(sol))

Alo(b)
Alo(sol)
b
np.dot(A, sol)

Alo(np.array([[1, 1, 1], [1, 1, 1], [1,1,1]]))

# these 3 together get us the diagonal of a
Alo(np.array([1, 0, 0]))
Alo(np.array([0, 1, 0]))
Alo(np.array([0, 1, 1]))

for i in range(0, 3):
    Alo(np.array[])

pylops.Diagonal(sol)
pylops.Diagonal(b)
pylops.matrix(Alo(sol))

x = np.array([1, 1, 1])
Alo(x)

np.apply_along_axis(abs_sum, 1, A)



# %% now create full working version
A = np.array([[ 0.4445,  0.4444, -0.2222],
              [ 0.4444,  0.4445, -0.2222],
              [-0.2222, -0.2222,  0.1112]])
N = A.shape[0]

b = np.array([[ 0.6667],
              [ 0.6667],
              [-0.3332]])



# create A as a linear operator, which is same concept as our jvp LO
Alo = aslinearoperator(A)  # equivalent to jvp
sparse.linalg.gmres(Alo, b)  # this is what I am doing now

lu, piv = spla.lu_factor(A)
spla.lu_factor(Alo) # does not work on linear operator

import jax
jax.scipy.linalg.lu(A)  # good
jax.scipy.linalg.lu(Alo)  # does not work

sparse.linalg.spilu(A)  # gives efficiency warning
sparse.linalg.spilu(Alo) # does not work on linear operator

sparse.linalg.splu(sA)
sparse.linalg.spilu(sA)
sparse.linalg.spilu(A)
M = sparse.linalg.LinearOperator((N,N), sA_iLU.solve)

np.dot(A, b)
sol, info = sparse.linalg.gmres(A, b)
def vp(A, v):
    return np.dot(A, v)
vp(A, sol)
lvp = lambda x: vp(A, x)
lvp(sol)

spla.lu(A)
jax.scipy.linalg.lu(A)
jax.scipy.linalg.lu(lvp)  # does not work with matrix vector product




    # l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    # l_diffs = jax.jit(l_diffs)
    # l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]

# %% more play
import pylops
from pylops import FirstDerivative
import timeit

nx = 7
x = range(-(nx // 2), nx // 2 + (1 if nx % 2 else 0))
x

Dlop = FirstDerivative(nx, dtype='float64')

# y = Dx
y = Dlop*x
# x = D'y
xadj = Dlop.H*y
# xinv = D^-1 y
xinv = Dlop / y

import timeit
import matplotlib.pyplot as plt
import numpy as np
import pylops

# setup command
cmd_setup ="""\
import numpy as np
import pylops
n = 10
d = np.arange(n) + 1.
x = np.ones(n)
Dop = pylops.Diagonal(d)
DopH = Dop.H
"""

# _matvec
cmd1 = 'Dop._matvec(x)'

# matvec
cmd2 = 'Dop.matvec(x)'

# @
cmd3 = 'Dop@x'

# *
cmd4 = 'Dop*x'

# timing
t1 = 1.e3 * np.array(timeit.repeat(cmd1, setup=cmd_setup,
                                   number=500, repeat=5))
t2 = 1.e3 * np.array(timeit.repeat(cmd2, setup=cmd_setup,
                                   number=500, repeat=5))
t3 = 1.e3 * np.array(timeit.repeat(cmd3, setup=cmd_setup,
                                   number=500, repeat=5))
t4 = 1.e3 * np.array(timeit.repeat(cmd4, setup=cmd_setup,
                                   number=500, repeat=5))

plt.figure(figsize=(7, 2))
plt.plot(t1, 'k', label=' _matvec')
plt.plot(t2, 'r', label='matvec')
plt.plot(t3, 'g', label='@')
plt.plot(t4, 'b', label='*')
plt.axis('tight')
plt.legend();


n = 10
d = np.arange(n) + 1.
d = np.array([1, 2, 3, 4, 3, 2, 1, 0, 0, -1])
d  # 1...10
x = np.ones(n)
x = np.array([2, 4, 6, 8, 10, 1, 3, 5, 7, 9])
Dop = pylops.Diagonal(d)
# I think this is like making a matrix with diatonal d
np.diag(d)
Dop._matvec(x)
# same result
np.dot(np.diag(d), x)

Dop.todense()

# https://pylops.readthedocs.io/en/latest/tutorials/dottest.html
# https://pylops.readthedocs.io/en/latest/tutorials/solvers.html
# https://pylops.readthedocs.io/en/latest/api/index.html#solvers
# https://github.com/PyLops/pylops_notebooks
# https://pylops.readthedocs.io/en/latest/tutorials/solvers.html#sphx-glr-tutorials-solvers-py
# https://pylops.readthedocs.io/en/latest/tutorials/solvers.html#sphx-glr-tutorials-solvers-py

# READ THIS:
# https://ask.csdn.net/questions/6196644






# %% play
# We use a large circulant matrix (non-symmetric) for this example:
z = 8
D = spla.circulant(np.arange(z))  # 4096
D
%timeit spla.lu(D)
%timeit spla.lu_factor(D)

P, L, U = spla.lu(D)
P.shape
L.shape
U.shape

# The outputs of the function lu_factor are resource-efficient. We obtain a
# matrix LU, with upper triangle U and lower triangle L. We also obtain a
# one-dimensional ndarray class of integer dtype, piv, indicating the pivot
# indices representing the permutation matrix P.

LU, piv = spla.lu_factor(D)

# The solver lu_solve takes the two outputs from lu_factor, a right-hand side
#  matrix b, and the optional indicator trans to the kind of basic system to solve:
spla.lu_solve(spla.lu_factor(D), np.ones(z))
sparse.linalg.lgmres(D,np.ones(z))

# The main difference between splu and spilu is that the latter computes an
# incomplete decomposition. With it, we can obtain really good approximations
# to the inverse of matrix A, and use matrix multiplication to compute the
# solution of large systems in a fraction of the time that it would take to
# calculate the actual solution.

# Let us illustrate this with a simple example, where the permutation of rows
# or columns is not needed. In a large lower triangular Pascal matrix, turn
# into zero all the even-valued entries and into ones all the odd-valued entries.
#  Use this as matrix A. For the right-hand side, use a vector of ones:

z = 1024  # 1024

A = (spla.pascal(z, kind='lower')%2 != 0)
A = spla.circulant(np.arange(z))
A.shape
A[0:9, 0:9]
A_csc = spsp.csc_matrix(A, dtype=np.float64)

invA = spspla.splu(A_csc)
%time invA.solve(np.ones(z))

invA = spspla.spilu(A_csc)
%time invA.solve(np.ones(z))


x, info = spspla.cg(A_csc, np.ones(z), x0=np.zeros(z))  # does not solve?
x
info 10, 240

Out[25]: (array([ nan,  nan,  nan, ...,  nan,  nan,  nan]), 1)
%time spspla.gmres(A_csc, np.ones(z), x0=np.zeros(z))

def callbackF(xk):
    global Nsteps
    print('{0:4d}  {1:3.6f}  {2:3.6f}'.format(Nsteps, xk[0],xk[1]))
    Nsteps += 1

Nsteps = 1
print('{0:4s}  {1:9s}  {1:9s}'.format('Iter', 'X[0]','X[1]'))
spspla.bicg(A_csc, np.ones(z), x0=np.zeros(z), callback=callbackF)
spspla.gmres(A_csc, np.ones(z), x0=np.zeros(z), callback=callbackF)
spspla.lgmres(A_csc, np.ones(z), x0=np.zeros(z), callback=callbackF)

x, info = spspla.bicg(A_csc, np.ones(z), x0=np.zeros(z))
x, info = spspla.gmres(A_csc, np.ones(z), x0=np.zeros(z))
x, info = spspla.lgmres(A_csc, np.ones(z), x0=np.zeros(z))
x
info


%timeit spspla.bicg(A_csc, np.ones(z), x0=np.zeros(z))
%timeit spspla.gmres(A_csc, np.ones(z), x0=np.zeros(z))
%timeit spspla.lgmres(A_csc, np.ones(z), x0=np.zeros(z))


# %% diagonals
from jax import jvp, grad, hessian
import jax.numpy as jnp
import numpy.random as npr

rng = npr.RandomState(0)
a = rng.randn(4)
x = rng.randn(4)

# function with diagonal Hessian that isn't rank-polymorphic
def f(x):
  assert x.ndim == 1
  return jnp.sum(jnp.tanh(a * x))


def hvp(f, x, v):
  return jvp(grad(f), (x,), (v,))[1]

x
f(x)
grad(f)(x)
hessian(f)(x)
jnp.diag(hessian(f)(x))
hvp(f, x, jnp.ones_like(x))
jvp(grad(f), (x,), (jnp.ones_like(x),))

