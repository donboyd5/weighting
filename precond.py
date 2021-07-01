
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
# (for factorization), and lu_solve (for solution) in the module scipy.linalg.
# For sparse matrices we have splu, and spilu, in the module scipy.sparse.linalg.


# https://link.springer.com/chapter/10.1007/978-3-030-50417-5_8

# %% imports
import importlib
import numpy as np
import numpy.random as npr
from numpy.random import seed
from numpy.linalg import norm

import scipy
from scipy import sparse
from scipy.sparse import linalg
import scipy.linalg as spla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla

from scipy.optimize import lsq_linear, root
from scipy.sparse import spdiags, kron
from scipy.sparse.linalg import spilu, LinearOperator
from scipy.sparse.linalg import aslinearoperator

from numpy import cosh, zeros_like, mgrid, zeros, eye

import jax
from jax import jvp, vjp, grad, hessian
import jax.numpy as jnp
# this next line is CRUCIAL or we will lose precision
from jax.config import config
config.update('jax_enable_x64', True)

import pylops
from pylops import FirstDerivative

import timeit
from timeit import default_timer as timer

import matplotlib.pyplot as plt

import src.make_test_problems as mtp
import src.microweight as mw

from collections import namedtuple, OrderedDict

import src.utilities as ut
import src.functions_geoweight_poisson as fgp


# %% reimports
importlib.reload(fgp)


# %% working example - DO NOT CHANGE
# solve for x in Ax = b, using preconditioner
A = np.array([[ 0.4445,  0.4444, -0.2222],
              [ 0.4444,  0.4445, -0.2222],
              [-0.2222, -0.2222,  0.1112]])
sA = sparse.csc_matrix(A)
sA.shape

b = np.array([[ 0.6667],
              [ 0.6667],
              [-0.3332]])

sA_iLU = sparse.linalg.spilu(sA)
sA_iLU.shape
sA_iLU.solve(b)

M = sparse.linalg.LinearOperator((3,3), sA_iLU.solve)
sparse.linalg.gmres(A, b, M=M)
sparse.linalg.gmres(A, b)

x = sparse.linalg.gmres(A, b, M=M)
# x = sparse

sparse.linalg.lgmres(A,b,M=M)
sparse.linalg.lgmres(A,b)

%timeit sparse.linalg.gmres(A, b, M=M)
%timeit sparse.linalg.gmres(A, b)

# %% variants
# use A and M from above


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



# parameters
nx, ny = 75, 75
hx, hy = 1./(nx-1), 1./(ny-1)

P_left, P_right = 0, 0
P_top, P_bottom = 1, 0

def get_preconditioner():
    """Compute the preconditioner M"""
    diags_x = zeros((3, nx))
    diags_x[0,:] = 1/hx/hx
    diags_x[1,:] = -2/hx/hx
    diags_x[2,:] = 1/hx/hx
    Lx = spdiags(diags_x, [-1,0,1], nx, nx)

    diags_y = zeros((3, ny))
    diags_y[0,:] = 1/hy/hy
    diags_y[1,:] = -2/hy/hy
    diags_y[2,:] = 1/hy/hy
    Ly = spdiags(diags_y, [-1,0,1], ny, ny)

    J1 = kron(Lx, eye(ny)) + kron(eye(nx), Ly)

    # Now we have the matrix `J_1`. We need to find its inverse `M` --
    # however, since an approximate inverse is enough, we can use
    # the *incomplete LU* decomposition

    J1_ilu = spilu(J1)

    # This returns an object with a method .solve() that evaluates
    # the corresponding matrix-vector product. We need to wrap it into
    # a LinearOperator before it can be passed to the Krylov methods:

    M = LinearOperator(shape=(nx*ny, nx*ny), matvec=J1_ilu.solve)
    return M


# %% solve
def solve(preconditioning=True):
    """Compute the solution"""
    count = [0]

    def residual(P):
        count[0] += 1

        d2x = zeros_like(P)
        d2y = zeros_like(P)

        d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2])/hx/hx
        d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
        d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

        d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
        d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
        d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

        return d2x + d2y + 5*cosh(P).mean()**2

    # preconditioner
    if preconditioning:
        M = get_preconditioner()
    else:
        M = None

    # solve
    guess = zeros((nx, ny), float)

    sol = root(residual, guess, method='krylov',
               options={'disp': True,
                        'jac_options': {'inner_M': M}})
    print('Residual', abs(residual(sol.x)).max())
    print('Evaluations', count[0])

    return sol.x


# %% test
solve(preconditioning=True)
solve(preconditioning=False)




def main():
    sol = solve(preconditioning=True)

    # visualize
    import matplotlib.pyplot as plt
    x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
    plt.clf()
    plt.pcolor(x, y, sol)
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()

# %% precond
nx, ny = 75, 75
hx, hy = 1./(nx-1), 1./(ny-1)

P_left, P_right = 0, 0
P_top, P_bottom = 1, 0

"""Compute the preconditioner M"""
diags_x = zeros((3, nx))
diags_x[0,:] = 1/hx/hx
diags_x[1,:] = -2/hx/hx
diags_x[2,:] = 1/hx/hx
Lx = spdiags(diags_x, [-1,0,1], nx, nx)  # nx by nx ie 75 x 85
Lx.todense()


diags_y = zeros((3, ny))
diags_y[0,:] = 1/hy/hy
diags_y[1,:] = -2/hy/hy
diags_y[2,:] = 1/hy/hy
Ly = spdiags(diags_y, [-1,0,1], ny, ny)
Ly.todense()

J1 = kron(Lx, eye(ny)) + kron(eye(nx), Ly)  # 5625 x 5625 + same, csr
J1.todense()

# Now we have the matrix `J_1`. We need to find its inverse `M` --
# however, since an approximate inverse is enough, we can use
# the *incomplete LU* decomposition

J1_ilu = spilu(J1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html
# Compute an incomplete LU decomposition for a sparse, square matrix.
# The resulting object is an approximation to the inverse of A.
# Returns invA_approxscipy.sparse.linalg.SuperLU
# object, which has a solve method.
J1_ilu.solve
J1_ilu.solve() # needs argument rhs

# This returns an object with a method .solve() that evaluates
# the corresponding matrix-vector product. We need to wrap it into
# a LinearOperator before it can be passed to the Krylov methods:

M = LinearOperator(shape=(nx*ny, nx*ny), matvec=J1_ilu.solve)

# %% try a test problem



# %% make problem
p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=0)
p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=10, k=8, xsd=.1, ssd=.5, pctzero=.2)
p = mtp.Problem(h=20000, s=20, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=30000, s=30, k=20, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=35000, s=40, k=25, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=50000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.2)

wh = p.wh
xmat = p.xmat
geotargets = p.geotargets
dw = fgp.jax_get_diff_weights(geotargets)  # (s, k)

# %% solve problem
bvec = np.zeros(geotargets.size)  # (s x k, )

diffs = fgp.jax_targets_diff_copy(bvec, wh, xmat, geotargets, dw) # (s x k, )

jacfn = jax.jacfwd(fgp.jax_targets_diff)
jacmat = jacfn(bvec, wh, xmat, geotargets, dw)
jacmat = np.array(jacmat).reshape((bvec.size, bvec.size))
jacmat  # (s x k, s x k)

np.diag(jacmat)

# get regular step
jinv = scipy.linalg.pinv(jacmat) # (s x k, s x k)
step = jnp.dot(jinv, diffs)
step # (s x k, )

# now try using the M preconditioner
# get M as diagonal of jacobian
invMjd = np.diag(jacmat)
Mjd = np.diag(1 / invMjd)
Mjd
# get M as gradient of sspd
fgrad = jax.grad(fgp.jax_sspd)
g = fgrad(bvec, wh, xmat, geotargets, dw)
Mg = np.diag(1 / g)

Mg2 = np.diag(1 / np.sqrt(np.abs(g)))
Mg3 = np.diag(1 / (np.sqrt(np.abs(g)) * np.sign(g)))

# get M as spilu
jilu = spilu(jacmat)
Mjsp = np.diag(jilu.solve(diffs))

# now get and compare steps
step_noM, info1 = scipy.sparse.linalg.lgmres(jacmat, diffs)
step_Mjd, info2 = scipy.sparse.linalg.lgmres(jacmat, diffs, M = Mjd)
step_Mg, info3 = scipy.sparse.linalg.lgmres(jacmat, diffs, M = Mg)
step_Mg2, info4 = scipy.sparse.linalg.lgmres(jacmat, diffs, M = Mg2)
step_Mg3, info4 = scipy.sparse.linalg.lgmres(jacmat, diffs, M = Mg3)
step_Mjsp, info4 = scipy.sparse.linalg.lgmres(jacmat, diffs, M = Mjsp)
step
step_noM
step_Mjd
step_Mg
step_Mg2
step_Mg3
step_Mjsp

# which step is closest to true step?
norm(step - step_noM, 2)
norm(step - step_Mjd, 2)
norm(step - step_Mg, 2)
norm(step - step_Mg2, 2)  # not as close as noM!
norm(step - step_Mg3, 2) # even worse!
norm(step - step_Mjsp, 2) # not as close as noM

# what if we don't have jacmat but only have a jvp??
l_diffs = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
l_jvp = lambda diffs: jax.jvp(l_diffs, (bvec,), (diffs,))[1]
l_vjp = lambda diffs: jax.vjp(l_diffs, bvec)[1](diffs)
Jsolver = scipy.sparse.linalg.LinearOperator((bvec.size, bvec.size), matvec=l_jvp, rmatvec=l_vjp)
# Jsolver = linop
step_lin, info_lin = scipy.sparse.linalg.lgmres(Jsolver, diffs)
step_lin  # just like step_noM
step_linMjd, info_linMjd = scipy.sparse.linalg.lgmres(Jsolver, diffs, M=Mjd)
step_linMjd  # great!
# now make M 0a sparse matrix
spMjd = scipy.sparse.diags(1 / invMjd)
step_linMjd2, info_linMjd2 = scipy.sparse.linalg.lgmres(Jsolver, diffs, M=spMjd)
step_linMjd2  # great!

diag_grad(l_diffs, bvec)

l_diffs(bvec)
jax.jacfwd(l_diffs)(bvec)
np.diag(jax.jacfwd(l_diffs)(bvec)) # good
invMjd

# check on step
bvec2 = bvec - step
diffs2 = fgp.jax_targets_diff_copy(bvec2, wh, xmat, geotargets, dw) # (s x k, )
diffs2
diffs

step, info = scipy.sparse.linalg.lgmres(jacmat, diffs, M = M, maxiter=options.lgmres_maxiter)


# %% diagonal of jacobian
# inspired by https://github.com/google/jax/issues/1563
f = lambda x: x**jax.numpy.arange(len(x))

def diag_grad(f, x):
    def partial_grad_f_index(i):
        def partial_grad_f_x(xi):
            return f(jax.ops.index_update(x, i, xi))[i]
        return jax.grad(partial_grad_f_x)(x[i])
    return jax.vmap(partial_grad_f_index)(jax.numpy.arange(x.shape[0]))

diag_grad(f, jax.device_put(jax.numpy.arange(4, dtype='float')))
jax.jacfwd(f)(jax.numpy.arange(4, dtype='float'))
# f(np.array([0, 1, 2, 3]))

# THIS WORKS!
f = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
diag_grad(f, jax.device_put(jax.numpy.array(bvec)))
jnp.diag(jax.jacfwd(f)(bvec))

%timeit diag_grad(f, jax.device_put(jax.numpy.array(bvec)))
%timeit jnp.diag(jax.jacfwd(f)(bvec))

# to reduce memory usage try
# jax.lax.map(f, xs) instead of vmap...but performance may suffer

# low mem
def diag_grad(f, x):
    def partial_grad_f_index(i):
        def partial_grad_f_x(xi):
            return f(jax.ops.index_update(x, i, xi))[i]
        return jax.grad(partial_grad_f_x)(x[i])
    return jax.lax.map(partial_grad_f_index)(jax.numpy.arange(x.shape[0]))




def f2(a, b):
    return a**2 + b
f2(3, 5)
gf2 = jax.grad(f2)
gf2(7., 5.)

jax.grad()

def diag_grad(f, x):
    def partial_grad_f_index(i):  # loop over indexes for x
        def partial_grad_f_x(xi):  # loop over each x
            return f(jax.ops.index_update(x, i, xi))[i]
        return jax.grad(partial_grad_f_x)(x[i]) # get Jac diagonal at this i
    return jax.vmap(partial_grad_f_index)(jax.numpy.arange(x.shape[0]))
x = jax.numpy.arange(4, dtype='float')
f(x)
i = 1
xi = x[i]
x
xi
def pgfi(i):
    def pgfx(xi):
        return f(jax.ops.index_update(x, i, xi))[i]
    return jax.grad(pgfx)(x[i]) # get Jac diagonal at this i
pgfi(3)
jax.vmap(pgfi)(jax.numpy.arange(x.shape[0]))

# Good, but for vmap:
def pgfi2(ibv):
    def pgfx2(bveci, wh, xmat, geotargets, dw):
        return fgp.jax_targets_diff(jax.ops.index_update(bvec, ibv, bveci), wh, xmat, geotargets, dw)[ibv]
    return jax.grad(pgfx2)(bvec[ibv], wh, xmat, geotargets, dw) # get Jac diagonal at this ibv
jax.vmap(pgfi2)(jax.numpy.arange(bvec.shape[0]))

jax.vmap(pgfi2, in_axes=(0))(jax.numpy.arange(bvec.shape[0]))

jax.vmap(pgfi2)(jax.numpy.arange(6))

jax.vmap(pgfi2)(np.array([0, 1, 2, 3, 4, 5]))
jax.vmap(pgfi2)(jnp.array([0, 1, 2, 3, 4, 5]))

jax.vmap(pgfi2)(jnp.arange(1))

def pgfi2(ibv):
    def pgfx2(bveci, wh, xmat, geotargets, dw):
        return fgp.jax_targets_diff(jax.ops.index_update(bvec, ibv, bveci), wh, xmat, geotargets, dw)[ibv]
    g = jax.grad(pgfx2)(bvec[ibv], wh, xmat, geotargets, dw)
    return g # get Jac diagonal at this ibv
pgfi2(5)
jax.vmap(pgfi2)(jax.numpy.arange(bvec.shape[0]))


pgfi2(0)

l_pgfi2 = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)

def dgrad(f, bvec, wh, xmat, geotargets, dw):
    def pgfi2(ibv):
        def pgfx2(bveci, wh, xmat, geotargets, dw):
            return f(jax.ops.index_update(bvec, ibv, bveci), wh, xmat, geotargets, dw)[ibv]
        return jax.grad(pgfx2)(bvec[ibv], wh, xmat, geotargets, dw) # get Jac diagonal at this ibv
    return jax.vmap(pgfi2)(jax.numpy.arange(bvec.shape[0]))
dgrad(fgp.jax_targets_diff, bvec, wh, xmat, geotargets, dw)

jax.vmap(pgfi2)(jax.numpy.arange(bvec.shape[0]))

def dgrad(f, bvec, wh, xmat, geotargets, dw):
    def pgfi2(ibv, bvec, wh, xmat, geotargets, dw):
        def pgfx2(bveci, bvec, wh, xmat, geotargets, dw):
            return f(jax.ops.index_update(bvec, ibv, bveci), wh, xmat, geotargets, dw)[ibv]
        return jax.grad(pgfx2)(bvec[ibv], wh, xmat, geotargets, dw) # get Jac diagonal at this ibv
    return jax.vmap(pgfi2, in_axes=(0, None, None, None, None, None))(jax.numpy.arange(bvec.shape[0]), bvec, wh, xmat, geotargets, dw)
dgrad(fgp.jax_targets_diff, bvec, wh, xmat, geotargets, dw)

pgfi2(2)

f(jax.ops.index_update(x, i, xi))[i]  # f value at i
jax.ops.index_update(x, i, xi)
def pgfx(x, xi, i):
    return f(jax.ops.index_update(x, i, xi))[i]

i = 3
jax.grad(pgfx)(x[i]) # returns J diagonal element
def pgfi(x, i):
    return jax.grad(pgfx)(x, x[i], i)  # the J diag element

pgfi(x, 0)
pgfi(x, 1)
pgfi(x, 3)
jax.vmap(pgfi)(jax.numpy.arange(x.shape[0]))

def pgfxi(i):
    return

x2 = np.array(x)
idx = i; y = xi
x2[idx] = y
x2
f(x2)

# 0: full array
# 1: full array
# 2: full array

jax.jacfwd(f)(jax.numpy.arange(4, dtype='float'))

def diag_grad2(f, bvec, wh, xmat, geotargets, dw):
    def partial_grad_f_index2(i):
        def partial_grad_f_x2(bveci, wh, xmat, geotargets, dw):
            return f(jax.ops.index_update(bvec, i, bveci), wh, xmat, geotargets, dw)[i]
        return jax.grad(partial_grad_f_x2)(bvec[i], wh, xmat, geotargets, dw)
    return jax.vmap(partial_grad_f_index2)(jax.numpy.arange(bvec.shape[0]))
diag_grad2(f, jax.device_put(jax.numpy.arange(4, dtype='float')))

diag_grad2(fgp.jax_targets_diff, bvec, wh, xmat, geotargets, dw)


def our_jacfwd(f):
    def jacfun(x):
        _jvp = lambda s: jvp(f, (x,), (s,))[1]
        Jt =vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun


# %% vmap play
def what(a,b,c):
  z = jnp.dot(a,b)
  return z + c

v_what = jax.vmap(what, in_axes=(None,0,None))

a = jnp.array([1,1,3])
b = jnp.array([2,2])
c = 1.0

v_what(a,b,c)
a
b
c


jax.vmap(what, in_axes=(None,0,None))(a, b, c)

jax.vmap(what, in_axes=(0,None,None))(a, b, c)
jnp.dot(a[3], b) + c

dbdiffs = fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)

def djb(i):
    return fgp.jax_targets_diff(jax.ops.index_update(bvec, i, bvec[i]), wh, xmat, geotargets, dw)[i]

djb(0)
djb(5)
jax.vmap(djb)(jax.numpy.arange(bvec.shape[0]))

jax.vmap(djb)(jax.device_put(jax.numpy.arange(6, dtype='float')))


# Good, but for vmap:
def pgfi2(ibv):
    def pgfx2(bveci, wh, xmat, geotargets, dw):
        return fgp.jax_targets_diff(jax.ops.index_update(bvec, ibv, bveci), wh, xmat, geotargets, dw)[ibv]
    return jax.grad(pgfx2)(bvec[ibv], wh, xmat, geotargets, dw) # get Jac diagonal at this ibv
pgfi2(0)
jax.vmap(pgfi2)

jax.vmap(pgfi2)(jax.numpy.arange(bvec.shape[0]))



f = lambda x: x**jax.numpy.arange(len(x))
x = jnp.array([0, 3, 5, 7])
f(x)

def diag_grad(f, x):
    def partial_grad_f_index(i):
        def partial_grad_f_x(xi):
            return f(jax.ops.index_update(x, i, xi))[i]
        return jax.grad(partial_grad_f_x)(x[i])
    return jax.vmap(partial_grad_f_index)(jax.numpy.arange(x.shape[0]))

diag_grad(f, jax.device_put(jax.numpy.arange(4, dtype='float')))

f = lambda x: x**jax.numpy.arange(len(x))



def partial_grad_f_index(i):
    def partial_grad_f_x(xi):
        return f(jax.ops.index_update(x, i, xi))[i]
    return jax.grad(partial_grad_f_x)(x[i])

x = jax.device_put(jax.numpy.arange(4, dtype='float')**2)
x
jax.vmap(partial_grad_f_index)(jax.numpy.arange(x.shape[0]))
jax.jacfwd(f)(jax.numpy.arange(4, dtype='float'))
jax.jacfwd(f)(x)

x = jax.device_put(jax.numpy.array(bvec))

f = lambda x: fgp.jax_targets_diff(x, wh, xmat, geotargets, dw)
f = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
jax.vmap(partial_grad_f_index)(jax.numpy.arange(x.shape[0]))
diag_grad(f, bvec)

# THIS WORKS!
f = lambda bvec: fgp.jax_targets_diff(bvec, wh, xmat, geotargets, dw)
diag_grad(f, jax.device_put(jax.numpy.array(bvec)))

# https://stackoverflow.com/questions/61786831/vmap-over-a-list-in-jax
# https://jiayiwu.me/blog/2021/04/05/learning-about-jax-axes-in-vmap.html
# https://stackoverflow.com/questions/66548897/jax-vmap-behaviour
# https://github.com/google/jax/issues/1563