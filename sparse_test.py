
# %% try to set a small problem up for ipopt that is in sparse form
# we need the jacobian structure
# right now that is np.nonzero(cc.T)

import numpy as np
import scipy.sparse as sps
import src.make_test_problems as mtp

p = mtp.Problem(h=20, s=1, k=3)

n = p.h  # n number variables
m = p.k  # m number constraints

xmat = p.xmat.copy()
wh = p.wh.copy()

# randomly set some elements of xmat to zero
pctzero = .2
indices = np.random.choice(np.arange(xmat.size), replace=False, size=int(xmat.size * pctzero))
xmat[np.unravel_index(indices, xmat.shape)] = 0 

p.xmat
xmat

#  constraint coefficients (constant)
cc = (xmat.T * wh).T

# how big could cc be?
# standard reweight:
# 40k variables, 40 targets, 1.6 million, maybe 80% nonzero

# state weights:
# 40k records
# 50 states
# 40 targets
# 2 million variables
# 2k constraints for states
# 40k adding up constraints
# 42k total constraints
# so 42k x 2m = 84e9 constraint coefficients
# but max nonzero constraint coefficients are:
# (a) for state-specific constraints:
#  2k constraints x 1 /50 x 2 m variables = 80e6
# (b) for adding-up constraints:
#  40k constraints x 50 states for each = 2e6
# for max nonzero constraint coefficients of 82e6
recs = 40e3; states = 50; targs = 40  # 82e6
recs = 20e3; states = 50; targs = 25  # 26e6
maxnzcc = (states * targs * recs + states * recs) / 1e6
maxnzcc

# for jacobian, we want the nonzero elements of the cc transpose
cc.T
np.nonzero(cc.T)
sps.find(cc.T)

%timeit np.nonzero(cc.T)  # 2.24us
%timeit sps.find(cc.T)  # 52.5us


# ijnz is a tuple with indexes and values of nonzero elements
# it has 3 elements, each of which is an array
ijnz = sps.find(cc.T)
ijnz[0] # row indexes of nz elements
ijnz[1] # col indexes of nz elements
ijnz[2] # nz elements

# %% manual setup for ipopt



# what we need for ipopt:





# define options (add_option)





# %% scratch
# https://cyipopt.readthedocs.io/en/stable/tutorial.html#tutorial
# https://cyipopt.readthedocs.io/en/stable/reference.html


np.nonzero(np.tril(np.ones((4, 4))))

x = [1, 2, 3, 4]
np.concatenate((np.prod(x)/x, 2*x))



# %% earlier play
# Construct a 1000x1000 lil_matrix and add some values to it:

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import find
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

A = np.matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
A


print(csc_matrix(A))
print(csr_matrix(A))
print(coo_matrix(A))
print(lil_matrix(A))

ijnz = find(A)
ijnz

type(ijnz)
dir(ijnz)

# ijnz is a tuple with indexes and values of nonzero elements
# it has 3 elements, each of which is an array
ijnz[0] # row indexes of nz elements
ijnz[1] # col indexes of nz elements
ijnz[2] # nz elements

# next, review what ipopt needs for the triplet format
ijnz

A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
v = np.array([1, 0, -1])
A.dot(v)
# array([ 1, -3, -1], dtype=int64)


rows = 1000
cols = 1000

A = lil_matrix((rows, cols))
A[0, :100] = rand(100)
A[1, 100:200] = A[0, :100]
A.setdiag(rand(rows))
A.shape
print(A[0:3, 0:15])



A = A.tocsr()
b = rand(1000)
x = spsolve(A, b)