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
# here's what i think
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