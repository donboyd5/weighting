# %% imports
import numpy as np
import tensorflow as tf

from numpy.linalg import norm

import jax
import jax.numpy as jnp
import src.make_test_problems as mtp
import src.functions_geoweight_poisson as fgp

# this next line is CRUCIAL or we will lose precision
from jax.config import config
config.update('jax_enable_x64', True)

from timeit import default_timer as timer

# %% tensor play
# https://stackoverflow.com/questions/41870228/understanding-tensordot
# We input the arrays and the respective axes along which the sum-reductions are intended.
# The axes that take part in sum-reduction are removed in the output and all of the remaining
# axes from the input arrays are spread-out as different axes in the output keeping the order
# in which the input arrays are fed.

# matrix-multiplication involves elementwise multiplication keeping an axis aligned and then
# summation of elements along that common aligned axis. With that summation, we are losing
# that common axis, which is termed as reduction, so in short sum-reduction

# tensordot swaps axes and reshapes the inputs so it can apply np.dot to 2 2d arrays.
# It then swaps and reshapes back to the target. It may be easier to experiment than to explain.
# There's no special tensor math going on, just extending dot to work in higher dimensions.
# tensor just means arrays with more than 2d. If you are already comfortable with einsum then it will
# be simplest compare the results to that.

a = tf.constant([1,2.])
b = tf.constant([2,3.])
a
b
print(f"{tf.tensordot(a, b, 0)}\t tf.einsum('i,j', a, b)\t\t- ((the last 0 axes of a), (the first 0 axes of b))")
tf.tensordot(a, a, 0)
tf.einsum('i,j', a, a)

print(f"{tf.tensordot(a, b, ((),()))}\t tf.einsum('i,j', a, b)\t\t- ((() axis of a), (() axis of b))")


print(f"{tf.tensordot(b, a, 0)}\t tf.einsum('i,j->ji', a, b)\t- ((the last 0 axes of b), (the first 0 axes of a))")
print(f"{tf.tensordot(a, b, 1)}\t\t tf.einsum('i,i', a, b)\t\t- ((the last 1 axes of a), (the first 1 axes of b))")
print(f"{tf.tensordot(a, b, ((0,), (0,)))}\t\t tf.einsum('i,i', a, b)\t\t- ((0th axis of a), (0th axis of b))")
print(f"{tf.tensordot(a, b, (0,0))}\t\t tf.einsum('i,i', a, b)\t\t- ((0th axis of a), (0th axis of b))")


# %% 2 x tf
a = tf.constant([[1,2],
                 [-2,3.]])

b = tf.constant([[-2,3],
                 [0,4.]])
print(f"{tf.tensordot(a, b, 0)}\t tf.einsum('ij,kl', a, b)\t- ((the last 0 axes of a), (the first 0 axes of b))")
tf.tensordot(a, b, 0)
tf.einsum('ij,kl', a, b)


print(f"{tf.tensordot(a, b, (0,0))}\t tf.einsum('ij,ik', a, b)\t- ((0th axis of a), (0th axis of b))")
tf.tensordot(a, b, (0,0))


print(f"{tf.tensordot(a, b, (0,1))}\t tf.einsum('ij,ki', a, b)\t- ((0th axis of a), (1st axis of b))")
tf.tensordot(a, b, (0,1))

print(f"{tf.tensordot(a, b, 1)}\t tf.matmul(a, b)\t\t- ((the last 1 axes of a), (the first 1 axes of b))")
tf.tensordot(a, b, 1)


print(f"{tf.tensordot(a, b, ((1,), (0,)))}\t tf.einsum('ij,jk', a, b)\t- ((1st axis of a), (0th axis of b))")
tf.tensordot(a, b, ((1,), (0,)))

print(f"{tf.tensordot(a, b, (1, 0))}\t tf.matmul(a, b)\t\t- ((1st axis of a), (0th axis of b))")
tf.tensordot(a, b, (1, 0))


print(f"{tf.tensordot(a, b, 2)}\t tf.reduce_sum(tf.multiply(a, b))\t- ((the last 2 axes of a), (the first 2 axes of b))")
tf.tensordot(a, b, 2)

print(f"{tf.tensordot(a, b, ((0,1), (0,1)))}\t tf.einsum('ij,ij->', a, b)\t\t- ((0th axis of a, 1st axis of a), (0th axis of b, 1st axis of b))")
tf.tensordot(a, b, ((0,1), (0,1)))


# %% more


A = np.random.randint(2, size=(2, 6, 5))
B = np.random.randint(2, size=(3, 2, 4))
A
B
# if we tensordot using axis 0 for A and 1 for b:
np.tensordot(A, B, axes=((0),(1))).shape
# then the 2 axis for A drops out (we get 6, 5)
# and the 2 axis for B drops out (we get 3, 4)
# and the result has shape (6, 5, 3, 4)
np.tensordot(A, B, axes=((0),(1)))


A = np.array([1,2])
B = np.array([3,4])
np.tensordot(A, B, axes=((0), (0))) # ????

# %% einsum
x = np.array([1,2])
xxp = np.einsum('i,j->ij', x, x.T)
np.einsum('i,j->ji', x, x)
np.einsum('i,j->ij', x, x)
xxp

x = np.array([[1, 2],
              [4, 5],
              [7, 8]])
np.einsum('ij,jk->ijk', x, x.T)

# djb this is it
np.einsum('ij,ik->ijk', x, x)

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
np.einsum('ij,ik->ijk', x, x)


np.einsum('ij,kl->ijk', x, x.T)
np.einsum('ij,kl->ijl', x, x)
np.einsum('ij,kl->ijj', x, x.T) # bad
np.einsum('ij,jk->ikj', x, x) # bad


#%% create array of xxpx matrices (x * xprime) since this will be constant

# If we have:
#    two tensors a and b
#    two array-like objects which denote axes, a_axes and b_axes.
# The tensordot() function sums the product of a’s elements and b’s elements over the axes specified by a_axes and b_axes.



# tensor product

A = np.array([1,2])
B = np.array([3,4])
C = np.tensordot(A, B, axes=0)
C
np.tensordot(A, A.T, axes=0)  # this is AAprime
# np.tensordot(A, A.T, axes=((1), (1)))
# same as
Ar = A.reshape(A.size, 1)
Ar.dot(Ar.T)
# NOT same as
np.dot(A.T, A)

x = np.array([[1, 2],
                [4, 5],
                [7, 8]])
def f(x):
    return np.tensordot(x, x.T, axes=0)
np.apply_along_axis(func1d = f, axis=1, arr=x)  # this gets us xxprime for each row of x

np.tensordot(x, x.T, axes=(0, 0))


x = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
x = np.array([[1, 2],
                [4, 5],
                [7, 8]])
np.einsum('ij,jk', x, x)
x
x[0, ].T.dot(x[0, ])

def f(x):
    return x.dot(x.T)

def f(x):
    x = x.reshape(x.size, 1)
    return x.dot(x.T)

x = np.random.rand(40000, 30)

# %% timings
# this is a fast enough way to get the xxp matrices
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

def f(x):
    x = x.reshape(x.size, 1)
    return x.dot(x.T)


%timeit xxp1 = np.apply_along_axis(func1d = f, axis=1, arr=x)
%timeit xxp2 = np.einsum('ij,ik->ijk', x, x)  # much faster
np.allclose(xxp1, xxp2)

y = np.array([10, 20, 30, 40])
xxp2
np.einsum('i,ijk->ijk', y, xxp2) # good
np.einsum('i,ijk->ijk', y, xxp2).sum(axis=0) # good
np.einsum('i,ijk->jk', y, xxp2) # good  # same result!!!

%timeit np.einsum('i,ijk->ijk', y, xxp2).sum(axis=0) # good
%timeit np.einsum('i,ijk->jk', y, xxp2) # good  # same result!!! at least twice as fast


# what if y is matrix of weights
whs = np.array([
                [10, 100],
                [20, 200],
                [30, 300],
                [40, 400]
                ])
whs
np.einsum('ijk,il->iljk', xxp2, whs) # .sum(axis=0) # good
np.einsum('il,ijk->ljk', whs, xxp2)  # this gives us s Ds matrices, each of which we will want to invert
%timeit np.einsum('il,ijk->ljk', whs, xxp2) # good  # same result!!! at least twice as fast


D = np.einsum('il,ijk->ljk', whs, xxp2)
np.linalg.inv(D[0,:,:])
np.linalg.inv(D[1,:,:])
np.apply_along_axis(func1d = f, axis=1, arr=x)



# %% put it together with something larger

h = 100000
k = 60
s= 100


h = 40000
k = 30
s= 50

h = 1000
k = 4
s= 10

h = 30000
k = 20
s= 50

h = 40
k = 3
s= 4


np.random.seed(1)
x = np.random.rand(h, k)
whs = np.random.rand(h, s)
x
whs

t1 = timer()
xxp = np.einsum('ij,ik->ijk', x, x)
t2 = timer()
t2 - t1

t3 = timer()
D = np.einsum('il,ijk->ljk', whs, xxp)
t4 = timer()
t4 - t3

t5 = timer()
invD = np.linalg.inv(D)
t6 = timer()
t6 - t5

t2 - t1 + t4 - t3 + t6 - t5

whs.shape
xxp.shape
D.shape
invD.shape

diffs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

 ds = diffs[s, ].reshape(p.k, 1)
    # print(ds.shape)
    steprow = Dsinv.dot(ds).flatten()


# invert the D matrices
D[0,:,:].shape
np.linalg.cond(D[0,:,:])
np.linalg.inv(D[0,:,:])

def f(m):
    print(m.shape)
    return # np.linalg.inv(m)

D.shape
np.apply_along_axis(func1d = f, axis=0, arr=D)

invD = np.linalg.inv(D)
invD.shape

s = 10
np.linalg.inv(D[s,:,:])
invD[s,:,:]


# %% end putting together



Ds = np.einsum('i,ijk->ijk', y, xxp2).sum(axis=0) # good
Dsinv = np.linalg.inv(Ds)  # singular

if np.linalg.cond(Ds) < 1/sys.float_info.epsilon:
    Dsinv = np.linalg.inv(Ds)
else:
    print("Ds is singular, cannot be inverted")

# https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
# In numerical computing it is usually considered bad practice to explicitly calculate the inverse.
# In most cases it is much better to calculate the LU decomposition with scipy.linalg.lu_factor,
# then later you can solve it quickly for many vectors using scipy.linalg.lu_solve

# also, singular value decomposition https://en.wikipedia.org/wiki/Singular_value_decomposition


np.einsum('ij,k', x, x.T)


# %% functions
def geths(h, s):
    xh = xmat[h].reshape(xmat.shape[1], 1)
    xxph = xh.dot(xh.T)  # k x k -- the same for all s, for a given h
    # print(xxph)
    # print(whs[h, s])
    whsxpx = whs[h, s] * xxph  # initial values same for all s of an h
    # print(whs[h, s])
    return whsxpx

# (cm*wghts).sum()
k = 10
M1 = np.array([[1, 2],
                [3, 4]])
wts = np.array([2, 3])
wM1 = (M1 * wts).sum()
wM1
np.dot(M1, wts)
np.dot(wts, M1)


M1 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
wts = np.array([5, 4, 3])

np.einsum('ij,k', M1, wts).sum(axis=2)
M1*wts[0] + M1 * wts[1] + M1 * wts[2]


%timeit np.einsum('ij,k', M1, wts).sum(axis=2)
%timeit M1*wts[0] + M1 * wts[1] + M1 * wts[2]



np.inner(M1, wts)

np.einsum('i,i', a, b)

np.tensordot(M1, wts, axes=((1, 0)))

np.average(M1, axis=0, weights=wts)[1]
np.average(solar_x, axis=0, weights=[3/4, 1/4])[1]

np.einsum('ij,j', a, b)
array([ 30,  80, 130, 180, 230])
np.einsum(a, [0,1], b, [1])
array([ 30,  80, 130, 180, 230])
np.dot(a, b)
array([ 30,  80, 130, 180, 230])

np.tensordot(M1, wts, axes=(0))

a = np.arange(60.).reshape(3,4,5)
b = np.arange(24.).reshape(4,3,2)
a
b
c = np.tensordot(a,b, axes=([1,0],[0,1]))
c.shape

M1 = np.zeros(shape=(k, k))



def gs2(s):
    Ds = np.zeros(shape=(p.k, p.k))
    t1 = timer()
    for h in range(p.h):
        Ds = np.add(Ds, geths(h, s))
    t2 = timer()
    print(f'build Ds {(t2 - t1): 6.4f}')
    t1 = timer()
    Dsinv = np.linalg.inv(Ds)
    t2 = timer()
    print(f'invert Ds {(t2 - t1): 6.4f}')

    ds = diffs[s, ].reshape(p.k, 1)

    steprow = Dsinv.dot(ds).flatten()

    return steprow


def gs(s):
    # Ds = (whs[:, s] * xxp).sum()
    Ds = np.zeros(shape=(p.k, p.k))
    t1 = timer()
    for h in range(p.h):
        Ds = np.add(Ds, geths(h, s))
    t2 = timer()
    print(f'build Ds {(t2 - t1): 6.4f}')
    t1 = timer()
    Dsinv = np.linalg.inv(Ds)
    t2 = timer()
    print(f'invert Ds {(t2 - t1): 6.4f}')
    # print(Dsinv.shape)
    ds = diffs[s, ].reshape(p.k, 1)
    # print(ds.shape)
    steprow = Dsinv.dot(ds).flatten()
    # print(steprow)
    # print(ds)
    # step = 1 / Ds * ds.T
    return steprow

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
    # diffs = diffs * diff_weights
    diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs # np.array(diffs)  # note that this is np, not jnp!

# %% play with matrix speed
def f(vec):
    return vec.dot(vec.T)
# xxp = np.apply_along_axis(f, 1, arr=xmat)  # (p.h, )

# xxp = xmat.dot(xmat.T)
# xxp = xmat.T.dot(xmat)  # k x k
# xxp.shape

# h = 0
# xh = xmat[h].reshape(xh.size, 1)
# xxph = xh.dot(xh.T)
# xxph.shape  # k x h
# xxph = xmat[h].dot(xmat[h].T)
# 7 * xxph

# %% choose problem
p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=3, k=3, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=10, k=8, xsd=.1, ssd=.5, pctzero=.2)
p = mtp.Problem(h=20000, s=20, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=30000, s=30, k=20, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=35000, s=40, k=25, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=50000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.2)
p = mtp.Problem(h=100000, s=80, k=40, xsd=.1, ssd=.5, pctzero=.2)


# %% add noise and set problem up
p.h
p.s
p.k
p.xmat
p.wh
p.whs
p.geotargets

xmat = p.xmat
wh = p.wh

# now add noise to geotargets
np.random.seed(1)
gnoise = np.random.normal(0, .05, p.k * p.s)
gnoise = gnoise.reshape(p.geotargets.shape)
geotargets = p.geotargets * (1 + gnoise)

dw = np.full(geotargets.shape, 1)

beta0 = np.full(geotargets.shape, 0.)

# %% functions for solving problem inefficiently
def geths(h, s):
    xh = xmat[h].reshape(xmat.shape[1], 1)
    xxph = xh.dot(xh.T)  # k x k -- the same for all s, for a given h
    # print(xxph)
    # print(whs[h, s])
    whsxpx = whs[h, s] * xxph  # initial values same for all s of an h
    # print(whs[h, s])
    return whsxpx

def gs(s):
    # Ds = (whs[:, s] * xxp).sum()
    Ds = np.zeros(shape=(p.k, p.k))
    # t1 = timer()
    for h in range(p.h):
        Ds = np.add(Ds, geths(h, s))
    # t2 = timer()
    # print(f'build Ds {(t2 - t1): 6.4f}')
    # t1 = timer()
    Dsinv = np.linalg.inv(Ds) # k x k
    # t2 = timer()
    # print(f'invert Ds {(t2 - t1): 6.4f}')
    # print(Dsinv.shape)  # k x k
    ds = diffs[s, ].reshape(p.k, 1)  # k x 1
    # print(ds)
    # print(ds.shape)
    steprow = Dsinv.dot(ds).flatten() # a row has k elements, there will be s rows
    # print(steprow)
    # print(ds)
    # step = 1 / Ds * ds.T
    return steprow


# a = np.array([[1, 2], [3, 4]])  # invDs, k x k
# b = np.array([[5], [6]])  # ds k x 1
# a.shape
# b.shape
# a.dot(b)  # k x 1
# a.dot(b).flatten()  # 1 x k
# # np.einsum('il,ijk->ljk', whs, xxp2)
# np.einsum('ij,ij->j', a, b)

# a = np.array([[1, 2], [3, 4]])  # invDs, k x k
# b = np.array([[5], [6]])  # ds k x 1

# # diffs must be s x k, let s = 3 and k = 2
# tdiffs = np.array([[1, 2],
#                    [4, 5],
#                    [7, 8]])
# tdiffs.shape  # good

# # invD must be s x k x k
# tinvD = np.array([
#                   [[1, 2], [3, 4]],
#                   [[5, 6], [7, 8]],
#                   [[9, 10], [11, 12]]
#                   ])
# tinvD.shape  # good

# # step will be s x k; each row of step will be dot product of a row of tdiffs with a row of tinvD
# np.einsum('ijk,ik->ij', tinvD, tdiffs)

# s = 2
# np.dot(tinvD[s,:,:], tdiffs[s,:])





# %% solve problem inefficiently
beta = beta0.copy()
count = 0
maxiter = 8

a = timer()
while count <= maxiter:
    count += 1
    beta_x = jnp.dot(beta, xmat.T)
    exp_beta_x = jnp.exp(beta_x)
    delta = jnp.log(wh / exp_beta_x.sum(axis=0))
    beta_xd = (beta_x + delta).T  # this is the same as before but with new betax delta
    whs = jnp.exp(beta_xd)
    targs = jnp.dot(whs.T, xmat)  # s x k
    diffs = targs - geotargets  # s x k
    sspd = np.square(diffs / targs * 100.).sum()
    rmse = np.sqrt(sspd / beta.size)

    step = np.zeros(beta.shape)
    t1 = timer()
    for s in range(beta.shape[0]):
        step[s, ] = gs(s)
    t2 = timer()
    steptime = t2 - t1
    print(f' {count: 4}  {rmse:10.4f}    {steptime: 6.4f}')
    beta = beta - step

b = timer()
b - a




# %% solve problem efficiently
beta = beta0.copy()
# beta = np.full(beta0.shape, 0.5)
count = 0
maxiter = 8

xxp = np.einsum('ij,ik->ijk', xmat, xmat)

a = timer()
while count <= maxiter:
    count += 1
    beta_x = jnp.dot(beta, xmat.T)
    exp_beta_x = jnp.exp(beta_x)
    delta = jnp.log(wh / exp_beta_x.sum(axis=0))
    beta_xd = (beta_x + delta).T  # this is the same as before but with new betax delta
    whs = jnp.exp(beta_xd)
    targs = jnp.dot(whs.T, xmat)  # s x k
    diffs = targs - geotargets  # s x k
    pdiffs = diffs / targs * 100.
    sspd = np.square(diffs / targs * 100.).sum()
    rmse = np.sqrt(sspd / beta.size)
    l2norm = norm(pdiffs, 2)

    t1 = timer()
    D = np.einsum('il,ijk->ljk', whs, xxp)
    invD = np.linalg.inv(D)
    step = np.einsum('ijk,ik->ij', invD, diffs)
    t2 = timer()
    steptime = t2 - t1
    print(f' {count: 4}  {l2norm: 10.2f}  {rmse:10.4f}    {steptime: 6.4f}')
    beta = beta - step

b = timer()
b - a

pdiffs = diffs / targs * 100.
np.round(np.quantile(pdiffs, [0, .1, .25, .5, .75, .9, 1]), 2)
norm(pdiffs, 2)
norm(diffs, 2)
np.max(np.abs(pdiffs))


# %% older stuff
beta_x = jnp.dot(beta, xmat.T)
exp_beta_x = jnp.exp(beta_x)
delta = jnp.log(wh / exp_beta_x.sum(axis=0))
delta
# get whs
# add the delta vector of household constants to every row of beta_x and transpose
beta_xd = (beta_x + delta).T  # this is the same as before but with new betax delta
whs = jnp.exp(beta_xd)
whs
# whs - p.whs
# ws = whs.sum(axis=0)

targs = jnp.dot(whs.T, xmat)  # s x k
diffs = targs - geotargets  # s x k
diffs
sspd = np.square(diffs / targs * 100.).sum()
sspd
rmse = np.sqrt(sspd / beta.size)
rmse

# xh = xmat[h].reshape(xh.size, 1)
# xxph = xh.dot(xh.T)  # k x k
# weight each xxph by the ws[h] for a state
# to get k x k matrix for a state

def geths(h, s):
    xh = xmat[h].reshape(xmat.shape[1], 1)
    xxph = xh.dot(xh.T)  # k x k -- the same for all s, for a given h
    # print(xxph)
    # print(whs[h, s])
    whsxpx = whs[h, s] * xxph  # initial values same for all s of an h
    # print(whs[h, s])
    return whsxpx
# h = 0
# s = 0
# geths(h, s)
# geths(0, 0)
# geths(1, 0)
# geths(2, 0)

# geths(0, 1)
# geths(1, 1)
# geths(2, 1)

# whs[0, ]


# Ds = np.zeros(shape=(geotargets.shape[1], geotargets.shape[1]))
# Ds
# for h in range(p.h):
#     Ds = np.add(Ds, geths(h, 0))
# Ds

# np.add(geths(0, 0), geths(1, 0))

def gs(s):
    # Ds = (whs[:, s] * xxp).sum()
    Ds = np.zeros(shape=(p.k, p.k))
    # a = timer()
    for h in range(p.h):
        Ds = np.add(Ds, geths(h, s))
    # b = timer()
    # print(f'build Ds: f')
    # print(Ds)
    Dsinv = np.linalg.inv(Ds)
    # print(Dsinv.shape)
    ds = diffs[s, ].reshape(p.k, 1)
    # print(ds.shape)
    steprow = Dsinv.dot(ds).flatten()
    # print(steprow)
    # print(ds)
    # step = 1 / Ds * ds.T
    return steprow
# gs(0).shape

step = np.zeros(beta.shape)
for s in range(beta.shape[0]):
    step[s, ] = gs(s)
step
beta = beta - step
beta

# betax = beta.dot(xmat.T)  # s x h
# const = betax.max(axis=0)  # 1 x h
# betax = jnp.subtract(betax, const)
# ebetax = jnp.exp(betax)  # s x h
# logdiffs = betax - jnp.log(ebetax.sum(axis=0)) # s x h
# shares = jnp.exp(logdiffs)  # s x h
# whs = jnp.multiply(wh, shares).T  # h x s



# figure out tpc step
    Ds = np.zeros(shape=(30, 30))
    for h in range(p.h):
        Ds = np.add(Ds, geths(h, s))
 Dsinv = np.linalg.inv(Ds)

amat = np.random.rand(30, 30)
a = timer()
for i in range(50*20):
    amatinv = np.linalg.inv(amat)
b = timer()
b - a



betax = beta.dot(xmat.T)  # s x h
# adjust betax to make exponentiation more stable numerically
# subtract column-specific constant (the max) from each column of betax
const = betax.max(axis=0)  # 1 x h
betax = jnp.subtract(betax, const)
ebetax = jnp.exp(betax)  # s x h
# delta = jnp.log(ebetax.sum(axis=0))  # ?? I think this is delta (h, )

beta_x = jnp.exp(jnp.dot(beta, xmat.T))
delta = jnp.log(wh / beta_x.sum(axis=0))
delta

logdiffs = betax - jnp.log(ebetax.sum(axis=0)) # s x h
shares = jnp.exp(logdiffs)  # s x h
whs = jnp.multiply(wh, shares).T  # h x s
targs = jnp.dot(whs.T, xmat)  # s x k
diffs = targs - geotargets  # s x k
diffs
sspd = np.square(diffs).sum()
sspd

step = np.zeros(beta.shape)
for s in range(beta.shape[0]):
    step[s, ] = getstate(s)
step
beta = beta - step


jacfn = jax.jacfwd(fgp.jax_targets_diff)

def getstate(s):
    jacmat = jacfn(beta[s, ], wh, xmat, geotargets[s, ], dw[s, ])
    ds = diffs[s, ].reshape(p.k, 1)
    Dsinv = np.linalg.inv(jacmat)
    step = Dsinv.dot(ds.shape)
    return step



# jacmat = jacfn(beta, wh, xmat, geotargets, dw)

# jax.jacwfd

# # pick an h
h = 0
xh = xmat[h, ].reshape(p.k, 1)  # column vector
xh.shape
xxph = xh.dot(xh.T)
xxph  # k x k

# jacmat = jacfn(beta[0, ], wh, xmat, geotargets[0, ], dw[0, ])  # see if I can calc this
# for state 0 we want
# jacmat
# DeviceArray([[151.43803315, -51.23259434],
#              [-49.28636185, 344.01464634]], dtype=float64)

# ds = diffs[0, ].reshape(p.k, 1)
# Dsinv = np.linalg.inv(jacmat)

# Dsinv.shape
# ds.shape

# betas = beta[0, ] - Dsinv.dot(ds.shape)




np.apply_along_axis(getstate, 1, arr=beta)
beta.sum(axis=0)


%timeit np.apply_along_axis(sum, 0, xmat)
%timeit xmat.sum(axis=0)

getstate(0)
getstate(1)
getstate(2)







def getDs(s, )







# xpx = xmat.dot(xmat.T)  # h x h
xpx = xmat.T.dot(xmat) # k x k
xpx.shape

xh1 = np.array([[1, 7, 5]]).T  # column vector
xh1 = np.array([[1, 7, 5]])  # row vector
xh1.shape
xxp = xh1.dot(xh1.T)
xxp


xh2 = np.array([3, 4, 6])
np.outer(xh1.T, xh1)  # k x k
np.outer(xh2.T, xh2)  # k x k

dot()

np.outer(xh1, xh1.T)


xh2 = np.array([[1, 7],  # 50
                [3, 4],  # 25
                [5, 6]])  # 61
xh2.shape
xh2.dot(xh2.T)
xh2.T.dot(xh2)

h=2
xh2[h].dot(xh2[h].T)
xh2[h].T.dot(xh2[h])

np.apply_along_axis(getstate, 1, arr=beta)

np.outer(xh2, xh2.T)
np.outer(xh2[h], xh2[h].T)
np.outer(xh2.T[h], xh2[h])

np.inner(hx2, xh2.T)

def f(vec):
    return vec.dot(vec.T)
np.apply_along_axis(f, 1, arr=xh2)


s = 1
ds = diffs[s]
Ds =







def get_whs_logs(beta_object, wh, xmat, geotargets):
    # note beta is an s x k matrix
    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    betax = beta.dot(xmat.T)
    # adjust betax to make exponentiation more stable numerically
    # subtract column-specific constant (the max) from each column of betax
    const = betax.max(axis=0)
    betax = jnp.subtract(betax, const)
    ebetax = jnp.exp(betax)
    # print(ebetax.min())
    # print(np.log(ebetax))
    logdiffs = betax - jnp.log(ebetax.sum(axis=0))
    shares = jnp.exp(logdiffs)
    whs = jnp.multiply(wh, shares).T
    return whs







# %% functions needed for residuals
def jax_get_delta(wh, beta, xmat):
    beta_x = jnp.exp(jnp.dot(beta, xmat.T))
    delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta


def jax_get_diff_weights(geotargets, goal=100):
    # establish a weight for each target that, prior to application of any
    # other weights, will give each target equal priority
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
    # diffs = diffs * diff_weights
    diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs # np.array(diffs)  # note that this is np, not jnp!

# %% map function over 3d array
np.array(map(f, x))

a = np.arange(24).reshape(2,3,4)
a
a.shape
# Sum over axes 0 and 2. The result has same number of dimensions as the original array:
np.apply_over_axes(np.sum, a, [0,2])

# if np.linalg.cond(Ds) < 1 / sys.float_info.epsilon:
#     Dsinv = np.linalg.inv(Ds)
# else:
#     print("Ds is singular, cannot be inverted")
# invD = np.linalg.inv(D)  # invert all of them

a1 = np.array([0, 1, -3,
               -3, -4, 4,
               -2, -2, 1]).reshape(3, 3)
a1
np.linalg.inv(a1)  # a1 is invertible
np.linalg.cond(a1)  # 120

a2 = np.array(
    [[ 0,  1,  2],
     [ 3,  4,  5],
     [ 6,  7,  8]])
np.linalg.inv(a2)  # a2 is NOT invertible
np.linalg.cond(a2)  # 2.4e16

a = np.array([a1,a2])
a.shape
np.linalg.cond(a)  # 2.4e16
np.linalg.inv(a)
tmp = np.apply_over_axes(np.sum, a, [1,2])
tmp.shape  # 2, 1, 1

np.apply_over_axes(np.linalg.cond, a, [1, 2])  # does not work
def g(m, axis):
    print("axis: ", axis)
    print(m)
    # np.array(np.linalg.cond(m[axis,:,:]))
    return m # np.array(np.linalg.cond(m[axis,:,:]))
z = np.apply_over_axes(g, a, [1, 2])  # does not work
z.shape
np.apply_over_axes(g, a, [1, 2])

np.apply_over_axes(f, a, [1, 2])

def f(m):
    return np.linalg.cond(m)

def f2(m):
    if np.linalg.cond(m) < 1 / sys.float_info.epsilon:
        print("inverting...")
        m2 = np.linalg.inv(m)
    else:
        print("can't invert...")
        m2 = m
    return m2

for m in range(a.shape[0]):
    print(m)
    print(f2(a[m,:,:]))

f(a)

f2(a)
vf2 = np.vectorize(f2)
vf2(a)

vfunc = np.vectorize(myfunc)
vfunc([1, 2, 3, 4], 2)

a[0,:]
np.linalg.inv(a[0,:])  # singular
np.linalg.inv(a[1,:])




