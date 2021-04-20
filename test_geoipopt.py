

# %% imports

# Load the autoreload extension
# %load_ext autoreload
# %reload_ext autoreload

# Autoreload reloads modules before executing code
# 0: disable
# 1: reload modules imported with %aimport
# 2: reload all modules, except those excluded by %aimport
# %autoreload 2

import importlib
import numpy as np
import scipy.sparse as sps
import cyipopt as cy

import src.make_test_problems as mtp
import src.geoweight_ipopt as gwi
import src.microweight as mw

# %% reimports
importlib.reload(mtp)
importlib.reload(gwi)
importlib.reload(mw)


# %% create data
# diff = geotargets_opt - geotargets
# pctdiff = diff / geotargets * 100

# %% base options
opt_base = {'xlb': .05, 'xub': 50,
         'crange': 0.0,
         'print_level': 0,
         'file_print_level': 5,
         # 'ccgoal': 10000,
         'addup': False,
         'max_iter': 100,
         'linear_solver': 'ma57',  # ma27, ma77, ma57, ma86 work, not ma97
         'quiet': False}

# %% test sparse version

p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=3, k=3, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=10, k=8, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=20000, s=20, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4)

geotargets = p.geotargets

# %% optionally add noise
np.random.seed(1)
noise = np.random.normal(0, .01, p.geotargets.size)
# np.round(noise * 100, 2)

geotargets = p.geotargets * (1 + noise.reshape((p.s, p.k)))

# %% run problem
opt_sparse = opt_base.copy()
opt_sparse.update({'output_file': '/home/donboyd/Documents/test_sparse.out'})
opt_sparse.update({'addup': True})
opt_sparse.update({'crange': .02})
opt_sparse
# djb here
jst, jsa = gwi.ipopt_geo(p.wh, p.xmat, geotargets, options=opt_sparse)
jst.shape
jsa.shape

res = gwi.ipopt_geo(p.wh, p.xmat, geotargets, options=opt_sparse)
res.elapsed_seconds
res.ipopt_info['status_msg']
qsums = res.Q_best.sum(axis=1)
np.quantile(qsums, q=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])

p.geotargets
res.geotargets
# geotargets
res.geotargets_opt
res.ipopt_info['g']
pdiff_init = res.geotargets_init / geotargets * 100 - 100
pdiff_opt = res.geotargets_opt / geotargets * 100 - 100
np.round(pdiff_init, 2)
np.round(pdiff_opt, 2)

res.g
np.quantile(res.g, q=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])

res.ipopt_info

res.Q_best
checksums = res.Q_best.sum(axis=1)
np.quantile(checksums, q=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])
p.whs
res.whs_opt


# %% additional examination
targ_opt = np.dot(p.whs.T, p.xmat)
xmat = p.xmat
wh = p.wh
geotargets = p.geotargets
dir(p)

checksums = res.Q_best.sum(axis=1)
np.quantile(checksums*100, q=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])


# %% array play
p2 = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
opt_sparse.update({'addup': True})
opt_sparse.update({'addup': False})
check = gwi.ipopt_geo(p2.wh, p2.xmat, p2.geotargets, options=opt_sparse)

check.Q_best
check.Q_best.sum(axis=1) * 100

jst.shape
jsa.shape
js = vstack([jst, jsa])
js.shape
js.shape[0]

a = np.array([0, 1, 2, 3])
b = np.array([7, 8, 9])
np.concatenate((a, b), axis=0)


check[0].max()
check.shape
sps.find(check)[0].max()
tmp2 = check.todense()
whs
tmp2[6, :]

from scipy.sparse import coo_matrix, csr_matrix, vstack
A = csr_matrix([[1, 2], [3, 4], [0, 1]])
B = csr_matrix([[5, 6]])
A.shape
B.shape

A.todense()
B.todense()
ABmat = vstack([A, B])
ABmat.todense()

sps.find(A)
sps.find(B)
sps.find(ABmat)

vstack([A, B]).toarray()


p2.whs
p2.wh
p2.whs.sum(axis=1) 

h = p2.h
s = p2.s

# row h * s  0 0 0 1 1 1 2 2 2 ... 19 19 19  # repeat h s times
# col s      0 3 6 1 4 7 2 5 8 ... col = row + s * (0, 1, 2)  
row = np.repeat(np.arange(0, h), s)
state_idx = np.tile(np.arange(0, s), h)
col = row + state_idx * h
# we'll use init whs values for the coefficients, not these opt values
nzvalues = p2.whs[row, state_idx]
jsparse_addup = sps.csr_matrix((nzvalues, (row, col)))
print(jsparse_addup)
p2.whs
tmp = jsparse_addup.todense()
tmp.shape
tmp[1, :]




# # creating an array from a Python sequence
# np.array([i**2 for i in range(5)])
# # array([ 0,  1,  4,  9, 16])

# # creating an array filled with ones
# np.ones((2, 4))
# # array([[ 1.,  1.,  1.,  1.],
# #        [ 1.,  1.,  1.,  1.]])

# # creating an array of evenly-spaced points
# np.linspace(0, 10, 5)
# # array([  0. ,   2.5,   5. ,   7.5,  10. ])


# # creating an array of a specified datatype
# np.array([1.5, 3.20, 5.78], dtype=int)
