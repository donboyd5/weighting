

# # -*- coding: utf-8 -*-


# %% imports
# for checking:
# import sys; print(sys.executable)
# print(sys.path)
import importlib

import numpy as np

import scipy
from scipy.optimize import lsq_linear
from numpy.random import seed
from timeit import default_timer as timer

import src.make_test_problems as mtp
import src.microweight as mw


# %% reimports
importlib.reload(mw)


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% functions
def targs(targvec, div=50, seed=seed(1234)):
    r = np.random.randn(targvec.size) / 50  # random normal
    targets = (targvec * (1 + r)).flatten()
    return targets


# %% make problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10, s=2, k=2)
p = mtp.Problem(h=40, s=2, k=3)
p = mtp.Problem(h=1000, s=10, k=5)
p = mtp.Problem(h=5000, s=10, k=5)
p = mtp.Problem(h=10000, s=10, k=10)
p = mtp.Problem(h=30000, s=50, k=20)
p = mtp.Problem(h=40000, s=30, k=30)
p = mtp.Problem(h=50000, s=50, k=30)
p = mtp.Problem(h=100000, s=10, k=5)
p = mtp.Problem(h=100000, s=50, k=30)
p = mtp.Problem(h=1000000, s=50, k=30)

p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=3, k=3, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=10, k=8, xsd=.1, ssd=.5, pctzero=.2)
p = mtp.Problem(h=20000, s=20, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=30000, s=30, k=20, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.5)
p = mtp.Problem(h=50000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.2)


# %% add noise and set problem up
p.h
p.s
p.k

np.random.seed(1)
targs(p.targets)
noise = np.random.normal(0, .01, p.k)
noise
ntargets = p.targets * (1 + noise)
# ntargets = p.targets

# now add noise to geotargets
np.random.seed(1)
gnoise = np.random.normal(0, .01, p.k * p.s)
gnoise = gnoise.reshape(p.geotargets.shape)
ngtargets = p.geotargets * (1 + gnoise)
# ngtargets = p.geotargets

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets, geotargets=ngtargets)


# %% what if we normalize xmat and geotargets?
# divide all xmat columns by their sum
# divide all corresponding geotargets
p.xmat.shape
scale = p.xmat.sum(axis=0) / 10000.0
scale.shape
ngtargets.shape  # s x k

xmat = np.divide(p.xmat, scale)
xmat.shape
xmat.sum(axis=0)

ngtargets2 = np.divide(ngtargets, scale)
ngtargets2.shape
ngtargets2.sum(axis=0)


prob = mw.Microweight(wh=p.wh, xmat=xmat, targets=ntargets, geotargets=ngtargets2)


# %% define options

uo = {'qmax_iter': 10}
uo = {'qmax_iter': 1, 'independent': True}
uo = {'qmax_iter': 10, 'quiet': True}
uo = {'qmax_iter': 3, 'quiet': True, 'verbose': 2}

# raking options (there aren't really any)
uoqr = {'qmax_iter': 10}

# empcal options
uoempcal = {'qmax_iter': 10, 'objective': 'ENTROPY'}
uoempcal = {'qmax_iter': 10, 'objective': 'QUADRATIC'}

# ipopt options
uoipopt = {'qmax_iter': 30,
           'quiet': True,
           'xlb': 0.001,
           'xub': 1000,
           'crange': .000001,
           'linear_solver': 'ma57'
           }


# lsq options
uolsq = {'qmax_iter': 10,
         'verbose': 0,
         'xlb': 0.001,
         'xub': 1000,
         'scaling': False,
         # bvls (default) or trf - bvls usually faster, better
         'method': 'bvls',
         'lsmr_tol': 'auto'  # 'auto'  # 'auto' or None
         }

# geoipopt options
geoipopt_base = {# 'xlb': .2, 'xub': 2, # default 0.1, 10.0
         # 'crange': 0.0,  # default 0.0
         # 'print_level': 0,
         # 'file_print_level': 5,
         # 'ccgoal': 10000,
         # 'addup': False,  # default is false
         'max_iter': 100,
         'linear_solver': 'ma57',  # ma27, ma77, ma57, ma86 work, not ma97
         'quiet': False}

geoipopt_opts = geoipopt_base.copy()
geoipopt_opts.update({'output_file': '/home/donboyd/Documents/test_sparse.out'})
geoipopt_opts.update({'addup': False})
geoipopt_opts.update({'addup': True})
geoipopt_opts.update({'crange': .005})
geoipopt_opts.update({'addup_range': .0})
geoipopt_opts.update({'linear_solver': 'ma86'})
geoipopt_opts.update({'xlb': .01})
geoipopt_opts.update({'xub': 10.0})
geoipopt_opts


# %% geoweight the problem
# gw1 = prob.geoweight(method='qmatrix', options=uoqr)
# gw2 = prob.geoweight(method='qmatrix-lsq', options=uolsq)
# gw3 = prob.geoweight(method='qmatrix-ipopt', options=uoipopt)
# gw4 = prob.geoweight(method='qmatrix-ec', options=uoempcal)
gw5 = prob.geoweight(method='poisson', options=uo)
gw5a = prob.geoweight(method='poisson_autodiff', options=uo)
# gw5b = prob.geoweight(method='poisson', options=uo)
# gw6 = prob.geoweight(method='geoipopt', options=geoipopt_opts)

# djb: overflow encountered in exp
#  beta_x = np.exp(np.dot(beta, xmat.T))
# so this is happening in the non-jax function
gw5.elapsed_seconds / 60
gw5b.elapsed_seconds / 60
gw6.elapsed_seconds / 60

np.corrcoef(gw5b.whs_opt.flatten(), gw6.whs_opt.flatten())
gw6.sspd

gw1.sspd
gw2.sspd
gw3.sspd

gw5.sspd
gw5a.sspd
gw5b.sspd
gw6.sspd


gw = gw5b  # one of the above
# dir(gw)
gw.method
gw.sspd
gw.elapsed_seconds

targets_check = np.dot(gw.whs_opt.T, p.xmat)
targets_check
ngtargets

prob.geotargets
gw.geotargets_opt
gw.pdiff
np.round(gw.pdiff, 2)
gw.whs_opt
gw.whs_opt.sum(axis=1) - p.wh
dir(gw.method_result)
np.max(np.abs(gw.pdiff))


gw.whs_opt

gw.sspd

gw3 = prob.geoweight(method='poisson')
gw3.method
gw3.sspd
gw3.elapsed_seconds
gw3.geotargets_opt
gw3.whs_opt
dir(gw3.method_result)

# sum of squared percentage differences
gw1.sspd
gw1a.sspd
gw1b.sspd
gw2.sspd
gw3.sspd

np.square((gw1.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw2.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw3.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()


# %% check

dir(gw3)
gw1.whs_opt
gw2.whs_opt
gw3.whs_opt

gw5.whs_opt
gw6.whs_opt

np.round(np.quantile(gw5.whs_opt - gw6.whs_opt, q=qtiles), 2)
np.round(np.quantile(gw5.whs_opt / gw6.whs_opt * 100 - 100, q=qtiles), 2)

gw = gw5
Q = np.divide(gw.whs_opt, p.wh.reshape(-1, 1))
np.round(np.quantile(Q.sum(axis=1), q=qtiles), 2)


# %% reweight the problem
opts = {'crange': 0.001, 'quiet': False}
# opts = {'crange': 0.001, 'xlb':0, 'xub':100, 'quiet': False}
rw1 = prob.reweight(method='ipopt', options=opts)
# dir(rw1a)
rw1.sspd


# so = {'increment': .00001, 'autoscale': False}  # best
opts = {'increment': .001}
opts = {'increment': .00001, 'autoscale': False}
opts = {'increment': .000001, 'autoscale': True}
opts = {'increment': .00000001, 'autoscale': False, 'objective': 'QUADRATIC'}
rw2 = prob.reweight(method='empcal', options=opts)
rw2 = prob.reweight(method='empcal')
rw2.sspd

rw3 = prob.reweight(method='rake')
rw3 = prob.reweight(method='rake', options={'max_rake_iter': 20})
rw3.sspd

opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'trf', 'max_iter': 20}
opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'trf', 'scaling': True, 'max_iter': 50}
opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'trf',
        'lsmr_tol': 1e-6, 'scaling': True, 'max_iter': 20}

opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls', 'max_iter': 20}
opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls', 'scaling': True, 'max_iter': 200}

opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls',
        'lsmr_tol': 1e-6,
        'scaling': True, 'max_iter': 20}

opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls',
        'lsmr_tol': 1e-6,
        'scaling': False, 'max_iter': 20}

rw4 = prob.reweight(method='lsq')
rw4 = prob.reweight(method='lsq', options={'scaling': True})
rw4 = prob.reweight(method='lsq', options=opts)
rw4.sspd
rw4.opts

rw5 = prob.reweight(method='minNLP')
rw5 = prob.reweight(method='minNLP', options={'scaling': False})  # fewer iters
rw5 = prob.reweight(method='minNLP', options={'xtol': 1e-6})
rw5.sspd
rw5.opts

ntargets
rw1.targets_opt
rw2.targets_opt
rw3.targets_opt
rw4.targets_opt
rw5.targets_opt

# time
rw1.elapsed_seconds
rw2.elapsed_seconds
rw3.elapsed_seconds
rw4.elapsed_seconds
rw5.elapsed_seconds

# sum of squared percentage differences
rw1.sspd
rw2.sspd
rw3.sspd
rw4.sspd
rw5.sspd

# percent differences
rw1.pdiff
rw2.pdiff
rw3.pdiff
rw4.pdiff
rw5.pdiff

# distribution of g values
np.quantile(rw1.g, qtiles)
np.quantile(rw2.g, qtiles)
np.quantile(rw3.g, qtiles)
np.quantile(rw4.g, qtiles)
np.quantile(rw5.g, qtiles)

rw1.g.sum()
rw2.g.sum()
rw3.g.sum()
rw4.g.sum()
rw5.g.sum()


# %% test linear least squares
# here we test ability to hit national (not state) targets, creating
# weights that minimize sum of squared differences from targets


# %% ..test problem definition
p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=10, s=1, k=3)
p = mtp.Problem(h=100, s=1, k=4)
p = mtp.Problem(h=3000, s=1, k=10)
p = mtp.Problem(h=30000, s=1, k=20)
p = mtp.Problem(h=30000, s=1, k=50)
p = mtp.Problem(h=300000, s=1, k=30)
p = mtp.Problem(h=500000, s=1, k=50)

p.h
p.s
p.k
p.xmat.shape


# %% ..add noise
seed(1234)
r = np.random.randn(p.targets.size) / 50  # random normal
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(r, q)
targets = (p.targets * (1 + r)).flatten()


init_pdiff = p.targets / targets * 100 - 100
np.round(init_pdiff, 1)

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=targets)


# interlude ----
# scale = 1e6
# prob = mw.Microweight(wh=p.wh, xmat=p.xmat / scale, targets=targets / scale)

scale = np.abs(targets / 100)
targets / scale
prob2 = mw.Microweight(wh=p.wh, xmat=np.divide(p.xmat, scale), targets=targets / scale)

# %% ..solve
ipo = {'crange': 0.0001, 'quiet': False}
lso = {'max_iter': 200, 'method': 'trf'}

lso2 = {'max_iter': 2000, 'method': 'trf', 'scaling': True,
        'xlb': 0.001, 'xub': 1000, 'tol': 1e-8}

lso3 = {'max_iter': 2000, 'method': 'bvls', 'scaling': False,
        'xlb': 0.001, 'xub': 1000, 'tol': 1e-8}

# rw_ls = prob.reweight(method='lsq')
rw_ls = prob.reweight(method='lsq', options=lso)

rw_ls2 = prob.reweight(method='lsq', options=lso2)
rw_ls3 = prob2.reweight(method='lsq', options=lso3)
# dir(rw_ls)
rw_ls2.pdiff

np.round(rw_ls2.pdiff, 1)
np.round(init_pdiff, 1)

np.round(rw_ls2.pdiff - init_pdiff, 1)
np.round(np.abs(rw_ls2.pdiff) - np.abs(init_pdiff), 1)

np.sum(np.square(init_pdiff))
rw_ls.sspd
rw_ls2.sspd
rw_ls3.sspd


targets
rw_ls2.targets_opt

# opts = {'crange': 0.001, 'xlb':0, 'xub':100, 'quiet': False}
rw_ip = prob.reweight(method='ipopt', options=ipo)
# dir(rw1a)
rw1a.sspd
dir(rw1a)
np.quantile(rw1a.g, q)


# so = {'increment': .00001, 'autoscale': False}  # best
so = {'increment': .001}
so = {'increment': .00001, 'autoscale': False}
so = {'increment': .000001, 'autoscale': True}  # good
so = {'increment': .00000001, 'autoscale': False, 'objective': 'QUADRATIC'}
rw2 = prob.reweight(method='empcal', solver_options=so)
rw2.sspd
rw2.elapsed_seconds

# we are solving Ax = b, where
#   b are the targets and
#   A x multiplication gives calculated targets
# using sparse matrix As instead of A

diff_weights = np.where(targets != 0, 100 / targets, 1)
diff_weights = np.ones_like(targets)

b = targets * diff_weights
b

wmat = p.xmat * diff_weights
At = np.multiply(p.wh.reshape(p.h, 1), wmat)
# At = np.multiply(p.wh.reshape(p.h, 1), p.xmat)
A = At.T
As = scipy.sparse.coo_matrix(A)

# calculate starting percent differences
Atraw = np.multiply(p.wh.reshape(p.h, 1), p.xmat)
# compare sdiff -- starting differences - to res.fun
sdiff = (np.dot(np.full(p.h, 1), Atraw) - targets) / targets * 100
sdiff
np.square(sdiff).sum()

lb = np.full(p.h, 0.1)
ub = np.full(p.h, 100)

# lb = np.full(p.h, 0)
# ub = np.full(p.h, np.inf)

p.h
p.k

start = timer()
res = lsq_linear(As, b, bounds=(lb, ub),
                 method='trf',
                 # tol=1e-6,
                 lsmr_tol='auto',
                 max_iter=500, verbose=2)
end = timer()

end - start

np.abs(sdiff).max()
np.abs(res.fun).max()

np.square(sdiff).sum()
np.square(res.fun).sum()

# compare to cost function
np.square(sdiff).sum() / 2
res.cost
# np.square(res.fun).sum() / 2


# %% by hand
p.wh
p.xmat
targets

p.xmat.shape

init_targs = np.dot(p.wh.T, p.xmat)
init_targs / targets * 100 - 100

# sum of squared diffs
ssd = np.square(init_targs - targets).sum()
ssd / 2 / 1e6

# targets * scale_vector
#    wmat = xmat * scale_vector

At = np.multiply(p.wh.reshape(-1, 1), p.xmat)
A = p.xmat.T

s = 10
lb = p.wh / s
ub = p.wh * s

lsq_info = lsq_linear(p.xmat.T, targets, bounds=(lb, ub),
                      method='bvls',
                      lsq_solver='exact',
                      tol=1e-18,
                      lsmr_tol=1e-10,
                      max_iter=200,
                      verbose=2)
lsq_info.success
lsq_info.x
lsq_info.x / p.wh

end_targs = np.dot(lsq_info.x.T, p.xmat)
end_targs / targets * 100 - 100

lsq_info.fun
end_targs - targets

# sum of squared diffs
ssd = np.square(init_targs - targets).sum()
ssd / 2 / 1e6


np.square(end_targs - targets).sum() / 2 / 1e6
init_pdiff = (init_targs - targets) / targets * 100
end_pdiff = (end_targs - targets) / targets * 100
init_pdiff
end_pdiff
np.square(init_pdiff).sum()
np.square(end_pdiff).sum()


# %% incorporate weights

p = mtp.Problem(h=100, s=1, k=4)
p = mtp.Problem(h=1000, s=1, k=10)
p = mtp.Problem(h=10000, s=1, k=20)

# stack a matrix like a with column of 1s
targets = targs(p.targets)
targets.shape
At = np.concatenate((p.xmat, np.identity(p.wh.size)), axis=1)
A = At.T
As = scipy.sparse.coo_matrix(A)  # sparse matrices not allowed with bvls
b = np.concatenate((targets, p.wh))
At.shape
b.shape

lb = np.concatenate(([0, 0], p.wh * .8))
ub = np.concatenate(([np.Inf, np.Inf], p.wh * 1.2))
lb.shape
b.shape
ub.shape

np.dot(p.wh.T, At) - b
A.shape
At.shape
b.shape

lb = p.wh * .75
ub = p.wh * 1.25

# bvls is very fast but can't use sparse matrices'

res = lsq_linear(A, b, bounds=(lb, ub),
                 method='bvls',
                 tol=1e-8,
                 # lsmr_tol='auto',
                 max_iter=100, verbose=2)

ress = lsq_linear(As, b, bounds=(lb, ub),
                  method='trf',
                  tol=1e-8,
                  # lsmr_tol='auto',
                  max_iter=100, verbose=2)

p.xmat.shape
res.x
p.wh
np.round(res.x / p.wh * 100 - 100, 1)
res.fun
res.fun / b * 100

res.fun[0:10] / b[0:10] * 100
res.fun[11:20] / b[11:20] * 100
