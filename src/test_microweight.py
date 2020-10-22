# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:22:13 2020

@author: donbo
"""

# %% imports
import numpy as np
import src.make_test_problems as mtp
import src.microweight as mw


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% functions
# def targs(x, wh, xmat):
#     return np.dot(xmat.T, wh * x)


# def sspd(x, wh, xmat, targets):
#     #  sum of squared percentage differences
#     diffs = np.dot(xmat.T, wh * x) - targets
#     pdiffs = diffs / targets * 100
#     return np.square(pdiffs).sum()


# %% make problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10, s=2, k=2)
p = mtp.Problem(h=40, s=2, k=3)
p = mtp.Problem(h=1000, s=10, k=5)
p = mtp.Problem(h=5000, s=10, k=5)
p = mtp.Problem(h=10000, s=10, k=10)
p = mtp.Problem(h=40000, s=10, k=30)
p = mtp.Problem(h=100000, s=10, k=5)
p = mtp.Problem(h=100000, s=10, k=30)

np.random.seed(1)
noise = np.random.normal(0, .0125, p.k)
noise
ntargets = p.targets * (1 + noise)
# ntargets = p.targets

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets, geotargets=p.geotargets)


# %% reweight the problem
opts = {'crange': 0.001, 'quiet': False}
opts = {'crange': 0.001, 'xlb':0, 'xub':100, 'quiet': False}
rw1a = prob.reweight(method='ipopt', options=opts)
# dir(rw1a)
rw1a.sspd

opts = {'crange': 0.0001, 'xlb': 0, 'xub':1e5, 'quiet': False}
rw1b = prob.reweight(method='ipopt', options=opts)
rw1b.sspd
rw1b.elapsed_seconds

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
        'lsmr_tol': 1e-6,'scaling': True, 'max_iter': 20}

opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls', 'max_iter': 20}
opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls', 'scaling': True, 'max_iter': 200}

opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls',
        'lsmr_tol': 1e-6,
        'scaling': True, 'max_iter': 20}

opts = {'xlb': 0.0, 'xub': 100.0, 'method': 'bvls',
        'lsmr_tol': 1e-6,
        'scaling': False, 'max_iter': 20}

rw4 = prob.reweight(method='lsq')
rw4 = prob.reweight(method='lsq', options=opts)
rw4.sspd

rw5 = prob.reweight(method='minNLP')
rw5 = prob.reweight(method='minNLP', options={'xtol': 1e-6})
rw5.sspd


ntargets
rw1a.targets_opt
rw1b.targets_opt
rw2.targets_opt
rw3.targets_opt
rw4.targets_opt

# time
rw1a.elapsed_seconds
rw1b.elapsed_seconds
rw2.elapsed_seconds
rw3.elapsed_seconds
rw4.elapsed_seconds

# sum of squared percentage differences
rw1a.sspd
rw1b.sspd
rw2.sspd
rw3.sspd
rw4.sspd

# percent differences
rw1a.pdiff
rw1b.pdiff
rw2.pdiff
rw3.pdiff
rw4.pdiff

# distribution of g values
np.quantile(rw1a.g, qtiles)
np.quantile(rw1b.g, qtiles)
np.quantile(rw2.g, qtiles)
np.quantile(rw3.g, qtiles)
np.quantile(rw4.g, qtiles)
np.quantile(rwmin.g, qtiles)

rw1a.g.sum()
rw1b.g.sum()
rw2.g.sum()
rw3.g.sum()
rw4.g.sum()



# %% geoweight the problem
gw1 = prob.geoweight(method='qmatrix')

uo = {'max_iter': 20}
gw1 = prob.geoweight(method='qmatrix', user_options=uo)

gw1.method_result.iter_opt

# dir(gw1)
gw1.method
gw1.elapsed_seconds
gw1.geotargets_opt
gw1.whs_opt
dir(gw1.method_result)

gw2 = prob.geoweight(method='qmatrix-ec')
uo = {'max_iter': 20}
so = {'objective': 'QUADRATIC'}
gw2 = prob.geoweight(method='qmatrix-ec', user_options=uo)
gw2 = prob.geoweight(method='qmatrix-ec', solver_options=so)
gw2.method
gw2.geotargets_opt
gw2.sspd

gw3 = prob.geoweight(method='poisson')
gw3.method
gw3.elapsed_seconds
gw3.geotargets_opt
gw3.whs_opt
dir(gw3.method_result)

# sum of squared percentage differences
gw1.sspd
gw2.sspd
gw3.sspd

np.square((gw1.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw2.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw3.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()


# %% check
import src.poisson as ps
dir(gw3)
gw1.whs_opt
gw2.whs_opt
gw3.whs_opt


# %% test linear least squares
# here we test ability to hit national (not state) targets, creating
# weights that minimize sum of squared differences from targets
import scipy
from scipy.optimize import lsq_linear
from numpy.random import seed
from timeit import default_timer as timer

p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=100, s=1, k=4)
p = mtp.Problem(h=3000, s=1, k=10)
p = mtp.Problem(h=30000, s=1, k=20)
p = mtp.Problem(h=300000, s=1, k=30)
p = mtp.Problem(h=500000, s=1, k=50)

seed(1)
r = np.random.randn(p.targets.size) / 50  # random normal
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(r, q)
targets = (p.targets * (1 + r)).flatten()

# interlude ----
prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=targets)

uo = {'crange': 0.0001, 'quiet': False}
so = {'max_iter': 200}
rw1a = prob.reweight(method='ipopt', user_options=uo, solver_options=so)
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
diff_weights =np.ones_like(targets)

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



