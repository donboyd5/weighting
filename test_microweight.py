

# # -*- coding: utf-8 -*-


# TODO:
# DONE: pass options to poisson
# DONE: poisson scaling
# DONE: consolidate all poisson methods in one file
# DONE: add poisson tpc method (Newton)
# DONE: make sure qmatrix approach is working properly
# DONE: geoipopt basic target scaling
# DONE: use jax to construct jacobian for poisson method
# DONE: use jax jvp to solve for Newton step without constructing jacobian
# reorganize poisson methods to avoid duplicate code
# run puf geoweighting
# run puf analysis
# possibly implement jvp and jit for tpc approach
# investigate improvements to empirical calibration - robustness
# clean up target scaling and make it more consistent
# openblas
# contact Matt J.
# Ceres???



# %% imports
# for checking:
# import sys; print(sys.executable)
# print(sys.path)
import importlib

import numpy as np
import gc  # gc.collect()

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
p = mtp.Problem(h=35000, s=40, k=25, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4) 
p = mtp.Problem(h=50000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.2)

p = mtp.Problem(h=10000, s=15, k=10, xsd=.1, ssd=.5, pctzero=.4)    


# %% add noise and set problem up
p.h
p.s
p.k

# add noise to main targets
np.random.seed(1)
noise = np.random.normal(0, .01, p.k)
ntargets = p.targets * (1 + noise)

# now add noise to geotargets
np.random.seed(1)
gnoise = np.random.normal(0, .05, p.k * p.s)
gnoise = gnoise.reshape(p.geotargets.shape)
ngtargets = p.geotargets * (1 + gnoise)

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets, geotargets=ngtargets)

# %% define poisson options
poisson_opts = {
    'scaling': True,
    'scale_goal': 1e3,
    'init_beta': 0.5,
    'jacmethod': 'jvp',  # vjp, jvp, full, findiff
    'quiet': True}


# %% geoweight: poisson


poisson_opts.update({'jacmethod': 'jvp'})  # default
gwp1 = prob.geoweight(method='poisson', options=poisson_opts)
gwp1.elapsed_seconds
gwp1.sspd

poisson_opts.update({'jacmethod': 'vjp'})  
gwp1a = prob.geoweight(method='poisson', options=poisson_opts)
gwp1a.elapsed_seconds
gwp1a.sspd

poisson_opts.update({'jacmethod': 'full'})
gwp2 = prob.geoweight(method='poisson', options=poisson_opts)
gwp2.elapsed_seconds
gwp2.sspd

poisson_opts.update({'jacmethod': 'findiff'})
gwp3 = prob.geoweight(method='poisson', options=poisson_opts)
gwp3.elapsed_seconds
gwp3.sspd

gwp4 = prob.geoweight(method='poisson-newton', options=poisson_opts)
gwp4.elapsed_seconds
gwp4.sspd

poisson_opts.update({'init_beta': 1.0})
poisson_opts.update({'init_beta': 0.5})
poisson_opts.update({'max_iter': 5})
gwp5 = prob.geoweight(method='poisson-newton_jvplsq', options=poisson_opts)
gwp5.elapsed_seconds
gwp5.sspd



poisson_opts.update({'jacmethod': 'jvp'})
poisson_opts.update({'scale_goal': 1e3})
poisson_opts.update({'init_beta': 1e-6})
poisson_opts.update({'init_beta': 0.5})
poisson_opts.update({'init_beta': 1.0})


# %% geoweight: geoipopt
# geoipopt options
geoipopt_base = {
        'xlb': .1, 'xub': 10., # default 0.1, 10.0
         'crange': 0.0,  # default 0.0
         # 'print_level': 0,
        'file_print_level': 5,
        # 'scaling': True,
        # 'scale_goal': 1e3,
         # 'ccgoal': 10000,
         'addup': True,  # default is false
         'output_file': '/home/donboyd/Documents/test_sparse.out',
         'max_iter': 100,
         'linear_solver': 'ma86',  # ma27, ma77, ma57, ma86 work, not ma97
         'quiet': False}

geoipopt_opts = geoipopt_base.copy()
geoipopt_opts.update({'addup': False})
geoipopt_opts.update({'addup': True})
geoipopt_opts.update({'scaling': True})
geoipopt_opts.update({'scale_goal': 1e3})
geoipopt_opts.update({'crange': .04})
geoipopt_opts.update({'addup_range': .005})
geoipopt_opts.update({'xlb': .01})
geoipopt_opts.update({'xub': 10.0})
geoipopt_opts

gwip1 = prob.geoweight(method='geoipopt', options=geoipopt_opts)
gwip1.elapsed_seconds
gwip1.sspd

geoipopt_opts.update({'linear_solver': 'ma77'})
geoipopt_opts.update({'output_file': '/home/donboyd/Documents/test_sparse77.out'})
gwip1a = prob.geoweight(method='geoipopt', options=geoipopt_opts)
gwip1a.elapsed_seconds
gwip1a.sspd



# %% general qmatrix options
# uo = {'qmax_iter': 1, 'independent': True}
# uo = {'qmax_iter': 3, 'quiet': True, 'verbose': 2}


# %% geoweight: qmatrix lsq
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

gwqm_lsq = prob.geoweight(method='qmatrix-lsq', options=uolsq)
gwqm_lsq.elapsed_seconds
gwqm_lsq.sspd

# %% geoweight: qmatrix ipopt
# ipopt options
uoipopt = {'qmax_iter': 30,
           'quiet': True,
           'xlb': 0.001,
           'xub': 1000,
           'crange': .000001,
           'linear_solver': 'ma57'
           }

gwqm_ip = prob.geoweight(method='qmatrix-ipopt', options=uoipopt)
gwqm_ip.elapsed_seconds
gwqm_ip.sspd


# %% geoweight empcal
# empcal options
uoempcal = {'qmax_iter': 10, 'objective': 'ENTROPY'}
uoempcal = {'qmax_iter': 10, 'objective': 'QUADRATIC'}
gwqm_ec = prob.geoweight(method='qmatrix-ec', options=uoempcal)
gwqm_ec.elapsed_seconds
gwqm_ec.sspd


# %% geoweight raking
# raking options (there aren't really any)
uoqr = {'qmax_iter': 10}
gwqm_rake = prob.geoweight(method='qmatrix', options=uoqr)


# %% check geoweight results

gw = gwp4  # gwp1, ..., 
gw = gwip1  # gwip1, ...
gw = gwqm_lsq  # gwqm1, ...
gw = gwqm_ip
gw = gwqm_ec
gw = gwqm_rake

# general results
gw.elapsed_seconds

# compute weight sums vs totals
wtsums = gw.whs_opt.sum(axis=1)
wtdiffs = wtsums - p.wh
wtpdiffs = wtdiffs / p.wh * 100
np.round(np.quantile(wtpdiffs, qtiles), 2)

Q = np.divide(gw.whs_opt, p.wh.reshape(-1, 1))
np.round(np.quantile(Q.sum(axis=1), q=qtiles), 2)

# compute geotargets vs totals
geotargets_opt = np.dot(gw.whs_opt.T, p.xmat)
targdiffs = geotargets_opt - ngtargets
targpdiffs = targdiffs / ngtargets * 100
np.round(np.quantile(targpdiffs, qtiles), 2)
np.max(np.abs(targpdiffs))

# sspd
np.round(np.square(targpdiffs).sum(), 4)


# %% compare results from multiple methods
np.corrcoef(gwp4.whs_opt.flatten(), gwqm_lsq.whs_opt.flatten())
np.corrcoef(gwp4.whs_opt.flatten(), gwqm_rake.whs_opt.flatten())
np.corrcoef(gwp4.whs_opt.flatten(), gwqm_ec.whs_opt.flatten())



# %% END GEOWEIGHT START REWEIGHT



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


# %% reweight linear least squares
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


# bvls is very fast but can't use sparse matrices'
