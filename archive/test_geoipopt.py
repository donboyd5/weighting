

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

# %% constants
qtiles = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]

# %% base options
opt_base = {# 'xlb': .2, 'xub': 2, # default 0.1, 10.0
         # 'crange': 0.0,  # default 0.0
         # 'print_level': 0,
         # 'file_print_level': 5,
         # 'ccgoal': 10000,
         # 'addup': False,  # default is false
         'max_iter': 100,
         'linear_solver': 'ma57',  # ma27, ma77, ma57, ma86 work, not ma97
         'quiet': False}

# %% create problem
p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=100, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=3, k=3, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=10, k=8, xsd=.1, ssd=.5, pctzero=.2)
p = mtp.Problem(h=20000, s=20, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.5)
p = mtp.Problem(h=50000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.2)

geotargets = p.geotargets

# any zero rows?
# p.xmat
p.h
p.xmat
p.xmat.size
np.count_nonzero(p.xmat) * p.s


# %% add noise
np.random.seed(1)
noise = np.random.normal(0, .02, p.geotargets.size)
# np.round(noise * 100, 2)

geotargets = p.geotargets * (1 + noise.reshape((p.s, p.k)))


# %% update options
opt_sparse = opt_base.copy()
opt_sparse.update({'output_file': '/home/donboyd/Documents/test_sparse.out'})
opt_sparse.update({'addup': False})
opt_sparse.update({'addup': True})
opt_sparse.update({'crange': .01})
opt_sparse.update({'addup_range': .0})
opt_sparse.update({'linear_solver': 'ma86'})
opt_sparse.update({'xlb': .01})
opt_sparse.update({'xub': 10.0})
opt_sparse


# %% run problem
res = gwi.ipopt_geo(p.wh, p.xmat, geotargets, options=opt_sparse)
res.elapsed_seconds
res.elapsed_seconds / 60
res.ipopt_info['status_msg']
qsums = res.Q_best.sum(axis=1)
np.quantile(qsums*100, q=qtiles)

res.geotargets
res.geotargets_opt
# res.ipopt_info['g']
pdiff_init = res.geotargets_init / geotargets * 100 - 100
pdiff_opt = res.geotargets_opt / geotargets * 100 - 100
np.round(pdiff_init, 2)
np.round(pdiff_opt, 2)

np.quantile(pdiff_init, q=qtiles)
np.quantile(pdiff_opt, q=qtiles)

sspd = np.square(pdiff_opt).sum()
sspd

np.round(np.quantile(res.g, q=qtiles), 2)

res.ipopt_info
res.whs_opt

np.round(np.quantile(res.whs_opt, q=qtiles), 2)

# targ_opt = np.dot(p.whs.T, p.xmat)


# %% compare to poisson

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, geotargets=geotargets)
# uo = {'qmax_iter': 3, 'quiet': True, 'verbose': 2}
gw5 = prob.geoweight(method='poisson')  # , options=uo
gw5.elapsed_seconds
gw5.sspd
dir(gw5)
gw5.pdiff
np.round(np.quantile(gw5.pdiff, q=qtiles), 2)
gw5.whs_opt

np.round(np.quantile(gw5.whs_opt, q=qtiles), 2)


# %%  play
qtiles
