

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
opt_base = {'xlb': .1, 'xub': 10,
         'crange': 0.0,
         'print_level': 0,
         'file_print_level': 5,
         # 'ccgoal': 10000,
         'max_iter': 100,
         'linear_solver': 'ma57',  # ma27, ma77, ma57, ma86 work, not ma97
         'quiet': False}

# %% test sparse version

p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=1000, s=5, k=4, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=10000, s=10, k=8, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=20000, s=20, k=15, xsd=.1, ssd=.5, pctzero=.4)
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5, pctzero=.4)

geotargets = p.geotargets

# %% optionally add noise
np.random.seed(1)
noise = np.random.normal(0, .02, p.geotargets.size)
# np.round(noise * 100, 2)

geotargets = p.geotargets * (1 + noise.reshape((p.s, p.k)))

# %% run problem
opt_sparse = opt_base.copy()
opt_sparse.update({'output_file': '/home/donboyd/Documents/test_sparse.out'})
res = gwi.ipopt_geo(p.wh, p.xmat, geotargets, options=opt_sparse)
res.elapsed_seconds
res.ipopt_info['status_msg']

p.geotargets
res.geotargets
# geotargets
res.geotargets_opt
res.ipopt_info['g']
pdiff_init = res.geotargets_init / geotargets * 100 - 100
pdiff_opt = res.geotargets_opt / geotargets * 100 - 100
np.round(pdiff_init, 2)
np.round(pdiff_opt, 2)

geotargets_init
geotargets_init

res.g
np.quantile(res.g*100, q=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])

res.ipopt_info
res.ipopt_info['g']  # constraints at optimal solution
res.Q_best
checksums = res.Q_best.sum(axis=1)
np.quantile(checksums*100, q=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])
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

