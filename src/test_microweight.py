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
def targs(x, wh, xmat):
    return np.dot(xmat.T, wh * x)


def sspd(x, wh, xmat, targets):
    #  sum of squared percentage differences
    diffs = np.dot(xmat.T, wh * x) - targets
    pdiffs = diffs / targets * 100
    return np.square(pdiffs).sum()


# %% make problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=40, s=2, k=3)
p = mtp.Problem(h=1000, s=10, k=5)
p = mtp.Problem(h=10000, s=10, k=10)
p = mtp.Problem(h=40000, s=10, k=30)

np.random.seed(1)
noise = np.random.normal(0, .0125, p.k)
noise
ntargets = p.targets * (1 + noise)

# ntargets = p.targets

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets, geotargets=p.geotargets)


# %% reweight the problem
uo = {'crange': 0.001, 'quiet': False}
rw1a = prob.reweight(method='ipopt', user_options=uo)
# dir(rw1a)
rw1a.sspd

uo = {'crange': 0.0001, 'xlb': 0, 'xub':1e5, 'quiet': False}
rw1b = prob.reweight(method='ipopt', user_options=uo)
rw1b.sspd

so = {'increment': .00001}
# so = {'increment': .00001, 'autoscale': False}
rw2 = prob.reweight(method='empcal', solver_options=so)
rw2 = prob.reweight(method='empcal')
rw2.sspd

rw3 = prob.reweight(method='rake')
rw3 = prob.reweight(method='rake', user_options={'maxiter': 20})
rw3.sspd

ntargets
rw1a.targets_opt
rw1b.targets_opt
rw2.targets_opt
rw3.targets_opt

# time
rw1a.elapsed_seconds
rw1b.elapsed_seconds
rw2.elapsed_seconds
rw3.elapsed_seconds

# sum of squared percentage differences
rw1a.sspd
rw1b.sspd
rw2.sspd
rw3.sspd

# distribution of g values
np.quantile(rw1a.g, qtiles)
np.quantile(rw1b.g, qtiles)
np.quantile(rw2.g, qtiles)
np.quantile(rw3.g, qtiles)

rw1a.g.sum()
rw1b.g.sum()
rw2.g.sum()
rw3.g.sum()



# %% geoweight the problem
gw1 = prob.geoweight(method='qmatrix')
# dir(gw1)
gw1.method
gw1.elapsed_seconds
gw1.geotargets_opt
gw1.whs_opt
dir(gw1.method_result)

gw2 = prob.geoweight(method='qmatrix-ec')
gw2.method
gw2.geotargets_opt

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



