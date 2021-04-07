#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 07:54:35 2021

@author: donboyd
"""

# %% imports
# for checking:
# import sys; print(sys.executable)
# print(sys.path)

import numpy as np

import scipy
from scipy.optimize import lsq_linear
from numpy.random import seed
from timeit import default_timer as timer

import src.make_test_problems as mtp
import src.microweight as mw

from __future__ import print_function, unicode_literals
from collections import namedtuple

import cyipopt
import src.reweight_ipopt as rwip
import src.microweight as mw

import src.utilities as ut

# import src.poisson as ps

# print(mw.__doc__)

# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% functions
def targs(targvec, div=50, seed=seed(1234)):
    r = np.random.randn(targvec.size) / 50  # random normal
    targets = (targvec * (1 + r)).flatten()
    return targets


# %% make problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=40, s=1, k=3)
p = mtp.Problem(h=1000, s=1, k=10)
p = mtp.Problem(h=10000, s=1, k=30)
p = mtp.Problem(h=20000, s=1, k=30)
p = mtp.Problem(h=200000, s=1, k=30)


# %% add noise to targets
np.random.seed(1)
targs(p.targets)
noise = np.random.normal(0, .03, p.k)
noise
ntargets = p.targets * (1 + noise)


# %% default options

user_defaults = {
    'xlb': 0.1,
    'xub': 100,
    'crange': .02,
    'ccgoal': 1,
    'objgoal': 100,
    'quiet': True}

solver_defaults = {
    'print_level': 0,
    'file_print_level': 5,
    'jac_d_constant': 'yes',
    'hessian_constant': 'yes',
    'max_iter': 100,
    'mumps_mem_percent': 100,  # default 1000
    'linear_solver': 'ma57'
}

options_defaults = {**solver_defaults, **user_defaults}


# %% get problem info
# run ipopt by hand
# rwip.rw_ipopt(self.wh, self.xmat, self.targets, options=options)
# get variables from p

# set options using reweight_ipopt

xmat = p.xmat
wh = p.wh
targets = ntargets


# %% next steps
n = xmat.shape[0]
m = xmat.shape[1]

# update options with any user-supplied options
options_all = options_defaults
options_all = options_defaults.copy()

# convert dict to named tuple for ease of use
opts = ut.dict_nt(options_all)

# constraint coefficients (constant)
cc = (xmat.T * wh).T
cc.shape

# scale constraint coefficients and targets
# ccscale = get_ccscale(cc, ccgoal=opts.ccgoal, method='mean')
ccscale = 1
cc = cc * ccscale  # mult by scale to have avg derivative meet our goal
targets_scaled = targets * ccscale  # djb do I need to copy?

callbacks = rwip.Reweight_callbacks(cc, opts.quiet)

# x vector starting values, and lower and upper bounds
x0 = np.ones(n)
lb = np.full(n, opts.xlb)
ub = np.full(n, opts.xub)

# constraint lower and upper bounds
cl = targets_scaled - abs(targets_scaled) * opts.crange
cu = targets_scaled + abs(targets_scaled) * opts.crange

nlp = cyipopt.Problem(
    n=n,
    m=m,
    problem_obj=callbacks,
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu)

# solve the problem
g, ipopt_info = nlp.solve(x0)
dir(ipopt_info)

wh_opt = g * wh

wh
wh_opt

targets_opt = np.dot(xmat.T, wh_opt)

targets
targets_opt

ntargets

np.square((targets_opt - targets) / targets * 100).sum()


# %% reweight the problem
prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets)
opts = {'xlb': 1e-4, 'xub': 1e4,
        'crange': 0.001,
        'print_level': 0,
        'file_print_level': 5,
        'derivative_test': 'first-order',
        'objgoal': 1, 'ccgoal': 1,
        'max_iter': 10,
        'linear_solver': 'ma27', 'quiet': False}
opts
# opts = {'crange': 0.001, 'xlb':0, 'xub':100, 'quiet': False}
rw1 = prob.reweight(method='ipopt', options=opts)
# dir(rw1)
init_vals = np.dot(p.xmat.T, p.wh)
rw1.elapsed_seconds
rw1.sspd
np.round(rw1.pdiff, 2)
np.round((init_vals - ntargets) / ntargets * 100, 2)

np.square((rw1.targets_opt - ntargets) / ntargets * 100).sum()
