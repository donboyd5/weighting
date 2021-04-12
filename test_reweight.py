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
from __future__ import print_function, unicode_literals
# import importlib
import numpy as np

import scipy
from scipy.optimize import lsq_linear
from numpy.random import seed
from timeit import default_timer as timer

import cyipopt  # so we can access ipopt directly

import src.make_test_problems as mtp
import src.microweight as mw

import src.reweight_ipopt as rwip  # to access reweight directly

from collections import namedtuple

import src.utilities as ut

# importlib.reload(src.reweight_ipopt)  # ensure reload upon changes
# importlib.reload(mw)
# import src.poisson as ps
# print(mw.__doc__)


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% functions
def targs(targvec, div=50, seed=seed(1234)):
    r = np.random.randn(targvec.size) / 50  # random normal
    targets = (targvec * (1 + r)).flatten()
    return targets


def f(g):
    return np.round(np.quantile(g, qtiles), 4)


# %% make problem of a desired size
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
# p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=40, s=1, k=3)
# p = mtp.Problem(h=1000, s=1, k=10)
p = mtp.Problem(h=10000, s=1, k=30)
# p = mtp.Problem(h=20000, s=1, k=30)
# p = mtp.Problem(h=100000, s=1, k=50)
# p = mtp.Problem(h=200000, s=1, k=30)
# p = mtp.Problem(h=500000, s=1, k=100)


# %% investigate sparse matrices]
A = p.xmat



# %% add noise to targets
np.random.seed(1)
targs(p.targets)
noise = np.random.normal(0, .02, p.k)
noise * 100
ntargets = p.targets * (1 + noise)
init_targs = np.dot(p.xmat.T, p.wh)

init_pdiff = (init_targs - ntargets) / ntargets * 100
# equivalently: 1 / (1 + noise) * 100 - 100

init_sspd = np.square(init_pdiff).sum()


# %% create problem object
p.h
p.k
prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets)

# %% define default options

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
    'linear_solver': 'ma86'
}

options_defaults = {**solver_defaults, **user_defaults}


# %% reweight with ipopt

optip = {'xlb': .1, 'xub': 10,
         'crange': 0.02,
         'print_level': 0,
         'file_print_level': 5,
         # 'derivative_test': 'first-order',
         'ccgoal': 100,
         'objgoal': 1,
         'max_iter': 1000,
         'linear_solver': 'ma57',  # ma27, ma77, ma57, ma86 work, not ma97
         # 'ma97_order': 'metis',
         # 'mumps_mem_percent': 100,  # default 1000
         'quiet': False}


# using coinhsl-2019.05.21
# # checking: coinhsl-2015.06.23
# ma57 gives:
#  Input Error: Incorrect initial partitioning scheme.

# ma97 repeatedly gives:
#  Intel MKL ERROR: Parameter 4 was incorrect on entry to DGEMM
# Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)


opts = {'crange': 0.001, 'xlb': 0, 'xub': 100, 'quiet': False}
rw1 = prob.reweight(method='ipopt', options=optip)
# dir(rw1)
rw1.elapsed_seconds


rw1.sspd


np.round(init_pdiff, 2)
np.round(rw1.pdiff, 2)

qtiles
f(rw1.g)


# %% reweight with lsq method

optlsq = {
    'xlb': 0.1,
    'xub': 10,
    # bvls or trf; trf seems more robust
    # bvls does not allow sparse matrices
    # so trf seems better choice in general
    'method': 'trf',
    'tol': 1e-6,  # 1e-6
    'lsmr_tol': 'auto',  # 'auto',  # 'auto',  # None
    'max_iter': 50,
    'verbose': 2,
    'scaling': True}

rw2 = prob.reweight(method='lsq', options=optlsq)
rw2.elapsed_seconds
rw2.sspd
f(rw2.g)

np.round(init_pdiff, 2)
np.round(rw2.pdiff, 2)


# %% reweight with empcal method
rw3 = prob.reweight(method='empcal')
rw3.sspd
f(rw3.g)

# %% reweight with rake method
rw4 = prob.reweight(method='rake')
rw4.sspd
f(rw4.g)

rw5 = prob.reweight(method='minNLP')
rw5.sspd
f(rw5.g)


# %% start of section to run ipopt manually
# rwip.rw_ipopt(self.wh, self.xmat, self.targets, options=options)
# get variables from p

# set options using reweight_ipopt

xmat = p.xmat
wh = p.wh
targets = ntargets


# %% setup for ipopt
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
