# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 06:20:10 2020

@author: donbo
"""

import warnings
import numpy as np

# wh = p.wh
# xmat = p.xmat
# targets = p.targets * (1 + noise)
# q = 1


# %% primary function
def rake(wh, xmat, targets, q=1, objective=None, max_iter=10):
    # this is a direct translation of the raking code of the calib function
    # in the R sampling package, as of 10/3/2020
    # wh vector of initial weights (d in R calib)
    # xmat the matrix of covariates (Xs in R calib)
    # targets vector of targets (targets in R calib)
    # q vector or scalar related to heteroskedasticity
    # returns g, which when multiplied by the initial wh gives the new weight

    # wh_adjusted (w1 in R calib)
    # targets_calc (tr in R calib)

    EPS = 1e-15  # machine double precision used in R
    EPS1 = 1e-8  # R calib uses 1e-6
    # max_iter = 10  # 10 in R calib

    # make sure inputs all have the right shape
    wh = wh.reshape((-1, 1))  # h rows, 1 column
    targets = targets.reshape((-1, 1))  # k rows, 1 column

    lam = np.zeros((xmat.shape[1], 1))  # lam is k x 1
    wh_adjusted = wh * np.exp(np.dot(xmat, lam) * q) # h(n) rows x 1 column

    # set initial value for g (djb addition to program)
    g = np.ones(wh.size)  # h dimension, no cols

    for i in range(max_iter):
        # phi is calc targets minus targets
        phi = np.dot(xmat.T, wh_adjusted) - targets  # phi is 1 col matrix
        # T1 has k (i.e., m -- number of targets) rows and
        # h (i.e., n -- # households) columns
        T1 = (xmat * wh_adjusted).T  # T1 is k x h
        phiprim = np.dot(T1, xmat)  # phiprim is k x k
        lam = lam - np.dot(np.linalg.pinv(phiprim, rcond=1e-15), phi)  # k x 1
        # wh_adjusted -- h (i.e., n) x 1; in R this is a vector??
        wh_adjusted = wh * np.exp(np.dot(xmat, lam) * q)
        if np.isnan(wh_adjusted).any() or np.isinf(wh_adjusted).any():
            warnings.warn("No convergence bad w1")
            g = None
            break
        # targets_calc = np.inner(xmat.T, wh_adjusted.T) # k x 1 # note wh_adjusted.T
        targets_calc = np.dot(xmat.T, wh_adjusted) # k x 1, slightly faster
        if np.max(np.abs(targets_calc - targets) / targets) < EPS1:
            break
        if i == max_iter:
            warnings.warn("No convergence after max iterations")
            g = None
        else:
            g = wh_adjusted / wh  # djb: what if wh has zeros? h x 1
        # djb temporary solution: force g to be float
        # TODO: explore where we have numerical problems and
        # fix them
        g = np.array(g, dtype=float)  # djb
        g = g.reshape((-1, ))
        # end of the for loop

    return g


# def rake_bak(Xs, d, total, q=1, objective=None, max_iter=10):
#     # my first transation (djb), before changing variable names
#     # this is a direct translation of the raking code of the calib function
#     # in the R sampling package, as of 10/3/2020
#     # Xs the matrix of covariates
#     # d vector of initial weights
#     # total vector of targets
#     # q vector or scalar related to heteroskedasticity
#     # returns g, which when multiplied by the initial d gives the new weight
#     EPS = 1e-15  # machine double precision used in R
#     EPS1 = 1e-8  # R calib uses 1e-6
#     # max_iter = 10

#     # make sure inputs all have the right shape
#     d = d.reshape((-1, 1))
#     total = total.reshape((-1, 1))

#     lam = np.zeros((Xs.shape[1], 1))  # lam is k x 1
#     w1 = d * np.exp(np.dot(Xs, lam) * q) # h(n) x 1

#     # set initial value for g (djb addition to program)
#     g = np.ones(w1.size)

#     for i in range(max_iter):
#         phi = np.dot(Xs.T, w1) - total  # phi is 1 col matrix
#         T1 = (Xs * w1).T # T1 has k(m) rows and h(n) columns
#         phiprim = np.dot(T1, Xs) # phiprim is k x k
#         lam = lam - np.dot(np.linalg.pinv(phiprim, rcond = 1e-15), phi) # k x 1
#         w1 = d * np.exp(np.dot(Xs, lam) * q)  # h(n) x 1; in R this is a vector??
#         if np.isnan(w1).any() or np.isinf(w1).any():
#             warnings.warn("No convergence bad w1")
#             g = None
#             break
#         tr = np.inner(Xs.T, w1.T) # k x 1
#         if np.max(np.abs(tr - total) / total) < EPS1:
#             break
#         if i == max_iter:
#             warnings.warn("No convergence after max iterations")
#             g = None
#         else:
#             g = w1 / d  # djb: what if d has zeros?
#         # djb temporary solution: force g to be float
#         # TODO: explore where we have numerical problems and
#         # fix them
#         g = np.array(g, dtype=float)  # djb
#         g = g.reshape((-1, ))
#         # end of the for loop

#     return g

