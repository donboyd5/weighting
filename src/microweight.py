# -*- coding: utf-8 -*-
"""
microweight module

Classes:

    Microweight

Functions:

    abc

@author: donboyd5@gmail.com
"""

# TODO:
# options function
# least-squares reweighting
# weight from scratch
# fsolve geographic weighting


# %% imports

import importlib

import gc
import numpy as np
import pandas as pd
from collections import namedtuple
from timeit import default_timer as timer

# import scipy.optimize as spo
# from scipy.optimize import least_squares

import src.utilities as ut
# import src.common as common

import src.geoweight_ipopt as gwip

import src.geoweight_poisson_ipopt as gwp_ipopt
import src.geoweight_poisson_lsq as gwp_lsq
import src.geoweight_poisson_minimize_scipy as gwp_minsp
import src.geoweight_poisson_minimize_jax as gwp_minjax
import src.geoweight_poisson_minimize_tflowjax as gwp_mintfjax

import src.geoweight_poisson_newton as gwpn
# import src.geoweight_poisson_nelder as gwpneld

import src.geoweight_qmatrix as gwqm

import src.reweight_empcalib as rwec
import src.reweight_ipopt_dense as rwip
import src.reweight_ipopt_sparse as rwips
import src.reweight_leastsquares as rwls
import src.reweight_minimizeNLP as rwmn
import src.reweight_raking as rwrk


# %% reimports
importlib.reload(gwip)

importlib.reload(gwp_ipopt)
importlib.reload(gwp_lsq)
importlib.reload(gwp_minsp)
importlib.reload(gwp_minjax)
importlib.reload(gwp_mintfjax)
importlib.reload(gwpn)
importlib.reload(gwqm)

importlib.reload(rwec)
importlib.reload(rwip)
importlib.reload(rwips)
importlib.reload(rwls)
importlib.reload(rwmn)
importlib.reload(rwrk)


# %% Microweight class
class Microweight:
    """Class with data and methods for microdata weighting.

        Common terms and definitions:
        h: number of households (tax records, etc.)
        k: number of characteristics each household has (wages, pensions, etc.)
        s: number of states or geographic areas

        xmat: h x k matrix of characteristics for each household
        wh: 1 x h vector of national weights for households
        whs: h x s matrix of state weights for households (to be solved for)
            for each household, the sum of state weights must equal the
            total household weight

        beta: s x k matrix of poisson model coefficients
            (same for all households)
        delta: 1 x h vector of poisson model constants, 1 per household
            these values are uniquely determined by a given set of beta
            coefficients and the wh values


    """

    def __init__(self, wh, xmat, targets=None, geotargets=None):
        self.wh = wh
        self.xmat = xmat
        self.targets = targets
        self.geotargets = geotargets
        self.targets_init = np.dot(self.xmat.T, self.wh)
        if self.targets is not None:
            self.pdiff_init = self.targets_init / self.targets * 100 - 100

    def reweight(self,
                 method='ipopt',
                 options=None):
        gc.collect()  # just to be safe
        if method == 'ipopt':
            method_result = rwip.rw_ipopt(
                self.wh, self.xmat, self.targets,
                options=options)
        elif method == 'ipopt_sparse':
            method_result = rwips.rw_ipopt(
                self.wh, self.xmat, self.targets,
                options=options)
        elif method == 'empcal':
            method_result = rwec.gec(
                self.wh, self.xmat, self.targets,
                options=options)
        elif method == 'rake':
            method_result = rwrk.rw_rake(
                self.wh, self.xmat, self.targets,
                options=options)
        elif method == 'lsq':
            method_result = rwls.rw_lsq(
                self.wh, self.xmat, self.targets,
                options=options)
        elif method == 'minNLP':
            method_result = rwmn.rw_minNLP(
                self.wh, self.xmat, self.targets,
                options=options)

        # calculate sum of squared percentage differences
        diff = method_result.targets_opt - self.targets
        pdiff = diff / self.targets * 100
        sspd = np.square(pdiff).sum()

        # here are the results we want for every method
        fields = ('method',
                  'elapsed_seconds',
                  'sspd',
                  'wh_opt',
                  'targets_opt',
                  'pdiff',
                  'g',
                  'opts',
                  'method_result')
        ReweightResult = namedtuple('ReweightResult', fields, defaults=(None,) * len(fields))

        rwres = ReweightResult(method=method,
                               elapsed_seconds=method_result.elapsed_seconds,
                               sspd=sspd,
                               wh_opt=method_result.wh_opt,
                               targets_opt=method_result.targets_opt,
                               pdiff=pdiff,
                               g=method_result.g,
                               opts=method_result.opts,
                               method_result=method_result)

        return rwres

    def geoweight(self,
                  method='qmatrix',
                  options=None):

        # input checks:
        # geotargets must by s x k
        gc.collect()  # just to be safe
        print("method input: ", method)


        if method == 'qmatrix':
            method_result = gwqm.qmatrix(self.wh, self.xmat, self.geotargets,
                                       method='raking',
                                       options=options)
        elif method == 'qmatrix-ec':
            method_result = gwqm.qmatrix(self.wh, self.xmat, self.geotargets,
                                       method='empcal',
                                       options=options)
        elif method == 'qmatrix-ipopt':
            method_result = gwqm.qmatrix(self.wh, self.xmat, self.geotargets,
                                       method='ipopt',
                                       options=options)
        elif method == 'qmatrix-lsq':
            method_result = gwqm.qmatrix(self.wh, self.xmat, self.geotargets,
                                       method='least_squares',
                                       options=options)
        elif method == 'geoipopt':
            method_result = gwip.ipopt_geo(self.wh, self.xmat, self.geotargets,
                                          options=options)

        elif method == 'poisson-newton':
            method_result = gwpn.poisson(self.wh, self.xmat, self.geotargets,
                                         options=options)
        elif method == 'poisson-lsq':
            method_result = gwp_lsq.poisson(self.wh, self.xmat, self.geotargets,
                                         options=options)
        elif method == 'poisson-minscipy':
            method_result = gwp_minsp.poisson(self.wh, self.xmat, self.geotargets,
                                         options=options)
        elif method == 'poisson-minjax':
            method_result = gwp_minjax.poisson(self.wh, self.xmat, self.geotargets,
                                         options=options)
        elif method == 'poisson-mintfjax':
            method_result = gwp_mintfjax.poisson(self.wh, self.xmat, self.geotargets,
                                         options=options)
        elif method == 'poisson-ipopt':
            method_result = gwp_ipopt.poisson(self.wh, self.xmat, self.geotargets,
                                         options=options)

        # calculate sum of squared percentage differences
        diff = method_result.geotargets_opt - self.geotargets
        pdiff = diff / self.geotargets * 100
        sspd = np.square(pdiff).sum()

        # here are the results we want for every method
        fields = ('method',
                  'elapsed_seconds',
                  'sspd',
                  'pdiff',
                  'whs_opt',
                  'geotargets_opt',
                  'method_result')
        GeoResult = namedtuple('GeoResult', fields, defaults=(None,) * len(fields))

        geores = GeoResult(method=method,
                           elapsed_seconds=method_result.elapsed_seconds,
                           sspd=sspd,
                           pdiff=pdiff,
                           whs_opt=method_result.whs_opt,
                           geotargets_opt=method_result.geotargets_opt,
                           method_result=method_result)

        return geores
