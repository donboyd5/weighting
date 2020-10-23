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
# needed for ipopt:
# from __future__ import print_function, unicode_literals

import numpy as np
import pandas as pd
from collections import namedtuple
from timeit import default_timer as timer
# import ipopt  # requires special installation

import src.utilities as ut
# import src.common as common
import src.geoweight_qmatrix as qm
import src.geoweight_poisson as ps
import src.reweight_ipopt as rwip
import src.reweight_empcalib as rwec
import src.reweight_raking as rwrk
import src.reweight_leastsquares as rwls
import src.reweight_minimizeNLP as rwmn

# import scipy.optimize as spo
# from scipy.optimize import least_squares


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

    def reweight(self,
                 method='ipopt',
                 options=None):
        if method == 'ipopt':
            method_result = rwip.rw_ipopt(
                self.wh, self.xmat, self.targets,
                options=options)
        elif method == 'empcal':
            method_result = rwec.gec(
                self.wh, self.xmat, self.targets,
                 options=options)
        elif method == 'rake':
            method_result =rwrk.rw_rake(
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
                  user_options=None,
                  solver_options=None,
                  Q=None,
                  drops=None,
                  maxiter=100):

        # input checks:
            # geotargets must by s x k

        if method == 'qmatrix':
            method_result = qm.qmatrix(self.wh, self.xmat, self.geotargets,
                                method='raking',
                                user_options=user_options,
                                solver_options=solver_options)
        elif method == 'qmatrix-ec':
            method_result = qm.qmatrix(self.wh, self.xmat, self.geotargets,
                                method='empcal',
                                user_options=user_options,
                                solver_options=solver_options)
        elif method == 'poisson':
            method_result = ps.poisson(self.wh, self.xmat, self.geotargets)

        # calculate sum of squared percentage differences
        diff = method_result.geotargets_opt - self.geotargets
        sspd = np.square(diff / self.geotargets * 100).sum()

        # here are the results we want for every method
        fields = ('method',
                  'elapsed_seconds',
                  'sspd',
                  'whs_opt',
                  'geotargets_opt',
                  'method_result')
        GeoResult = namedtuple('GeoResult', fields, defaults=(None,) * len(fields))

        geores = GeoResult(method=method,
                           elapsed_seconds=method_result.elapsed_seconds,
                           sspd=sspd,
                           whs_opt=method_result.whs_opt,
                           geotargets_opt=method_result.geotargets_opt,
                           method_result=method_result)

        return geores


    def help():
        print("\nThe microweight class requires the following arguments:",
              "\twh:\t\t\th-length vector of national weights for households",
              "\txmat:\t\th x k matrix of characteristices (data) for households",
              "\tgeotargets:\ts x k matrix of targets", sep='\n')
        print("\nThe goal of the method geoweight is to find state weights" +
              " that will",
              "hit the targets while ensuring that each household's state",
              "weights sum to its national weight.\n", sep='\n')
