# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:55:34 2020

@author: donbo
"""

import numpy as np
import src.make_test_problems as mtp

p = mtp.Problem(h=40, s=2, k=3)

wh = p.wh.copy().reshape((-1, 1))
wh.shape

np.dot(p.xmat.T, wh)
np.inner(p.xmat.T, wh.T)

%timeit np.dot(p.xmat.T, wh)
%timeit np.inner(p.xmat.T, wh.T)


(p.xmat * p.wh).T


class Result:
    def __init__(self, elapsed_seconds):
        self.elapsed_seconds = elapsed_seconds

res1 = Result(elapsed_seconds=10 - 3)

res1.elapsed_seconds = 2
res1.elapsed_seconds
res1.color = "green"
res1.color

res2 = Result()

from collections import namedtuple

fields = ('elapsed_seconds',
                        'Q_opt',
                        'whs_opt',
                        'geotargets_opt',
                        'pctdiff',
                        'iter_opt')
Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

a = Result()
a
dir(a)
a.iter_opt = 10

Res2 = namedtuple()

fields = ('elapsed_seconds', 'x', 'other')
Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

a = Result(10, (2, 4, 6))
a = Result(10, (2, 4, 6), other = {'a': 1, 'b':2, 'c':[3, 5]})
a

