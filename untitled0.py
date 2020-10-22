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

context = {**defaults, **user}

user_defaults = {
    'xlb': 0.1,
    'xub': 100,
    'crange': .02,
    'ccgoal': 1,
    'objgoal': 100,
    'quiet': True}

user_updates = {
    'abc': 10,
    'ccgoal': 100}

if user_updates['def'] == 10:
    print("ok")

if user_updates.get('def') == 10:
    print("ok")

del(user_updates)
if user_updates is None:
    print("no")

if 'user_updates' not in locals():
    print("no")


{**user_defaults, **user_updates}
from collections import namedtuple
MyTuple = namedtuple('MyTuple', sorted(d))
my_tuple = MyTuple(**d)

from collections import namedtuple
def dict_nt(d):
    # convert dict to named tuple
    return namedtuple('ntd', sorted(d))(**d)

dict_nt(user_defaults)


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


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

