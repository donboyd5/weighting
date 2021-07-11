
import numpy as np

methods = ('jac', 'krylov', 'jvp')

methods[1]

from itertools import cycle

lst = ['a', 'b', 'c']

pool = cycle(lst)
next(pool)


pool = cycle(methods)
next(pool)

a = 0
b = 1
a, b += 1

def fa(x):
    return x * 3

def fb(x):
    return x**2

def fc(x):
    return x**3

x = 4
fa(x)
fb(x)
fc(x)

fns = {'fa': fa,
       'fb': fb,
       'fc': fc}

fns['fa'](x)

check = 6

goal_met = 10 <= check
goal_met


a = 'abc'
b = 'def'
a += b
a

meth = ('a', 'b', 'c')

if 'a' in meth: print("yes")
if 'z' in meth: print("yes")









# for item in pool:
#     print item,

