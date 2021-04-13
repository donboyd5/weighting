# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 07:26:01 2020

@author: donbo
"""

import numpy as np
import geoweight as gw
import make_test_problems as mtp


# %% verify that this produces same results as R did for a test problem
p2 = mtp.rProblem()
g2 = gw.Geoweight(p2.wh, p2.xmat, p2.targets)
g2.xmat
g2.wh
g2.targets

g2.geosolve()
g2.result


# %% results from r problem - for checking against

# dw from get_dweights should be:
# 1.801604 1.635017 1.760851 1.365947 1.240773 1.325983

# delta when the beta matrix is 0 should be:
# 2.673062, 2.838026, 2.567032, 2.762710, 2.707713,
#     2.683379, 2.521871, 2.457231, 2.720219, 2.770873

# state weights when beta is 0 and we use the associated delta:
# > whs0
#           [,1]     [,2]     [,3]
#  [1,] 14.48426 14.48426 14.48426
#  [2,] 17.08202 17.08202 17.08202
#  [3,] 13.02710 13.02710 13.02710
#  [4,] 15.84272 15.84272 15.84272
#  [5,] 14.99494 14.99494 14.99494
#  [6,] 14.63447 14.63447 14.63447
#  [7,] 12.45187 12.45187 12.45187
#  [8,] 11.67245 11.67245 11.67245
#  [9,] 15.18365 15.18365 15.18365
# [10,] 15.97258 15.97258 15.97258

# targets when beta is 0
#          [,1]     [,2]
# [1,] 57.81941 76.40666
# [2,] 57.81941 76.40666
# [3,] 57.81941 76.40666

# sse_weighted 5.441764e-21

# $beta_opt_mat
#             [,1]        [,2]
# [1,] -0.02736588 -0.03547895
# [2,]  0.01679640  0.08806331
# [3,] -0.05385230  0.03097379

# $targets_calc
#          [,1]     [,2]
# [1,] 55.50609 73.20929
# [2,] 61.16143 80.59494
# [3,] 56.79071 75.41574

# $whs (optimal)
#           [,1]     [,2]     [,3]
#  [1,] 13.90740 15.09438 14.45099
#  [2,] 16.34579 18.13586 16.76441
#  [3,] 12.42963 13.97414 12.67753
#  [4,] 15.60913 16.07082 15.84823
#  [5,] 14.44566 15.85272 14.68645
#  [6,] 14.06745 15.51522 14.32073
#  [7,] 11.70919 13.28909 12.35734
#  [8,] 11.03794 12.39991 11.57950
#  [9,] 14.90122 15.59650 15.05323
# [10,] 15.72018 16.31167 15.88589

