
# %% notes
#   h number of households
#   k number of targets for a single state
#   s number of states
#   wh h x 1 vector of total household weights
#   whs_opt (TBD) h x s matrix of optimal household weights by state
#   whs_init h x s matrix of initial household weights by state
#   xmat h x k vector of characteristics
#   x (TBD) vector of length (h x s) that, when multiplied by whs_init yields whs_opt

# For example, if we have 
#      h=40,000 households
#      s=50 states, and 
#      k=30 targets per state, then
#  The vector of x variables (TBD) has length 40e3 x 50 = 2e6, or 2 million

# The constraint Jacobian will be huge but highly sparse:
#   2 million columns (1 for each x variable - i.e., h x s)
#   Two kinds of rows - rows for target constraints and rows for adding-up constraints
#     2,000 rows for target constraints (s x k rows -- a set of targets for each state)
#    40,000 rows for adding-up constraints - one constraint for each household, where the
#           sum of each household's state weights must add to the household's total weight
#   Thus in this example the Jacobian will have 84 billion elements: 
#     2 million columns x 42,000 rows

# But the Jacobian will be very sparse. The maximum number of nonzero elements will be:
#   (a) For the target constraints:
#       The columns for a given state will only affect the constraint rows for that state. Thus
#       each set of state columns (h=40e3 columns for a state) will influence k=30 targets.
#       Therefore each state will have a maximum of 1.2 million nonzero constraint coefficients 
#       (40e3 x 30). Since s=50, there are a maximum of 60 million nonzero constraint coefficients
#       for the targets.
#   (b) For the adding-up constraints:
#       Each household will have 1 constraint per state so there will be a maximum of 
#       40e3 x 50 = 2 million nonzero constraint coefficients for the adding-up constraints,
#   For a total maxmum of 62 million nonzero constraint coefficients.






# %% imports
import numpy as np
from timeit import default_timer as timer
from collections import namedtuple

import cyipopt as cy
import make_test_problems as mtp


# %% main function
def ipopt_geo(wh, xmat, geotargets,
        options=None):

    # inputs:
    #   wh: (h, )
    #   xmat:  (h, k)
    #   geotargets: (s, k)

    # outputs:
    #   whs (h, s)


    a = timer()

    whs_opt = None
    geotargets_opt = geotargets

    b = timer()


    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets',
              'geotargets_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 whs_opt=whs_opt,
                 geotargets = geotargets,
                 geotargets_opt=geotargets_opt)
    return res
    

# %% test runs
p = mtp.Problem(h=40, s=3, k=2)
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
# p = mtp.Problem(h=10000, s=20, k=15)

# n = p.xmat.shape[0]
# m = p.targets.shape[0]

gw = ipopt_geo(p.wh, p.xmat, p.geotargets)
gw.elapsed_seconds
gw.whs_opt




