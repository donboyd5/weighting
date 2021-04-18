
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

# The Hessian





# %% imports
import numpy as np
import scipy.sparse as sps
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
    
# %% geoweight class for ipopt
class GW:

    def __init__(self, cc, n, quiet):
        self.cc = cc  # is this making an unnecessary copy??
        self.jstruct = np.nonzero(cc.T)
        # consider sps.find as possibly faster than np.nonzero, not sure
        self.jnz = cc.T[self.jstruct]
        # self.jnz = sps.find(cc)[2]

        hidx = np.arange(0, n, dtype='int64')
        self.hstruct = (hidx, hidx)
        self.hnz = np.full(n, 2)

        self.quiet = quiet

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return np.sum((x - 1)**2)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return 2*x - 2

    def constraints(self, x):
        """Returns the constraints."""
        # np.dot(x, self.cc)  # dense calculation
        # self.cc.T.dot(x)  # sparse calculation
        return np.dot(x, self.cc)

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return self.jnz

    def jacobianstructure(self):
        """ Define sparse structure of Jacobian. """
        return self.jstruct        

    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        return obj_factor * self.hnz

    def hessianstructure(self):
        """ Define sparse structure of Hessian. """
        return self.hstruct

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        if(not self.quiet):
            print(f'{"":10} {iter_count:5d} {"":15} {obj_value:13.7e} {"":15} {inf_pr:13.7e}')



# %% test runs
p = mtp.Problem(h=40, s=3, k=2, pctzero=.4)
# p = mtp.Problem(h=40000, s=50, k=30, pctzero=.4)
# p = mtp.Problem(h=40000, s=50, k=30)
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
# p = mtp.Problem(h=10000, s=20, k=15)

xmat = p.xmat
wh = p.wh
geotargets = p.geotargets

h = xmat.shape[0]
k = xmat.shape[1]
s = geotargets.shape[0]

# matrix of initial state weight shares -- each row sums to 1
# Q = np.full((h, s), 1 / s) # each household's weight shares
# get initial state weights from first column of geotargets
qshares = geotargets[:, 0] / geotargets[:, 0].sum()

Q = np.empty((h, s))
Q[:, 0] = 0.2
Q[:, 1] = 0.5
Q[:, 2] = 0.3
Q
Q.flatten()

whs = np.multiply(Q, wh.reshape((-1, 1))) 
whs.flatten()

np.dot(whs.T, xmat)  # s x k matrix of calculated targets

whs.shape

cc = (xmat.T * wh).T   # dense multiplication
# cc = xmat.T.multiply(wh).T  # sparse multiplication
cc
cc.shape

# construct jacobian
jbase = cc.T
ijvalnz = sps.find(jbase)
row, col, nzvalue = sps.find(jbase)

# c2 = np.concatenate((col, col + h), axis=None)
# r2 = np.concatenate((row, row + k), axis=None)
# v2 = np.concatenate((nzvalue, nzvalue))
# xvals = np.full(h * 2, .33)
# np.concatenate([row, row + k, row +2 * k])



rows = np.array([])
cols = np.array([])
nzvalues = np.array([])
for state in np.arange(0, s):
    rows = np.concatenate([rows, row + k * state])
    cols = np.concatenate([cols, col + h * state])
    nzvalues = np.concatenate([nzvalues, nzvalue * Q[row, state]])

rows = rows.astype('int32')
cols = cols.astype('int32')
rows.shape
cols.shape
nzvalues.shape

rows
cols
# xvals = np.full(h * s, 1/s)
# nzvalues = np.tile(nzvalue, s)

jsparse = sps.csr_matrix((nzvalues, (rows, cols)))
print(jsparse)
jsparse.shape
xvals.shape

jbase
jdense = jsparse.todense()
jdense[0, ]
jdense[1, ]

x = np.ones(h * s)
geo_est = jsparse.dot(x) 
geotargets.flatten()
geo_est.flatten()

geotargets.sum()
geo_est.sum()











gw = ipopt_geo(p.wh, p.xmat, p.geotargets)
gw.elapsed_seconds
gw.whs_opt





# %%
