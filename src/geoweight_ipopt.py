# -*- coding: utf-8 -*-

# %% notes

# documentation and examples
#   https://cyipopt.readthedocs.io/en/stable/
#   https://cyipopt.readthedocs.io/en/stable/tutorial.html#problem-interface
#   https://cyipopt.readthedocs.io/en/stable/reference.html

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
import src.utilities as ut

# import microweight as mw

# %% default options

user_defaults = {
    'xlb': 0.1,
    'xub': 100,
    'crange': .02,
    'ccgoal': False,
    'objgoal': 100,
    'addup': False,
    'quiet': True}

solver_defaults = {
    'print_level': 0,
    'file_print_level': 5,
    'jac_d_constant': 'yes',
    'hessian_constant': 'yes',
    'max_iter': 100,
    'mumps_mem_percent': 100,  # default 1000
    'linear_solver': 'ma57'
}

options_defaults = {**solver_defaults, **user_defaults}


# %% main function
def ipopt_geo(wh, xmat, geotargets,
        options=None):

    # inputs:
    #   wh: (h, )  national (i.e., total) weights for each household
    #   xmat:  (h, k)
    #   geotargets: (s, k)

    # outputs:
    #   whs (h, s)


    a = timer()

    # stop if xmat has any rows that are all zero (reconsider later)
    zero_rows = np.where(~xmat.any(axis=1))[0]
    if zero_rows.size > 0:
        print("found xmat rows that have all zeros, exiting...")
        print(zero_rows)
        return

    # update options with any user-supplied options
    if options is None:
        options_all = options_defaults.copy()
    else:
        options_all = options_defaults.copy()
        options_all.update(options)
        # options_all = {**options_defaults, **options}

    # convert dict to named tuple for ease of use
    opts = ut.dict_nt(options_all)

    h = xmat.shape[0]  # number of households
    k = xmat.shape[1]  # number of targets for each state
    s = geotargets.shape[0]  # number of states or geographic areas

    n = h * s # number of variables to solve for
    targs = k * s  # total number of target constraints

    # create Q, matrix of initial state weight shares -- each row sums to 1
    # for now get initial state weight shares from first column of geotargets
    state_shares = geotargets[:, 0] / geotargets[:, 0].sum()  # vector length s
    Q = np.tile(state_shares, (h, 1))  # h x s matrix
    # Q.sum(axis=1)

    whs_init = np.multiply(Q.T, wh).T

    targets = geotargets.flatten() # all targets for first state, then next, then ...
    targets_scaled = targets

    # construct the jacobian, which is constant (and highly sparse)
    # start with constraint coefficients for national weights
    cc = (xmat.T * wh).T   # dense multiplication because this is not large
    # np.multiply(xmat.T, wh).T  # same result
    # cc = xmat.T.multiply(wh).T  # sparse multiplication
    # cc is h x k, so cc.T is k x h

    # scale constraint coefficients and targets
    # ccscale = 1
    # if opts.ccgoal is not False:
    #     ccscale = get_ccscale(cc, ccgoal=opts.ccgoal, method='mean')
        
    # # ccscale = 1
    # cc = cc * ccscale  # mult by scale to have avg derivative meet our goal

    # targets_scaled = targets * ccscale  # djb do I need to copy?

    # construct row and column indexes, and values of the jacobian
    row, col, nzvalue = sps.find(cc.T)
    rows = np.array([])
    cols = np.array([])
    nzvalues = np.array([])

    for state in np.arange(0, s):
        rows = np.concatenate([rows, row + k * state]) # constraints
        cols = np.concatenate([cols, col + h * state])  # households
        nzvalues = np.concatenate([nzvalues, nzvalue * Q[col, state]])

    rows = rows.astype('int32')
    cols = cols.astype('int32')
    jsparse_targets = sps.csr_matrix((nzvalues, (rows, cols)))
    

    # adding up constraints, if needed
    jsparse_addup = None
    if opts.addup:
        # define row indexes
        row = np.repeat(np.arange(0, h), s)  # row we want from whs_init

        # define column indexes
        state_idx = np.tile(np.arange(0, s), h) # which state do we want
        col = row + state_idx * h

        # use init whs values for the coefficients
        nzvalues = whs_init[row, state_idx]
        jsparse_addup = sps.csr_matrix((nzvalues, (row, col)))

    # return jsparse_targets, jsparse_addup
    jsparse = sps.vstack([jsparse_targets, jsparse_addup])
    
    m = jsparse.shape[0]  # TOTAL number of constraints - targets plus adding-up

    x0 = np.ones(n)
    lb = np.full(n, opts.xlb)
    ub = np.full(n, opts.xub)    

    # constraint lower and upper bounds
    cl_targets = targets_scaled - abs(targets_scaled) * opts.crange
    cu_targets = targets_scaled + abs(targets_scaled) * opts.crange

    cl = cl_targets
    cu = cu_targets
    if opts.addup:
        whlow = wh * .99
        whhigh = wh * 1.01
        cl = np.concatenate((cl_targets, whlow), axis=0)
        cu = np.concatenate((cu_targets, whhigh), axis=0)

    nlp = cy.Problem(
        n=n,
        m=m,
        problem_obj=GW(jsparse, n, opts.quiet),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu)

    user_keys = user_defaults.keys()
    solver_options = {key: value for key, value in options_all.items() if key not in user_keys}

    for option, value in solver_options.items():
        nlp.add_option(option, value)      

    if(not opts.quiet):
        print(f'\n {"":10} Iter {"":25} obj {"":22} infeas')  

    g, ipopt_info = nlp.solve(x0)

    Q_best = np.multiply(Q, g.reshape(s, h).T)
    # Q_best.sum(axis=1)

    whs_opt = np.multiply(Q_best.T, wh).T


    # next 2 lines produce the same result, in about the same time
    # %timeit whs_opt = np.multiply(Q_best, wh.reshape((-1, 1)))  # same result
    # %timeit whs_opt = (Q_best.T * wh).T  # same result
    # (whs_opt[:, 0] * xmat[:, 0]).sum()

    geotargets_opt = np.dot(whs_opt.T, xmat)
    geotargets_init = np.dot(whs_init.T, xmat)

    b = timer()


    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_init',
              'whs_opt',
              'geotargets',
              'geotargets_opt',
              'geotargets_init',
              'g',
              'Q_best',
              'opts',
              'ipopt_info')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 whs_opt=whs_opt,
                 whs_init=whs_init,
                 geotargets = geotargets,
                 geotargets_opt=geotargets_opt,
                 geotargets_init=geotargets_init,
                 g=g,
                 Q_best=Q_best,
                 opts=opts,
                 ipopt_info=ipopt_info)
    return res
    
# %% geoweight class for ipopt
class GW:

    def __init__(self, jsparse, n, quiet):
        self.jsparse = jsparse # is this making an unnecessary copy??
        rows, cols, self.jnz = sps.find(jsparse)
        self.jstruct = (rows, cols)

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
        return self.jsparse.dot(x)

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


# %% functions available outside of the RW class

def get_ccscale(jsparse_constraints, ccgoal, method='mean'):
    """
    Create multiplicative scaling vector ccscale.

    cc is the constraint coefficients matrix (h x k)
    ccgoal is what we would like the typical scaled coefficient to be
      (maybe around 100, for example)

    For scaling the constraint coefficients and the targets.
    Returns the ccscale vector.

    # use mean or median of absolute values of coeffs as the denominator
    # denom is a k-length vector (1 per target)

    """

    if(method == 'mean'):
        denom = np.abs(cc).sum(axis=0) / cc.shape[0]
    elif(method == 'median'):
        denom = np.median(np.abs(cc), axis=0)

    # it is hard to imagine how denom ever could be zero but just in
    # case, set it to 1 in that case
    denom[denom == 0] = 1

    ccscale = np.absolute(ccgoal / denom)
    # ccscale = ccscale / ccscale
    return ccscale



    if(method == 'mean'):
        denom = np.abs(cc).sum(axis=0) / cc.shape[0]
    elif(method == 'median'):
        denom = np.median(np.abs(cc), axis=0)

    # it is hard to imagine how denom ever could be zero but just in
    # case, set it to 1 in that case
    denom[denom == 0] = 1

    ccscale = np.absolute(ccgoal / denom)
    # ccscale = ccscale / ccscale
    return ccscale




