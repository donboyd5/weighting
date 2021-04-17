"""
Reweight class
"""

# %% imports

import os

import numpy as np
import scipy.sparse as sps

from timeit import default_timer as timer
from collections import namedtuple

import cyipopt as cy

import src.utilities as ut


# %% default options

user_defaults = {
    'xlb': 0.1,
    'xub': 100,
    'crange': .02,
    'ccgoal': 1,
    'objgoal': 100,
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



# %% rw_ipopt - the primary function
def rw_ipopt(wh, xmat, targets,
             options=None):
    r"""
    Build and solve the reweighting NLP.

    Good general settings seem to be:
        get_ccscale - use ccgoal=1, method='mean'
        get_objscale - use xbase=1.2, objgoal=100
        no other options set, besides obvious ones

    """

    a = timer()
    # xmat = sps.csr_matrix(xmat) # make sparse version right away; csr is fastest for matrix-vector dot products
    n = xmat.shape[0]
    m = xmat.shape[1]

    # update options with any user-supplied options
    if options is None:
        options_all = options_defaults.copy()
    else:
        options_all = options_defaults.copy()
        options_all.update(options)
        # options_all = {**options_defaults, **options}

    # convert dict to named tuple for ease of use
    opts = ut.dict_nt(options_all)

    # constraint coefficients (constant)
    cc = (xmat.T * wh).T   # dense multiplication
    # cc = xmat.T.multiply(wh).T  # sparse multiplication

    # scale constraint coefficients and targets
    # ccscale = get_ccscale(cc, ccgoal=opts.ccgoal, method='mean')
    ccscale = 1
    cc = cc * ccscale  # mult by scale to have avg derivative meet our goal
    targets_scaled = targets * ccscale  # djb do I need to copy?
    
    # x vector starting values, and lower and upper bounds
    x0 = np.ones(n)
    lb = np.full(n, opts.xlb)
    ub = np.full(n, opts.xub)

    # constraint lower and upper bounds
    cl = targets_scaled - abs(targets_scaled) * opts.crange
    cu = targets_scaled + abs(targets_scaled) * opts.crange

    nlp = cy.Problem(
        n=n,
        m=m,
        problem_obj=RW(cc, n, opts.quiet),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu)

    # objective function scaling - add to options dict
    # djb should I pass n and callbacks???
    # objscale = get_objscale(objgoal=opts.objgoal,
    #                         xbase=1.2,
    #                         n=n,
    #                         callbacks=callbacks)
    # options_all['obj_scaling_factor'] = objscale

    # create a dict that only has solver options, for passing to ipopt
    user_keys = user_defaults.keys()
    solver_options = {key: value for key, value in options_all.items() if key not in user_keys}

    for option, value in solver_options.items():
        nlp.add_option(option, value)

    if(not opts.quiet):
        print(f'\n {"":10} Iter {"":25} obj {"":22} infeas')

    # solve the problem
    g, ipopt_info = nlp.solve(x0)

    wh_opt = g * wh
    # targets_opt = xmat.T.dot(wh_opt)  # sparse
    targets_opt = np.dot(xmat.T, wh_opt)  # dense
    b = timer()

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'wh_opt',
              'targets_opt',
              'g',
              'opts',
              'ipopt_info')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds=b - a,
                 wh_opt=wh_opt,
                 targets_opt=targets_opt,
                 g=g,
                 opts=opts,
                 ipopt_info=ipopt_info)

    return res


# %% reweight class
class RW:

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
