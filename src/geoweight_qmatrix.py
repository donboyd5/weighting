# -*- coding: utf-8 -*-
"""

@author: donbo

qrake calculates geographic area weights for a national microdata file
such that:

    (1) the sum of the area weights for each household equals the household's
    national weight, and

    (2) the weighted totals for selected variables equal or come as close
    as practical to target totals, for each area, for each variable.

The method is based largely on the method described in:

    Randrianasolo, Toky, and Yves Tillé. “Small Area Estimation by Splitting
    the Sampling Weights.” Electronic Journal of Statistics 7, no. 0
    (2013): 1835–55. https://doi.org/10.1214/13-EJS827.

The code is based largely on R code provided by Toky Randrianasol to
Don Boyd by email on on October 1, 2020, and ported to python by Boyd
and subsequently revised by Boyd.

I would recommend citing the above paper and the authors' code in any
public work that uses the code below.

The main deviation from the R code is that the R code used the raking method
of the R function calib from the R package sampling (written by Yves Tillé).
Instead, I use the python function maybe_exact_calibrate from the python
package empirical_calibration.

The empirical_calibration package is described in the paper:

    Wang, Xiaojing, Jingang Miao, and Yunting Sun. “A Python Library For
    Empirical Calibration.” ArXiv:1906.11920 [Stat], July 25, 2019.
    http://arxiv.org/abs/1906.11920.

I use maybe_exact_calibrate in my main approach because:

    (1) It appears to be very robust. In particular, if it cannot find a
    solution that fully satisfies every target, it automatically relaxes
    constraints to find a very good solution that is close to satisfying
    targets.

    (2) It allows us to assign priority weights to targets, which may be
    useful in the future.

    (3) As with the calib raking method, it can solve for a set of weighting
    factors that are near to baseline weights.

    (4) It is very fast, although I have not tested it to see whether it
    is faster than calib's raking method.

The function also has an autoscale option intended to reduce potential
numerical difficulties, but in my experimentation it has not worked well
so I do not use this option in the code below.

One important adjustment: maybe_exact_calibrate solves for weights that
hit or come close to target weighted means, whereas normally we specify
our problems so that the targets are weighted totals. I take this into
account by:

    (1) Converting area target totals to are target means by dividing
    each area target total by the area target population,

    (2) Solving for mean-producing weights, and

    (3) Converting to sum-producing weights by multiplying the
    mean-producing weights by the area target population.

The python code for maybe_exact_calibrate is at:

    https://github.com/google/empirical_calibration/blob/master/empirical_calibration/core.py

I have also ported the R code for calib's raking method (but not other
methods) to python, and provide that as a backup method.

The empirical_calibration package can be installed with:

    pip install -q git+https://github.com/google/empirical_calibration


****** Here is a summary of how the code works. ******

N:  number of households (tax returns, etc.), corresponding index is i
D:  number of areas (states, etc.), corresponding index is j
w:  vector of national household weights of length N, indexed by i

[NOTE: must update code below to use proper i and j indexes]

Q:  an N x D matrix of proportions of national weights, where:
    Q[i, j] is the proportion of household i's national weight w[i] that is
        in area j

    The sum of each row of Q must be 1. This imposes the requirement that
    a household's area weights must sum to the household's national weights,
    for every household.

In the code below:

We need the equivalent of the g weights returned by R's calib function.
See https://www.rdocumentation.org/packages/sampling/versions/2.8/topics/calib.

Those values, when multiplied by population weights, give sample weights that
hit or come close to desired targets.

At each iteration, for each state i, we multiply Q[i, ] by g to get updated
Q[i, ] for next iteration, where Q[i, k] .

The return from ec.maybe_exact_calibrate

https://github.com/google/empirical_calibration/blob/master/empirical_calibration/core.py
https://github.com/google/empirical_calibration/blob/master/notebooks/survey_calibration_cvxr.ipynb

"""

# %% imports

import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from timeit import default_timer as timer

# pip install -q git+https://github.com/google/empirical_calibration
import empirical_calibration as ec

import src.utilities as ut
import src.raking as raking
import src.reweight_ipopt as rwip
import src.reweight_leastsquares as rwls


# %% default options for the user and for each of the possible methods

# user can pass in different values, but these are the only user options
user_defaults = {
    'Q': None,
    'qshares': None,
    'qmax_iter': 100,
    'drops': None,
    'independent': False}

# solver options below here - user can pass different value and can also
# pass additional valid options for a solver
ec_defaults = {
    'target_weights': None,
    'objective': 'ENTROPY',
    'autoscale': False,
    'increment': 0.001
    }

# most of the ipopt options below are defined and used in reweight_ipopt code
# many additional ipopt options are possible
ipopt_defaults = {
    'xlb': 0.1,
    'xub': 100,
    'crange': .02,
    'ccgoal': 1,
    'objgoal': 100,
    'quiet': False,
    'max_iter': 100
    }

lsq_defaults = {
    'xlb': 0.1,
    'xub': 100.0,
    'method': 'bvls',  # bvls or trf
    'scaling': False,
    'verbose': 0
    }

raking_defaults = {
    'max_rake_iter': 10
    }


# %% constants

SMALL_POSITIVE = np.nextafter(np.float64(0), np.float64(1))
# not sure if needed: a small nonzero number that can be used as a divisor
SMALL_DIV = SMALL_POSITIVE * 1e16
# 1 / SMALL_DIV  # does not generate warning

QUADRATIC = ec.Objective.QUADRATIC
ENTROPY = ec.Objective.ENTROPY


# %% qmatrix - the primary function
def qmatrix(wh, xmat, geotargets,
            method='raking',
            options=None):
    """Docstring.

    """

    # TODO:

    a = timer()

    # define gfn, the function to use in the qmatrix loop
    #   gfn is passed:
        #  column of the Q matrix, representing an area
        #  wh-weighted xmat with good columns only
        # row of the geotargets matrix, representing an area, good columns only
        # optional parameter, objective, needed only for empirical calibration
    #   gfn returns g, ratio of new weights to old weights

    if method == 'raking':
        gfn = g_raking
        solver_defaults = raking_defaults
    elif method == 'empcal':
        gfn = g_ec
        solver_defaults = ec_defaults
    elif method == 'ipopt':
        gfn = g_ipopt
        solver_defaults = ipopt_defaults
    elif method == 'least_squares':
        gfn = g_lsq
        solver_defaults = lsq_defaults

    options_defaults = {**solver_defaults, **user_defaults}

    # update options with any user-supplied options
    # copy seemed safer than ** to me but I am not sure why
    if options is None:
        options_all = options_defaults.copy()
    else:
        options_all = options_defaults.copy()
        options_all.update(options)
        # options = {**options_defaults, **options}

    if method == 'empcal':
        # replace string name for obective with the corresponding empcal object
        if options_all['objective'] == 'ENTROPY':
            options_all['objective'] = ENTROPY
        elif options_all['objective'] == 'QUADRATIC':
            options_all['objective'] = QUADRATIC

    # create a dict that only has solver options, for passing to gfn
    user_keys = user_defaults.keys()
    solver_options = {key: value for key, value in options_all.items() if key not in user_keys}

    # convert options_all dict to named tuple for ease of use
    # uo = ut.dict_nt(user_options)
    # so = ut.dict_nt(solver_options)

    opts = ut.dict_nt(options_all)

    # if user_options is None:
    #     user_options = user_defaults
    # else:
    #     user_options = {**user_defaults, **user_options}

    # if solver_options is None:
    #     solver_options = solver_defaults
    # else:
    #     solver_options = {**solver_defaults, **solver_options}

    # unpack selected user options
    Q = opts.Q
    drops = opts.drops
    qmax_iter = opts.qmax_iter
    # independent means we do one iteration and use UNADJUSTED Q!!!
    if opts.independent:
        qmax_iter = 1  # should add warning if independent and qmax_iter !=1
    print(f'max Q iterations: {qmax_iter}')

    # constants
    # EPS = 1e-5  # acceptable weightsum error (tolerance) - 1e-5 in R code
    TOL_WTDIFF = 0.0005  # tolerance for difference between weight sum and 1
    TOL_TARGPCTDIFF = 1.0  # tolerance for geotargets percent difference

    # initialize stopping criteria values
    ediff = 1  # error, called ver in Toky R code
    iter = 1  # initialize iteration count called k in Toky R. R code
    iter_best = iter

    # difference in weights - Toky R. used sum of absolute weight differences,
    # I use largest absolute weight difference
    max_weight_absdiff = 1e9  # initial maximum % difference between sums of weights for a household and 100
    max_targ_abspctdiff = 1e9  # initial maximum % difference vs geotargets
    max_diff_best = max_targ_abspctdiff

    m = geotargets.shape[0]  # number of states
    n = wh.size  # number of households
    wh = wh.reshape((-1, 1))  # ensure the proper shape

    # If initial Q was not provided construct a Q
    if Q is None:
        Q = np.full((n, m), 1 / m)

    # compute xmat_wh before loop (calib calculates it in the loop)
    xmat_wh = xmat * wh  # shape -  n x number of geotargets

    # numbers of geotargets
    nt_per_area = geotargets.shape[1]
    nt_possible = nt_per_area * m
    if drops is None:
        drops = np.zeros(geotargets.shape, dtype=bool)  # all False
        nt_dropped = 0
    else:
        # nt_dropped = sum([len(x) for x in drops.values()])
        nt_dropped = drops.sum()
    nt_used = nt_possible - nt_dropped
    good_targets = np.logical_not(drops)


    # Making a copy of Q is crucial. We don't want to change the
    # original Q. Am I sure of this??
    Qmat = Q.copy()
    Q_best = Q.copy()
    Q_unadjusted = Q.copy()  # Q_unadjusted is Q prior to forced summation to 1

    print('')
    print_problem(wh, m, nt_per_area, nt_possible, nt_dropped, nt_used)

    h1 = "                  max weight      max target       p95 target"
    h2 = "   iteration        diff           pct diff         pct diff"
    print('\n')
    print(h1)
    print(h2, '\n')

    while not end_loop(iter, max_targ_abspctdiff, qmax_iter, TOL_TARGPCTDIFF):

        print(' '*3, end='')
        print('{:4d}'.format(iter), end='', flush=True)

        for j in range(m):  # j indexes areas
            # print(f'iter {iter:4d}, area {j:5d}')

            good_cols = good_targets[j, :]

            g = gfn(Qmat[:, j],
                    xmat_wh[:, good_cols],
                    geotargets[j, good_cols],
                    options=solver_options)

            # if method == 'raking' and g is None:
            #     # try to recover by using the alternate method
            #     g = g_ec(xmat_wh[:, good_cols], Q[:, j], geotargets[j, good_cols])
            if g is None:
                g = np.ones(n)

            if np.isnan(g).any() or np.isinf(g).any() or g.any() == 0:
                print('bad g')
                g = np.ones(g.size)
                # we'll need to do this one again
            else:
                pass
                # print("done with this area")
            # print(g)
            Qmat[:, j] = Qmat[:, j] * g.reshape(g.size, )  # end for loop for this area

        # print(Qmat)

        # when we arrive here we have completed all areas for this iteration
        # calc max weight difference BEFORE recalibrating Q
        abswtdiff = np.abs(Qmat.sum(axis=1) - 1)  # sum of weight-shares for each household
        max_weight_absdiff = abswtdiff.max()  # largest difference from 1 across all households
        print(' '*11, end='')
        print(f'{max_weight_absdiff:8.4f}', end='')
        if np.isinf(abswtdiff).any():
            # these weight shares are not good, do another iteration
            # ediff = EPS
            max_weight_absdiff = TOL_WTDIFF
            print("Existence of infinite coefficients --> non-convergence.")

        #print("Weight sums max percent difference: {}".format(maxadiff))  # ediff
        # if iter == 1:
        # Q_unadjusted = Qmat.copy()  # save for possible postprocessing
        if not opts.independent:  # update matrix to force summation to 1, if NOT independent
            Qmat = Qmat / Qmat.sum(axis=1)[:, None]  # Recalibrate Q. Note None so that we have proper broadcasting

        # calculate geotargets pct diff AFTER recalibrating Q
        # this is simply for interim reporting
        whs = np.multiply(Qmat, wh.reshape((-1, 1)))  # faster
        diff = np.dot(whs.T, xmat) - geotargets
        abspctdiff = np.abs(diff / geotargets * 100)
        max_targ_abspctdiff = abspctdiff[good_targets].max()

        ptile = np.quantile(abspctdiff[good_targets], (.95))
        print(' '*6, end='')
        print(f'{max_targ_abspctdiff:8.2f} %', end='')
        print(' '*7, end='')
        print(f'{ptile:8.2f} %')

        # final processing before next iteration
        if max_targ_abspctdiff < max_diff_best:
            Q_best = Qmat.copy()
            max_diff_best = max_targ_abspctdiff.copy()
            iter_best = iter
        iter = iter + 1
        # end while loop

    # WE ARE NOW DONE WITH ALL LOOPING AND WILL POST-PROCESS RESULTS
    # post-processing after exiting while loop, using Q_best, not Q
    # Q_best = Q_unadjusted  # djb!!
    whs_opt = np.multiply(Q_best, wh.reshape((-1, 1)))  # faster
    geotargets_opt = np.dot(whs_opt.T, xmat)
    diff = geotargets_opt - geotargets
    pctdiff = diff / geotargets * 100
    abspctdiff = np.abs(pctdiff)
    # calculate weight difference AFTER final calibration
    abswtdiff = np.abs(Q_best.sum(axis=1) - 1)  # sum of weight-shares for each household
    max_weight_absdiff = abswtdiff.max()  # largest diff from 1 across all households

    if iter > qmax_iter:
        print('\nMaximum number of iterations exceeded.\n')

    print('\n')
    print_problem(wh, m, nt_per_area, nt_possible, nt_dropped, nt_used)

    print(f'\nPost-calibration max abs diff between sum of household weights and 1, across households: {max_weight_absdiff:9.5f}')
    print()

    # compute and print good and all values for various quantiles
    p100a = abspctdiff.max()
    p100m = abspctdiff[good_targets].max()
    p99a = np.quantile(abspctdiff, (.99))
    p99m = np.quantile(abspctdiff[good_targets], (.99))
    p95a = np.quantile(abspctdiff, (.95))
    p95m = np.quantile(abspctdiff[good_targets], (.95))
    sspd = np.square(pctdiff).sum()
    print('Results for calculated targets versus desired targets:')
    print( '                                                              good             all\n')
    print(f'    Max abs percent difference                           {p100m:9.3f} %     {p100a:9.3f} %')
    print(f'    p99 of abs percent difference                        {p99m:9.3f} %     {p99a:9.3f} %')
    print(f'    p95 of abs percent difference                        {p95m:9.3f} %     {p95a:9.3f} %')
    print('\n')
    print(f'Sum of squared percentage differences:      {sspd:9.3g}')
    print(f'Number of iterations:                       {iter - 1:5d}')
    print(f'Best target difference found at iteration:  {iter_best:5d}')

    b = timer()
    print('\nElapsed time: {:8.1f} seconds'.format(b - a))

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whs_opt',
              'geotargets',
              'geotargets_opt',
              'Q_opt',
              'Q_unadjusted',
              'iter_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    res = Result(elapsed_seconds = b - a,
                 whs_opt = whs_opt,
                 geotargets = geotargets,
                 geotargets_opt = geotargets_opt,
                 Q_opt = Q_best,
                 Q_unadjusted = Q_unadjusted,
                 iter_opt = iter_best)
    return res


# %% classes and functions

def end_loop(iter, max_targ_abspctdiff, qmax_iter, TOL_TARGPCTDIFF):
    # define stopping criteria
    iter_rule = (iter > qmax_iter)
    target_rule = (max_targ_abspctdiff <= TOL_TARGPCTDIFF)
    no_more = iter_rule or target_rule
    return no_more


def g_ec(wh, xmat, targets, options):
         # options
         # target_weights: np.ndarray = None,
         # objective: ec.Objective = ec.Objective.ENTROPY,
         # increment: float = 0.001):

    # this is a wrapper to get g, the ratio of new weights to old weights,
    # for the empirical calibration function

    # small_positive = np.nextafter(np.float64(0), np.float64(1))
    wh = np.where(wh == 0, SMALL_POSITIVE, wh)

    pop = wh.sum()
    tmeans = targets / pop

    # ompw:  optimal means-producing weights
    ompw, l2_norm = ec.maybe_exact_calibrate(
        covariates=xmat,
        target_covariates=tmeans.reshape((1, -1)),
        baseline_weights=wh,
        # target_weights=np.array([[.25, .75]]), # target priorities
        target_weights=options['target_weights'],  # target priorities???
        autoscale=options['autoscale'],  # doesn't always seem to work well
        # note that QUADRATIC weights often can be zero
        objective=options['objective'],  # ENTROPY or QUADRATIC
        increment=options['increment']
    )
    # print(l2_norm)

    # wh, when multiplied by g, will yield the targets
    g = ompw * pop / wh
    g = np.array(g, dtype=float).reshape((-1, ))  # djb

    return g


def g_ipopt(wh, xmat, targets, options):
    # this is a wrapper to get g, the ratio of new weights to old weights,
    # for ipopt
    res = rwip.rw_ipopt(wh, xmat, targets, options=options)
    return res.g


def g_lsq(wh, xmat, targets, options):
    # this is a wrapper to get g, the ratio of new weights to old weights,
    # for the least_squares function
    res = rwls.rw_lsq(wh, xmat, targets, options=options)
    return res.g


def g_raking(wh, xmat, targets, options):
    # this is a wrapper to get g, the ratio of new weights to old weights,
    # for the raking function
    return raking.rake(wh, xmat, targets)


def get_drops(targets, drop_dict):
    drops = np.zeros(targets.shape, dtype=bool) # start with all values False
    if drop_dict is not None:
        for row, cols in drop_dict.items():
            drops[row, cols] = True
    return drops


def print_problem(wh, m, nt_per_area, nt_possible, nt_dropped, nt_used):
    print(' Number of households:                {:8,}'.format(wh.size))
    print(' Number of areas:                     {:8,d}'.format(m))
    print()
    print(' Number of targets per area:          {:8,d}'.format(nt_per_area))
    print(' Number of potential targets, total:  {:8,d}'.format(nt_possible))
    print(' Number of targets dropped:           {:8,d}'.format(nt_dropped))
    print(' Number of targets used:              {:8,d}'.format(nt_used))


# %% examples - uncomment code below to run
# import src.make_test_problems as mtp

# # p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
# p = mtp.Problem(h=10000, s=20, k=15)

# n = p.xmat.shape[0]
# m = p.targets.shape[0]
# shares = p.targets[:, 0] / p.targets[:, 0].sum()
# Q = np.tile(shares, n).reshape((n, m))
# # Q = np.full((n, m), 1 / m)

# res = qmatrix(p.wh, p.xmat, p.targets, method='raking', max_iter=50)
# res = qmatrix(p.wh, p.xmat, p.targets,
#               Q = Q,
#               method='raking', max_iter=50)
# # res = qmatrix(Q, p.wh, p.xmat, p.targets, method='raking-ec', max_iter=50)
# res = qmatrix(p.wh, p.xmat, p.targets, Q=Q, method='raking-ec', objective=QUADRATIC, max_iter=50)
# res
# res._fields
# res.elapsed_seconds
# res.iter_opt
# # res = qmatrix(Q, wh, xmat, targets, method='raking-ec', max_iter=100, drops=drops)  # GOOD
# res = qmatrix(Q, wh, xmat, targets, method='raking-ec', max_iter=20, drops=drops, objective=QUADRATIC)


# %% original R code from Toky Randrianasolo 2020-10-01

# MatrixCalib <- function(Q,w,Xs){
# 	ver=1
# 	k=1
# 	while(ver>10^(-5) & k <=500)
# 	{
# 		cat(" n.iter = ", k,"\n")
# 		for(i in 1:m)
# 		{
# 			cat("Domain ",nom[i],": calibration ")
# 			g = calib((Xs*w),Q[,i],TTT[i,],method="raking")
# 			if (is.null(g) | any(is.na(g)) | any(g == 0) | any(is.infinite(g)) ) {g = rep(1,length(Q[,i]));cat("non done","\n")}
# 			else {cat("done","\n")}
# 			Q[,i]=Q[,i]*g
# 		}
# 	ver = sum(abs(rowSums(Q)-1))
# 	if (any(is.infinite(abs(rowSums(Q)-1)))) {ver = 10^(-5);cat("Existence of infinite coefficient(s) : non convergence\n")}
# 	cat("Stop condition :\n ")
# 	print(ver)
# 	Q=Q/rowSums(Q)
# 	k=k+1
# 	if (k > 500) cat("Maximal number of iterations not achieved : non convergence \n")
# 	}
# 	Q
# }

