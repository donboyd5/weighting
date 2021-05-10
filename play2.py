
import numpy as np
import src.make_test_problems as mtp
import src.microweight as mw

qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)

# %% test problem
p = mtp.Problem(h=10, s=3, k=2, xsd=.1, ssd=.5)
p = mtp.Problem(h=1000, s=20, k=5, xsd=.1, ssd=.5)
wh = p.wh
xmat = p.xmat
geotargets = p.geotargets
targs = p.targets
iwhs = p.whs
np.dot(iwhs.T, xmat)

prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=geotargets)

opts = {'scaling': True,
        'max_nfev': 200}
gw = prob.geoweight(method='poisson-lsq', options=opts)
# geotargets_opt = np.multiply(geotargets_opt, scale_factors)
xmat2, geotargets2, scale_factors = scale_problem(xmat, geotargets, 10.)
gw.sspd
gw.geotargets_opt
gw.whs_opt
gw.whs_opt.T.dot(xmat)
dir(gw.method_result)
beta = gw.method_result.beta_opt  # s x k
beta

def get_whs(beta, xmat, wh, adjust=False):
    # note beta is an s x k matrix
    # beta = beta.reshape((s, k))
    betax = beta.dot(xmat.T)
    # adjust betax to make exponentiation more stable
    # we can add a column-specific constant to each column of betax
    # the best constant will make the max absolute value of the column the lowest possible
    bmax = betax.max(axis=0)
    bmin = betax.min(axis=0)
    mid = (bmax - bmin) / 2.0
    const = np.subtract(mid, bmax)
    if adjust:
        betax = np.add(betax, const)
    ebetax = np.exp(betax)
    worst = np.abs(ebetax).max()
    if worst > 100.:
        print('Warning, worst exponent is > 100:', np.abs(ebetax).max())
    shares = np.divide(ebetax, ebetax.sum(axis=0))
    whs = np.multiply(wh, shares).T
    return whs

whs3 = get_whs(beta, xmat2, wh)
whs3 = get_whs(beta, xmat2, wh, adjust=True)
whs3
np.round(np.dot(whs3.T, xmat) - geotargets, 2)
pdiffs = np.dot(whs3.T, xmat) / geotargets * 100. - 100.
np.quantile(pdiffs, qtiles)

beta

betax = beta.dot(xmat.T)  # ok
ebetax = np.exp(betax)  # large
ebetax
ebetaxs = ebetax.sum(axis=0)  # large
s1 = ebetax / ebetaxs
s1
s1.sum(axis=0)
whs1 = np.multiply(wh, s1).T
np.dot(whs1.T, xmat)
geotargets


# subtract a constant from each column (i.e. for each person)
# bx2 = betax - 100.  # fine to subtract a constant from beta
sub = betax.max(axis=0) # fine to subtract a constant from each column of beta
# sub = np.median(betax, axis=0)
sub
bx2 = np.subtract(betax, sub)
bx2
ebx2 = np.exp(bx2)
ebx2
ebx2s = ebx2.sum(axis=0)
s2 = ebx2 / ebx2s
s2
s2.sum(axis=0)
whs2 = np.multiply(wh, s2).T
np.dot(whs2.T, xmat)
geotargets

# NO NO NO subtract a constant from each row
sub = betax.max(axis=1)
bx2 = np.subtract(betax, sub.reshape((3, 1)))
bx2
ebx2 = np.exp(bx2)
ebx2s = ebx2.sum(axis=0)
s2 = ebx2 / ebx2s
s2
s2.sum(axis=0)
whs2 = np.multiply(wh, s2).T
np.dot(whs2.T, xmat)
geotargets



# can we double subtract?
sub = betax.max(axis=1)


betax
betax - np.ndarray([100., 200., 300.]).reshape((3, 1))

np.subtract(betax, sub, axis=0)


delta = np.log(wh / ebetax.sum(axis=0))
betaxd = (betax + delta).T
whs1 = np.exp(betaxd)
np.dot(whs1.T, xmat)


whs2 = (wh * ebetax / ebetaxs).T
whs2
np.dot(whs2.T, xmat)

ebetax.sum(axis=0)  # h x 1

part1 = np.exp(betax).T
part2d = part1.T.sum(axis=0)
part2 = np.divide(wh, part2d)
delta = np.log(part2)
np.exp(delta)
whs1 = np.exp(betax + delta).T
np.dot(whs1.T, xmat)


beta_x = np.dot(beta, xmat.T)
ebeta_x = np.exp(np.dot(beta, xmat.T))
delta = np.log(wh / ebeta_x.sum(axis=0))
beta_xd = (beta_x + delta).T
weights = np.exp(beta_xd)





part1.shape
part2.shape
cwhs = np.multiply(part1, part2.reshape((10, 1)))
np.dot(cwhs.T, xmat)
geotargets

m = np.array([[1., 2., 3.],
              [4., 5., 6.]])
m

np.exp(m)

# %% temp
beta_x = np.dot(beta, xmat.T)
ebeta_x = np.exp(np.dot(beta, xmat.T))
delta = np.log(wh / ebeta_x.sum(axis=0))
beta_xd = (beta_x + delta).T
weights = np.exp(beta_xd)

delta_opt = get_delta(wh, beta, xmat)
whs_opt = get_geoweights(beta, delta_opt, xmat)
# geotargets_opt = get_geotargets(beta_opt, wh, xmat)
# geotargets_opt = np.multiply(geotargets_opt, scale_factors)



# %% non-jax functions
def get_delta(wh, beta, xmat):
    beta_x = np.exp(np.dot(beta, xmat.T))
    delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta

def get_diff_weights(geotargets, goal=100):
    goalmat = np.full(geotargets.shape, goal)
    diff_weights = np.where(geotargets != 0, goalmat / geotargets, 1)
    return diff_weights

def get_geotargets(beta, wh, xmat):
    delta = get_delta(wh, beta, xmat)
    whs = get_geoweights(beta, delta, xmat)
    targets_mat = np.dot(whs.T, xmat)
    return targets_mat

def get_geoweights(beta, delta, xmat):
    beta_x = np.dot(beta, xmat.T)

    # add the delta vector of household constants to every row
    # of beta_x and transpose
    # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
    beta_xd = (beta_x + delta).T
    weights = np.exp(beta_xd)

    return weights



# %% jax functions
def get_delta(wh, beta, xmat):
    beta_x = jnp.exp(jnp.dot(beta, xmat.T))
    delta = jnp.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta

def get_diff_weights(geotargets, goal=100):
    goalmat = jnp.full(geotargets.shape, goal)
    diff_weights = jnp.where(geotargets != 0, goalmat / geotargets, 1)
    return diff_weights

def get_geotargets(beta, wh, xmat):
    delta = get_delta(wh, beta, xmat)
    whs = get_geoweights(beta, delta, xmat)
    targets_mat = jnp.dot(whs.T, xmat)
    return targets_mat

def get_geoweights(beta, delta, xmat):
    beta_x = np.dot(beta, xmat.T)

    # add the delta vector of household constants to every row
    # of beta_x and transpose
    # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
    beta_xd = (beta_x + delta).T
    weights = np.exp(beta_xd)

    return weights


# %% stub problem
import pickle
import pandas as pd

# %% alternatively get pickled problem
IGNOREDIR = '/media/don/ignore/'
WEIGHTDIR = IGNOREDIR + 'puf_versions/weights/'

pkl_name = IGNOREDIR + 'pickle.pkl'
open_file = open(pkl_name, "rb")
pkl = pickle.load(open_file)
open_file.close()

targvars, ht2wide, pufsub, dropsdf_wide = pkl
wfname_national = WEIGHTDIR + 'weights2017_georwt1.csv'
wfname_national
final_national_weights = pd.read_csv(wfname_national)
pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])


# %% get stub and make problem from alternative data
stub = 4
qx = '(ht2_stub == @stub)'

pufstub = pufsub.query(qx)[['pid', 'ht2_stub'] + targvars]
# pufstub.replace({False: 0.0, True: 1.0}, inplace=True)
pufstub[targvars] = pufstub[targvars].astype(float)

# get targets and national weights
targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
whdf = pd.merge(pufstub[['pid']], final_national_weights[['pid', 'weight']], how='left', on='pid')

wh = whdf.weight.to_numpy()

xmat = pufstub[targvars].astype(float).to_numpy()
xmat
xmat[:, 0:7]
xmat.sum(axis=0)

geotargets = targetsdf[targvars].to_numpy()
(geotargets==0).sum()
# geotargets = np.where(geotargets==0, 1e3, geotargets)


# %% scaling
def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = np.divide(xmat, scale_factors)
    geotargets = np.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors


# %%
