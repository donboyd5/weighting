
import numpy as np
import src.make_test_problems as mtp
import src.microweight as mw

qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)

opts = {'scaling': False}


# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10, s=3, k=2, xsd=.1, ssd=.5)
beta0 = np.full(p.geotargets.shape, 0.0)
beta05 = np.full(p.geotargets.shape, 0.5)
beta1 = np.full(p.geotargets.shape, 1.0)
beta0.dot(xmat.T)
beta05.dot(xmat.T) # around 100
beta1.dot(xmat.T) # around 200

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=p.targets, geotargets=p.geotargets)
gw = prob.geoweight(method='poisson-lsq', options=opts)
dir(gw.method_result)
beta = gw.method_result.beta_opt
delta = gw.method_result.delta_opt
whs_opt = gw.whs_opt


wh = p.wh
xmat = p.xmat
gt = p.geotargets
targs = p.targets
iwhs = p.whs

# all good
gt
iwhs.T.dot(xmat)
whs_opt.T.dot(xmat)

wh.shape # (h, )
xmat.shape # (h, k)
gt.shape  # (s, k)
targs.shape # (k, )

targs
wh.dot(xmat)  # (k, )

beta.dot(xmat.T)  # around 400, we want this to be small positive or small negative and give right results

# now let's scale
scale = 1 / xmat.sum(axis=0)
xmat2 = np.multiply(xmat, scale)
xmat2
xmat2.shape
xmat2.sum(axis=0)

wh.dot(xmat2)
gt2 = np.multiply(gt, scale)

np.round(np.quantile(beta05.dot(xmat2.T), qtiles), 2)


prob2 = mw.Microweight(wh=wh, xmat=xmat2, geotargets=gt2)
gw2 = prob2.geoweight(method='poisson-lsq', options=opts)
gw2.sspd
beta2 = gw2.method_result.beta_opt
beta2  # pretty close to 1
np.quantile(beta2.dot(xmat2.T), qtiles)  # ranges from -0.4 to + 0.3
delta2 = gw2.method_result.delta_opt
delta2  # close to 2
whs_opt2 = gw2.whs_opt
whs_opt2 - whs_opt
gt2_opt = whs_opt2.T.dot(xmat)  # use original xmat
# compare to original geo targets
p.geotargets
gt2_opt

