
import numpy as np
import src.make_test_problems as mtp

p = mtp.Problem(h=10, s=2, k=2)
p.h
p.xmat
p.wh
# np.dot(beta, xmat.T)

z = np.dot(p.xmat, p.xmat.T)
z.shape

z2 = np.dot(p.xmat.T, p.xmat)
z2.shape
z2