
# scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)[source]
# scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method='brent', tol=None, options=None)[source]
from scipy.optimize import minimize, minimize_scalar


number = 0.0000001
f"Number: {number}"
f"Number: {number:f}"
f"Number: {number:.10f}"




def f(p):
    l2 = -p + p**2 + 0.1
    return l2

f(0.1)
f(1)
f(0.5)
f(0.25)

p1 = 0.1
lp1 = f(p1)
p2 = 1.0
lp2 = f(p2)

lbest = min(lp1, lp2)
pbest =
p = min(p1, p2)


p = p +
if lp1 < lp2:
    lbest = lp1
    pbest = p1
else:
    lbest = lp2
    pbest = p2
print(p, pbest, lbest)

minimize(f, 1.0, bounds=(0, 1))
res = minimize_scalar(f, bounds=(0, 1), method='bounded', options={'maxiter': 7, 'disp': True})
res

# optionsdict, optional
# A dictionary of solver options.

# maxiterint
# Maximum number of iterations to perform.

# dispbool
# Set to True to print convergence messages.
