# documentation and examples
#   https://cyipopt.readthedocs.io/en/stable/
#   https://cyipopt.readthedocs.io/en/stable/tutorial.html#problem-interface
#   https://cyipopt.readthedocs.io/en/stable/reference.html


# set up a small problem for ipopt in sparse form
# we need the jacobian structure
# right now that is np.nonzero(cc.T)


# %% imports

# Load the autoreload extension
# %load_ext autoreload
# %reload_ext autoreload

# Autoreload reloads modules before executing code
# 0: disable
# 1: reload modules imported with %aimport
# 2: reload all modules, except those excluded by %aimport
# %autoreload 2

import importlib
import numpy as np
import scipy.sparse as sps
import cyipopt as cy

import src.make_test_problems as mtp
import src.microweight as mw

# %% reimports
importlib.reload(mw)


# %% create data
p = mtp.Problem(h=30, s=1, k=3)
# p = mtp.Problem(h=200000, s=1, k=30)

n = p.h  # n number variables
m = p.k  # m number constraints

xmat = p.xmat.copy()
wh = p.wh.copy()

# randomly set some elements of xmat to zero
pctzero = .2
np.random.seed(1)
indices = np.random.choice(np.arange(xmat.size), replace=False, size=int(xmat.size * pctzero))
xmat[np.unravel_index(indices, xmat.shape)] = 0 

p.xmat
xmat

#  constraint coefficients (constant)
cc = (xmat.T * wh).T
cc # same shape as xmat

x0 = np.ones(n)

# %% create targets, adding noise
np.random.seed(1)
noise = np.random.normal(0, .02, m)
np.round(noise * 100, 2)

init_vals = np.dot(x0, cc)
targets = init_vals * (1 + noise)
np.dot(xmat.T, wh)

init_pdiff = (init_vals - targets) / targets * 100
# equivalently: 100 / (1 + noise) - 100

init_sspd = np.square(init_pdiff).sum()

# %% test solve with microweight
prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets)
optip = {'xlb': .1, 'xub': 10,
         'crange': 0.005,
         'print_level': 0,
         'file_print_level': 5,
         # 'derivative_test': 'first-order',
         # 'ccgoal': 1e2,
         # 'objgoal': 1,
         'max_iter': 100,
         'linear_solver': 'ma86',  # ma27, ma77, ma57, ma86 work, not ma97
         # 'ma86_order': 'metis',
         # 'ma97_order': 'metis',
         # 'mumps_mem_percent': 100,  # default 1000
         # 'obj_scaling_factor': 1e0, # must be float
         # 'linear_system_scaling': 'slack-based',
         # 'ma57_automatic_scaling': 'yes',
         'quiet': False}

rw1 = prob.reweight(method='ipopt', options=optip)      
rw1.elapsed_seconds
rw1.sspd
np.round(rw1.pdiff, 1)


# %% create class
class RW():

    # def __init__(self):
    #    pass

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return np.sum((x - 1)**2)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return 2*x - 2

    def constraints(self, x):
        """Returns the constraints."""
        return np.dot(x, cc)   

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return jnz

    def jacobianstructure(self):
        """ Define sparse structure of Jacobian. """
        return jstruct        

    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        return obj_factor * hnz

    def hessianstructure(self):
        """ Define sparse structure of Hessian. """
        return hstruct

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

        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

# ctrl-k-c comment, ctrl-k-u uncomment

# %% run the nlp
lb = np.full(n, 0.1)
ub = np.full(n, 10)

cl = targets - abs(targets) * .005
cu = targets + abs(targets) * .005

J = cc.T
jstruct = np.nonzero(J)
jnz = J[jstruct]

hidx = np.arange(0, n, dtype='int64')
hstruct = (hidx, hidx)
hnz = np.full(n, 2)

nlp = cy.Problem(
    n=n,
    m=m,
    problem_obj=RW(),
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu
    )

outfile = '/home/donboyd/Documents/test.out'
nlp.add_option('output_file', outfile) 
nlp.add_option('file_print_level', 5)
nlp.add_option('jac_d_constant', 'yes')
nlp.add_option('hessian_constant', 'yes')
nlp.add_option('linear_solver', 'ma86')
# nlp.add_option('derivative_test', 'second-order')
# nlp.add_option('max_iter', 3)

g, ipopt_info = nlp.solve(x0)

g



# nlp.add_option('output_file', outfile) 

# solver_defaults = {
#     'print_level': 0,
#     'file_print_level': 5,
#     'jac_d_constant': 'yes',
#     'hessian_constant': 'yes',
#     'max_iter': 100,
#     'mumps_mem_percent': 100,  # default 1000
#     'linear_solver': 'ma57'
# }


# %% options
optip = {'xlb': .1, 'xub': 10,
         'crange': 0.005,
         'print_level': 0,
         'file_print_level': 5,
         'max_iter': 100,
         'linear_solver': 'ma86',  # ma27, ma77, ma57, ma86 work, not ma97
         'quiet': False}


# %% test the new sparse version
prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets)

rw1a = prob.reweight(method='ipopt', options=optip)  
rw1a.elapsed_seconds
rw1a.sspd
rw1a.pdiff
# print(sps.csc_matrix(xmat))


# %% examine

# for jacobian, we want the nonzero elements of the cc transpose
cc.T
np.nonzero(cc.T)
sps.find(cc.T)

%timeit np.nonzero(cc.T)  # 2.24us
%timeit sps.find(cc.T)  # 52.5us


# ijnz is a tuple with indexes and values of nonzero elements
# it has 3 elements, each of which is an array
ijnz = sps.find(cc.T)
ijnz[0] # row indexes of nz elements
ijnz[1] # col indexes of nz elements
ijnz[2] # nz elements



# what we need for ipopt:





# define options (add_option)

# %% how big could cc be?
# standard reweight:
# 40k variables, 40 targets, 1.6 million, maybe 80% nonzero

# state weights:
# 40k records
# 50 states
# 40 targets
# 2 million variables
# 2k constraints for states
# 40k adding up constraints
# 42k total constraints
# so 42k x 2m = 84e9 constraint coefficients
# but max nonzero constraint coefficients are:
# (a) for state-specific constraints:
#  2k constraints x 1 /50 x 2 m variables = 80e6
# (b) for adding-up constraints:
#  40k constraints x 50 states for each = 2e6
# for max nonzero constraint coefficients of 82e6
recs = 40e3; states = 50; targs = 40  # 82e6
recs = 20e3; states = 50; targs = 25  # 26e6
maxnzcc = (states * targs * recs + states * recs) / 1e6
maxnzcc



# %% scratch
# https://cyipopt.readthedocs.io/en/stable/tutorial.html#tutorial
# https://cyipopt.readthedocs.io/en/stable/reference.html


np.nonzero(np.tril(np.ones((4, 4))))

x = [1, 2, 3, 4]
np.concatenate((np.prod(x)/x, 2*x))



# %% earlier play
# Construct a 1000x1000 lil_matrix and add some values to it:

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import find
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

A = np.matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
A


print(csc_matrix(A))
print(csr_matrix(A))
print(coo_matrix(A))
print(lil_matrix(A))

ijnz = find(A)
ijnz

type(ijnz)
dir(ijnz)

# ijnz is a tuple with indexes and values of nonzero elements
# it has 3 elements, each of which is an array
ijnz[0] # row indexes of nz elements
ijnz[1] # col indexes of nz elements
ijnz[2] # nz elements

# next, review what ipopt needs for the triplet format
ijnz

A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
v = np.array([1, 0, -1])
A.dot(v)
# array([ 1, -3, -1], dtype=int64)


rows = 1000
cols = 1000

A = lil_matrix((rows, cols))
A[0, :100] = rand(100)
A[1, 100:200] = A[0, :100]
A.setdiag(rand(rows))
A.shape
print(A[0:3, 0:15])



A = A.tocsr()
b = rand(1000)
x = spsolve(A, b)

# %% example classes

#    print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
class HS071():
    # this is from the tutorial
    # https://cyipopt.readthedocs.io/en/stable/tutorial.html#problem-interface
    # it does not have __init__ as in the lower case
    # capitalization of class names is PEP8

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return np.array([
            x[0]*x[3] + x[3]*np.sum(x[0:3]),
            x[0]*x[3],
            x[0]*x[3] + 1.0,
            x[0]*np.sum(x[0:3])
        ])

    def constraints(self, x):
        """Returns the constraints."""
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return np.concatenate((np.prod(x)/x, 2*x))

    def hessianstructure(self):
        """Returns the row and column indices for non-zero vales of the
        Hessian."""

        # NOTE: The default hessian structure is of a lower triangular matrix,
        # therefore this function is redundant. It is included as an example
        # for structure callback.

        return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""

        H = obj_factor*np.array((
            (2*x[3], 0, 0, 0),
            (x[3],   0, 0, 0),
            (x[3],   0, 0, 0),
            (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
            (0, 0, 0, 0),
            (x[2]*x[3], 0, 0, 0),
            (x[1]*x[3], x[0]*x[3], 0, 0),
            (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        row, col = self.hessianstructure()

        return H[row, col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value))


    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=hs071(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
        )

    #
    # Set solver options
    #
    #nlp.addOption('derivative_test', 'second-order')
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)

    #
    # Scale the problem (Just for demonstration purposes)
    #
    nlp.set_problem_scaling(
        obj_scaling=2,
        x_scaling=[1, 1, 1, 1]
        )
    nlp.add_option('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)
