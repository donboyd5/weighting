# https://personal.ntu.edu.sg/lixiucheng/books/jax/jax-autodiff.html
# http://implicit-layers-tutorial.org/implicit_functions/
# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

# https://scholar.princeton.edu/sites/default/files/nickmcgreivy/files/simons_summer_school_talk_august_21st.pdf
# A Jacobian-vector (JVP) product with one column of an identity matrix
# gives one column of the Jacobian matrix


import jax
import jax.numpy as jnp
import numpy as np


# %% setup

def f(x):
    return jnp.asarray([x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])

x = jnp.array([1., 2., 3.])
jax.jacrev(f)(x)
jf = jax.jacobian(f)

jf(x)

jax.vjp(f)
jax.vjp(f, x)
vfn = jax.vjp(f, x)
jax.vmap(vfn)

def vmap_mjp(f, x):
    vjp_fun = jax.vjp(f, x)
    return jax.vmap(vjp_fun)


def vmap_mjp(f, x):
    y, vjp_fun = jax.vjp(f, x)
    return jax.vmap(vjp_fun)

vmap_mjp(f, x)
print(res)

# %% more
# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
from jax import jacfwd as builtin_jacfwd

def f(x):
    return jnp.asarray([x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])


def our_jacfwd(f):
    def jacfun(x):
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun

x = jnp.array([1., 2., 3.])
f(x)
builtin_jacfwd(f)(x)
our_jacfwd(f)(x)
mj = our_jacfwd(f)
mj(x)


assert jnp.allclose(builtin_jacfwd(f)(x), our_jacfwd(f)(x)), 'Incorrect forward-mode Jacobian results!'

# %% even more
# %% more
# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
from jax import jacfwd as builtin_jacfwd

# f = lambda W: predict(W, b, inputs)
# def vmap_mjp(f, x, M):
#     y, vjp_fun = vjp(f, x)
#     outs, = vmap(vjp_fun)(M)
#     return outs

# def vmap_jmp(f, W, M):
#     _jvp = lambda s: jvp(f, (W,), (s,))[1]
#     return vmap(_jvp)(M)

x = jnp.array([1., 2., 3.])
y = 7.0

def f(x, y):
    return jnp.asarray([x[0]*y, 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])

def vmap_jmp(f, x, M):
    _jvp = lambda s: jvp(f, (x,), (s,))[1]
    return vmap(_jvp)(x)




def our_jacfwd(f):
    def jacfun(x, y):
        _jvp = lambda s: jax.jvp(f, (x, y), (s,))[1]
        Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun

f(x, y)
builtin_jacfwd(f)(x, y)
our_jacfwd(f)(x, y)

from jax import random
key = random.PRNGKey(0)
num_vecs = 128
S = random.normal(key, (num_vecs,) + x.shape)
f = lambda W: predict(W, b, inputs)

g = lambda x: f(x, y)

def vmap_jmp(f, x, M):
    _jvp = lambda s: jvp(f, (x,), (s,))[1]
    return vmap(_jvp)(x)

f(x)

vmap_vs = vmap_jmp(f, W, M=S)


# djb this is it solution starts here 4/23/2021
def g(x, y):
    return jnp.asarray([x[0]*y*y, 5*x[2], 4*x[1]**2 - 2*x[2]])


f = lambda x: g(x, y)
x = jnp.array([1., 2., 3.])
y = 3.0


g(x, y)
u = lambda s: jax.jvp(f, (x,), (s,))[1]
jt = jax.vmap(u, in_axes=1)(jnp.eye(len(x)))
jt.T
jax.jacfwd(g)(x, y)


# create a function-based version of same -------

def g(x, y):
    return jnp.asarray([x[0]*y*y, 5*x[2], 4*x[1]**2 - 2*x[2]])

def jac_jvp(g):  # this is good
    f = lambda x: g(x, y)
    def jacfun(x, y):
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun

def jac_jvp(g, y):
    f = lambda x: g(x, y)
    def jacfun(x, y):
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun


myfn = jac_jvp(g, y)
x = jnp.array([1., 2., 3.])
y = 3.0
myfn(x, y)
jax.jacfwd(g)(x, y)


def g(x, y, z):
    return jnp.asarray([x[0]*y*y, 5*x[2]*z, 4*x[1]**2 - 2*x[2]])

def jac_jvp(g):  # this is good
    f = lambda x: g(x, y, z)
    def jacfun(x, y, z):
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun
f2 = jac_jvp(g)
x = jnp.array([1., 2., 3.])
y = 3.0
z = -10.0
f2(x, y, z)
jax.jacfwd(g)(x, y, z)


# older below here
jnp.matmul(uvec, imat)
u(x)(jnp.eye(len(x)))
uvec = u(x)
imat = jnp.eye(len(x))

# Push forward the vector `v` along `f` evaluated at `W`
z, u = jax.jvp(f, (x,), (x,))
z
u
# z is g(x, y) evaluated at x, y

