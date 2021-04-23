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
x = np.arange(10, dtype='float')
y = np.arange(3, 13, dtype='float')
x.size
y.size

# def f(x, y):
#     return jnp.square(x - 1) + jnp.square(y)

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

def our_jacfwd(f):
    def jacfun(x):
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun

f(x)
builtin_jacfwd(f)(x)
our_jacfwd(f)(x)

assert jnp.allclose(builtin_jacfwd(f)(x), our_jacfwd(f)(x)), 'Incorrect forward-mode Jacobian results!'

