

# %% imports
import numpy as np

import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import random
from jax import value_and_grad
from jax import vmap

# Importing TFP on JAX
from tensorflow_probability.substrates import jax as tfp
# tfd = tfp.distributions
# tfb = tfp.bijectors
# tfpk = tfp.math.psd_kernels

# %% set up problem
minimum = jnp.array([1.0, 1.0])  # The center of the quadratic bowl.
scales = jnp.array([2.0, 3.0])  # The scales along the two axes.

# The objective function and the gradient.
def quadratic_loss(x):
  return jnp.sum(scales * jnp.square(x - minimum))

start = jnp.array([0.6, 0.8])  # Starting point for the search.

# %% solve BFGS can find the minimum of this loss.

optim_results = tfp.optimizer.bfgs_minimize(
    value_and_grad(quadratic_loss), initial_position=start, tolerance=1e-8)

# Check that the search converged
assert(optim_results.converged)
# Check that the argmin is close to the actual value.
np.testing.assert_allclose(optim_results.position, minimum)
# Print out the total number of function evaluations it took. Should be 5.
print("Function evaluations: %d" % optim_results.num_objective_evaluations)

# %% L-BFGS solve

optim_results = tfp.optimizer.lbfgs_minimize(
    value_and_grad(quadratic_loss), initial_position=start, tolerance=1e-8)

dir(optim_results)    
optim_results.num_iterations


# Check that the search converged
assert(optim_results.converged)
# Check that the argmin is close to the actual value.
np.testing.assert_allclose(optim_results.position, minimum)
# Print out the total number of function evaluations it took. Should be 5.
print("Function evaluations: %d" % optim_results.num_objective_evaluations)
