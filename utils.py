import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

import numpy as np


def net(x, init_stddev=1.0):
  init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
  x = hk.nets.MLP((2,))(x)
  mu, stddev = jnp.split(x, 2, -1)
  mu = mu.squeeze(-1)
  stddev = stddev.squeeze(-1)
  init_stddev = np.log(np.exp(init_stddev) - 1.0)
  stddev = jnn.softplus(stddev + init_stddev) + 1e-6
  return mu, stddev
