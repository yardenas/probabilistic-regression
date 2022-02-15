import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

import numpy as np

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def net(x, init_stddev=0.001):
  init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
  x = hk.nets.MLP((2,), w_init=init)(x)
  mu, stddev = jnp.split(x, 2, -1)
  mu = mu.squeeze(-1)
  stddev = stddev.squeeze(-1)
  init_stddev = np.log(np.exp(init_stddev) - 1.0)
  stddev = jnn.softplus(stddev + init_stddev) + 1e-4
  return mu, 0.1 * stddev


def to_list_preds(mus, stddevs):
  yhats = list(map(lambda mu, stddev: tfd.Normal(mu, stddev), mus, stddevs))
  return yhats
