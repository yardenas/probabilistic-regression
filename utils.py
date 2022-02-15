import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

import numpy as np

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def inv_softplus(x):
    if x>20:
        return x
    return np.log(np.exp(x) - 1.)


get_stddev = jnn.softplus


def net(x, activation=jnn.relu, n_layers=0, n_hidden=50, init_stddev=0.1, sd_min=1e-4, sd_max=1e10):
  init = hk.initializers.VarianceScaling(1.0, "fan_in", "normal")
  layers = [n_hidden] * n_layers + [2]
  x = hk.nets.MLP(tuple(layers), activation=activation, w_init=init)(x)
  mu, raw_stddev = jnp.split(x, 2, -1)
  raw_stddev = jnp.clip(
    raw_stddev + inv_softplus(init_stddev), *map(inv_softplus, (sd_min, sd_max)))
  return mu.squeeze(-1), raw_stddev.squeeze(-1)


def to_list_preds(mus, raw_stddevs):
  yhats = list(map(lambda mu, raw_stddev: tfd.Normal(mu, get_stddev(raw_stddev)), mus, raw_stddevs))
  return yhats
