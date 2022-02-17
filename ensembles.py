import functools
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import utils

tfd = tfp.distributions


class Ensemble:

  def __init__(self, example: jnp.ndarray, n_particles: int, model: Callable):
    net = hk.without_apply_rng(hk.transform(lambda x: model(x)))
    self.net = net.apply
    init = jax.vmap(net.init, (0, None))
    seed_sequence = hk.PRNGSequence(666)
    self.particles = init(jnp.asarray(seed_sequence.take(n_particles)), example)

  def update_step(self, params, _, x, y):
    grads = self._grad_step(params, x, y)
    return grads

  @property
  def params(self):
    return self.particles

  @params.setter
  def params(self, state):
    self.particles = state

  @functools.partial(jax.jit, static_argnums=0)
  def _grad_step(self, params, x, y):

    def neg_log_likelihood(params):
      mu, raw_stddev = self.net(params, x)
      dist = tfd.Normal(mu, utils.get_stddev(raw_stddev))
      return -dist.log_prob(y).mean()

    neg_log_likelihood_grad = jax.vmap(jax.grad(neg_log_likelihood))(params)
    return neg_log_likelihood_grad

  def predict(self, x):
    return self._predict(x, self.particles)

  @functools.partial(jax.jit, static_argnums=0)
  def _predict(self, x, particles):
    forward = jax.vmap(self.net, (0, None))
    mus, raw_stddevs = forward(particles, x)
    return utils.to_list_preds(mus, utils.get_stddev(raw_stddevs))
