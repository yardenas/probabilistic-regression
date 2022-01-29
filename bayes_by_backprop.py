from typing import Callable

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class MeanField(hk.Module):
  def __init__(
      self,
      params: hk.Params,
      stddev=1.0,
      learnable=True
  ):
    super(MeanField, self).__init__()
    flat_ps, tree_structure = jax.tree_flatten(params)
    self._ps = flat_ps
    self._flat_event_shape = jax.tree_map(lambda x: x.shape, flat_ps)
    self._stddev = stddev
    self._learnable = learnable

  def __call__(self):
    if self._learnable:
      mus = [hk.get_parameter(
        'mean_field_mu%d' % index, p.shape, init=hk.initializers.Constant(p)
      ) for index, p in enumerate(self._ps)]
      stddevs = [hk.get_parameter(
        'mean_field_stddev%d' % index, s,
        init=hk.initializers.UniformScaling(self._stddev)
      ) for index, s in enumerate(self._flat_event_shape)]
    else:
      mus = [np.zeros_like(p) for p in self._ps]
      stddevs = [np.ones_like(p) * self._stddev for p in self._ps]
    flat_params = jax.tree_map(np.ravel, self._ps)
    flat_params = np.concatenate(flat_params)
    # Add a bias to softplus such that a zero value to the stddevs parameter
    # gives the empirical standard deviation.
    empirical_stddev = flat_params.std()
    init = np.log(np.exp(empirical_stddev) - 1.0)
    stddevs = jax.tree_map(lambda x: jnn.softplus(x + init) + 1e-6, stddevs)
    out = tfd.JointDistributionSequential(
      [tfd.Independent(tfd.MultivariateNormalDiag(*ps), len(ps[0].shape) - 1)
       for ps in zip(mus, stddevs)]
    )
    return out


class BNN(hk.Module):
  def __init__(
      self,
      apply_fn: Callable,
      params: hk.Params,
      posterior_stddev: float,
      prior_stddev: float
  ):
    super(BNN, self).__init__()

    self.forward = apply_fn
    self.posterior = MeanField(params, posterior_stddev)
    self.prior = MeanField(params, prior_stddev, learnable=False)
    flat_ps, tree_structure = jax.tree_flatten(params)
    self._unflatten = tfb.Restructure(
      jax.tree_unflatten(tree_structure, range(len(flat_ps)))
    )

  def __call__(self, x: jnp.ndarray):
    params = self.tree_dist.sample(seed=hk.next_rng_key())
    return self.forward(params, x)

  @property
  def tree_dist(self):
    return tfd.TransformedDistribution(self.posterior(), self._unflatten)

  def posterior(self):
    return self.tree_dist

  def prior(self):
    return tfd.TransformedDistribution(self.prior(), self._unflatten)
