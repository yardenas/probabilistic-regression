import functools
from typing import Callable

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import utils

tfd = tfp.distributions
tfb = tfp.bijectors


class ParamsTree:

  def __init__(self, params: hk.Params):
    flat_ps, tree_def = jax.tree_flatten(params)
    self.flattened_params = flat_ps
    self.tree_def = tree_def

  @functools.partial(jax.jit, static_argnums=0)
  def unflatten(self, sample):
    out = []
    i = 0
    for p in self.flattened_params:
      n = p.size
      out.append(sample[i:i + n].reshape(p.shape))
      i += n
    return jax.tree_unflatten(self.tree_def, out)


class MeanField(hk.Module):

  def __init__(self, name: str, params: hk.Params, stddev=1.0, learnable=True):
    super(MeanField, self).__init__(name)
    self._params_tree = ParamsTree(params)
    self._stddev = stddev
    self._learnable = learnable
    flat_params = jax.tree_map(jnp.ravel, self._params_tree.flattened_params)
    flat_params = jnp.concatenate(flat_params)
    self._flat_params = flat_params

  def __call__(self):
    if self._learnable:
      mus = hk.get_parameter(
          'mean_field_mu', (len(self._flat_params),),
          init=hk.initializers.Constant(self._flat_params))
      stddevs = hk.get_parameter(
          'mean_field_stddev', (len(self._flat_params),),
          init=hk.initializers.UniformScaling(self._stddev))
      empirical_stddev = self._flat_params.std()
      init = jnp.log(jnp.exp(empirical_stddev) - 1.0)
    else:
      mus = jnp.zeros_like(self._flat_params)
      stddevs = jnp.ones_like(self._flat_params) * self._stddev
      init = jnp.log(jnp.exp(self._stddev) - 1.0)
    # Add a bias to softplus such that a zero value to the stddevs parameter
    # gives the empirical standard deviation.
    stddevs = jnn.softplus(stddevs + init) + 1e-6
    return tfd.MultivariateNormalDiag(mus, stddevs)

  def sample(self):
    params_vec = self.__call__().sample(seed=hk.next_rng_key())
    return self._params_tree.unflatten(params_vec)


class BNN(hk.Module):

  def __init__(self, apply_fn: Callable, params: hk.Params,
               posterior_stddev: float, prior_stddev: float):
    super(BNN, self).__init__()

    self.forward = apply_fn
    self.posterior = MeanField('posterior', params, posterior_stddev)
    self.prior = MeanField('prior', params, prior_stddev, learnable=False)

  def __call__(self, x: jnp.ndarray):
    params = self.posterior.sample()
    return self.forward(params, x)

  def posterior(self):
    return self.posterior()

  def prior(self):
    return self.prior()


class BayesByBackprop:

  def __init__(self, example: jnp.ndarray, samples: int, model: Callable):
    self.keys = hk.PRNGSequence(jax.random.PRNGKey(42))
    self.samples = samples
    init_key = next(self.keys)

    def bayes_net():
      net = hk.without_apply_rng(hk.transform(lambda x: model(x)))
      model_params = net.init(init_key, example)
      bayes_net_ = BNN(net.apply, model_params, 1.0, 1.0)

      def init():
        return bayes_net_.posterior(), bayes_net_.prior()

      def call(x):
        return bayes_net_(x)

      def posterior():
        return bayes_net_.posterior()

      def prior():
        return bayes_net_.prior()

      return init, (call, posterior, prior)

    bnn = hk.multi_transform(bayes_net)
    self.bnn = bnn.apply
    self.params = bnn.init(next(self.keys))

  def update_step(self, params, key, x, y):
    grads = self._grad_step(params, key, x, y)
    return grads

  @functools.partial(jax.jit, static_argnums=0)
  def _grad_step(self, params, key, x, y):
    forward, posterior, prior = self.bnn

    def elbo(params):
      keys = jnp.asarray(jax.random.split(key, self.samples))
      mu, raw_stddev = jax.vmap(lambda key: forward(params, key, x))(keys)
      dist = tfd.Normal(mu, utils.get_stddev(raw_stddev))
      log_likelihood = dist.log_prob(y).mean()
      kl = tfd.kl_divergence(posterior(params, None), prior(params,
                                                            None)).mean()
      return -log_likelihood + kl * 1.0 / x.shape[0]

    return jax.grad(elbo)(params)

  def predict(self, x):
    return self._predict(self.keys.take(self.samples), self.params, x)

  @functools.partial(jax.jit, static_argnums=0)
  def _predict(self, keys, params, x):
    forward, *_ = self.bnn
    forward = jax.vmap(functools.partial(forward, params, x=x))
    mus, raw_stddevs = forward(jnp.asarray(keys))
    return utils.to_list_preds(mus, utils.get_stddev(raw_stddevs))
