import functools
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import utils

tfd = tfp.distributions


class FunctionalParticleOptimization:

  def __init__(self, example: jnp.ndarray, n_particles: int, model: Callable):
    net = hk.without_apply_rng(hk.transform(lambda x: model(x)))
    self.net = jax.vmap(net.apply, (0, None))
    init = jax.vmap(net.init, (0, None))
    seed_sequence = hk.PRNGSequence(666)
    self.particles = init(jnp.asarray(seed_sequence.take(n_particles)), example)
    self.priors = init(jnp.asarray(seed_sequence.take(n_particles)), example)

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
  def _grad_step(self, particles, x, y):
    # dy_dtheta_vjp allow us to compute the jacobian-transpose times the
    # vector of # the stein operators for each particle. See eq.4 in Wang et
    # al. (2019) https://arxiv.org/abs/1902.09754.
    predictions, dy_dtheta_vjp = jax.vjp(lambda p: self.net(p, x), particles)
    # TODO (yarden): batch size as leftmost axis.
    predictions = jnp.asarray(predictions).transpose(1, 2, 0)
    prior = self._prior(x)

    def log_joint(predictions):
      dist = tfd.Normal(predictions[:, 0], predictions[:, 1])
      log_likelihood = dist.log_prob(y).mean()
      log_prior = prior.log_prob(predictions).mean()
      return log_likelihood + log_prior

    # The predictions are independent of the evidence (a.k.a. normalization
    # factor), so the gradients of the log-posterior equal to those of the
    # log-joint.
    log_posterior_grad = jax.vmap(jax.grad(log_joint))(predictions)
    # Batch size as leading dimension.
    log_posterior_grad = log_posterior_grad.transpose(1, 0, 2)
    tmp_preds = predictions.transpose(1, 0, 2)
    kxy, kernel_vjp = jax.vjp(lambda x: rbf_kernel(x, tmp_preds), tmp_preds)
    # Summing along the 'particles axis'.
    # See https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    # and eq. 8 in Liu et al. (2016) https://arxiv.org/abs/1608.04471.
    dkxy_dx = kernel_vjp(-jnp.ones(kxy.shape))[0]
    stein_grads = ((jnp.matmul(kxy, log_posterior_grad) + dkxy_dx) /
                   len(self.particles))
    stein_grads = stein_grads.transpose(1, 0, 2)
    return dy_dtheta_vjp((-stein_grads[..., 0], -stein_grads[..., 1]))[0]

  def _prior(self, x):
    predictions = self.net(self.priors, x)
    predictions = jnp.asarray(predictions).transpose(1, 2, 0)
    mean = predictions.mean(0)
    cov = tfp.stats.cholesky_covariance(predictions)
    return tfd.MultivariateNormalTriL(mean, cov)

  @functools.partial(jax.jit, static_argnums=0)
  def predict(self, x):
    mus, stddevs = self.net(self.particles, x)
    return utils.to_list_preds(mus, stddevs)


# Based on tf-probability implementation of batched pairwise matrices:
# https://github.com/tensorflow/probability/blob
# /f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python
# /math/psd_kernels/internal/util.py#L190
def rbf_kernel(x, y):
  row_norm_x = (x**2).sum(-1)[..., None]
  row_norm_y = (y**2).sum(-1)[..., None, :]
  pairwise = row_norm_x + row_norm_y - 2. * jnp.matmul(x, y.transpose(0, 2, 1))
  jnp.clip(pairwise, 0., pairwise)
  n_x = pairwise.shape[-2]
  bandwidth = jnp.median(pairwise)
  bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
  bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
  k_xy = jnp.exp(-pairwise / bandwidth / 2)
  return k_xy
