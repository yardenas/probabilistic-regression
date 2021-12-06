from typing import Callable

import functools

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn

import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class MeanField(hk.Module):
    def __init__(
            self,
            params: hk.Params,
            stddev=1.0,
            fixed_stddev=False
    ):
        super(MeanField, self).__init__()
        flat_ps, tree_def = jax.tree_flatten(params)
        self._flattened_params = flat_ps
        self._tree_def = tree_def
        self._stddev = stddev
        self._fixed_stddev = fixed_stddev

    def __call__(self):
        flat_params = jax.tree_map(jnp.ravel, self._flattened_params)
        flat_params = jnp.concatenate(flat_params)
        mus = hk.get_parameter(
            'mean_field_mu', (len(flat_params),),
            init=hk.initializers.Constant(flat_params)
        )
        if self._fixed_stddev:
            stddevs = jnp.ones_like(flat_params)
        else:
            stddevs = hk.get_parameter(
                'mean_field_stddev', (len(flat_params),),
                init=hk.initializers.RandomUniform(0, self._stddev)
            )
        stddevs = jnn.softplus(stddevs) + 1e-3
        return tfd.MultivariateNormalDiag(mus, stddevs)

    @functools.partial(jax.jit, static_argnums=0)
    def unflatten(self, sample):
        out = []
        i = 0
        for p in self._flattened_params:
            n = p.size
            out.append(sample[i:i + n].reshape(p.shape))
            i += n
        return jax.tree_unflatten(self._tree_def, out)


class BNN(hk.Module):
    def __init__(
            self,
            apply_fn: Callable,
            params: hk.Params,
            posterior_samples: int,
            stddev: float
    ):
        super(BNN, self).__init__()

        self.forward = apply_fn
        self.posterior = MeanField(params, stddev)
        self.prior = MeanField(params, stddev, fixed_stddev=True)
        self._samples = posterior_samples

    def __call__(self, x: jnp.ndarray):
        return self._wrap(self.forward)(x)

    def kl(self):
        return tfd.kl_divergence(self.posterior(), self.prior()).mean()

    def _wrap(self, f):
        def wrapped(*args, **kwargs):
            # Define a function for v-map
            def apply(key: jnp.ndarray, *args, **kwargs):
                sampled_params = self.posterior.unflatten(
                    self.posterior().sample(seed=key)
                )
                return f(sampled_params, *args, **kwargs)

            keys = hk.next_rng_keys(self._samples)
            return jax.vmap(lambda key: apply(key, *args, *kwargs), (0,))(keys)

        return wrapped
