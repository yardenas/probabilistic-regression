from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from dreamer.rssm import State, Action, Observation

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
        flat_params = jnp.asarray(self._flattened_params)
        mus = hk.get_parameter(
            'mean_field_mu', (len(self._flattened_params),),
            init=hk.initializers.Constant(flat_params)
        )
        if self._fixed_stddev:
            stddevs = np.ones_like(flat_params)
        else:
            stddevs = hk.get_parameter(
                'mean_field_stddev', (len(self._flattened_params),),
                init=hk.initializers.RandomUniform(0, self._stddev)
            )
        return tfd.MultivariateNormalDiag(mus, stddevs)

    def unflatten(self, sample):
        return jax.tree_unflatten(self._tree_def, sample)


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

    def __call__(self, prev_state: State, prev_action: Action,
                 observation: Observation
                 ) -> Tuple[Tuple[tfd.MultivariateNormalDiag,
                                  tfd.MultivariateNormalDiag],
                            State]:
        return self._wrap(self.forward)(prev_state, prev_action, observation)

    def kl(self):
        return tfd.kl_divergence(self.posterior(), self.prior()).mean()

    def _wrap(self, f):
        def wrapped(*args, **kwargs):
            # Define a function for v-map
            def apply(key: jnp.ndarray, *args, **kwargs):
                params = self.posterior.unflatten(
                    self.posterior().sample(seed=key)
                )
                key = hk.next_rng_key()
                return f(params, key, *args, **kwargs)

            keys = hk.next_rng_keys(self._samples)
            return jax.vmap(lambda key: apply(key, *args, *kwargs), (0,))(keys)

        return wrapped
