import numpy as np
import seaborn as sns

import haiku as hk

from tensorflow_probability.substrates import jax as tfp

sns.reset_defaults()
# sns.set_style('whitegrid')
# sns.set_context('talk')
sns.set_context(context='talk', font_scale=0.7)

tfd = tfp.distributions


def load_dataset(x_range, n=150, n_tst=150):
    np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g ** 2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    x = x[..., np.newaxis]
    x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
    x_tst = x_tst[..., np.newaxis]
    return y, x, x_tst


def main():
    w0 = 0.125
    b0 = 5.
    x_range = [-20, 60]
    y, x, x_tst = load_dataset(x_range)
