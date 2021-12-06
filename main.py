import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from tensorflow_probability.substrates import jax as tfp

from bnn import BNN

sns.reset_defaults()
sns.set_context(context='talk', font_scale=0.7)

tfd = tfp.distributions


def net(x):
    x = hk.nets.MLP((2,))(x)
    mu, stddev = jnp.split(x, 2, -1)
    mu = mu.squeeze(-1)
    stddev = stddev.squeeze(-1)
    stddev = jnn.softplus(stddev) + 1e-3
    return mu, stddev


def load_dataset(x_range, b0, w0, n=150, n_tst=150):
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


def dataset(x, y, batch_size):
    ids = np.arange(len(x))
    while True:
        ids = np.random.choice(ids, batch_size, False)
        yield x[ids], y[ids]


def plot(x_range, x, y, x_tst, yhats):
    plt.figure(figsize=[12, 3.0])  # inches
    plt.plot(x, y, 'b.', label='observed')

    avgm = np.zeros_like(x_tst[..., 0])
    for i, yhat in enumerate(yhats):
        m = np.squeeze(yhat.mean())
        s = np.squeeze(yhat.stddev())
        if i < 15:
            plt.plot(x_tst, m, 'r', label='ensemble means' if i == 0 else None,
                     linewidth=1.)
            plt.plot(x_tst, m + 2 * s, 'g', linewidth=0.5,
                     label='ensemble means + 2 ensemble stdev' if i == 0 else
                     None)
            plt.plot(x_tst, m - 2 * s, 'g', linewidth=0.5,
                     label='ensemble means - 2 ensemble stdev' if i == 0 else
                     None)
        avgm += m
    plt.plot(x_tst, avgm / len(yhats), 'r', label='overall mean', linewidth=4)
    plt.ylim(-0., 17)
    plt.yticks(np.linspace(0, 15, 4)[1:])
    plt.xticks(np.linspace(*x_range, num=9))
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='center left', fancybox=True, framealpha=0.,
               bbox_to_anchor=(0.95, 0.5))
    plt.tight_layout()
    plt.show()


def main():
    w0 = 0.125
    b0 = 5.
    x_range = [-20, 60]
    batch_size = 32
    y, x, x_tst = load_dataset(x_range, b0, w0)
    data = iter(dataset(x, y, batch_size))
    mlp = hk.without_apply_rng(hk.transform(lambda x: net(x)))
    keys = hk.PRNGSequence(jax.random.PRNGKey(42))
    net_params = mlp.init(next(keys), x[:batch_size])

    def bayes_net():
        bayes_net_ = BNN(mlp.apply, net_params, 100, 1.0)

        def init():
            return bayes_net_.kl()

        def call(x):
            return bayes_net_(x)

        def kl():
            return bayes_net_.kl()

        return init, (call, kl)

    bnn = hk.multi_transform(bayes_net)
    opt = optax.adam(0.01)

    forward, kl = bnn.apply

    def predict(params, key, x):
        mu, stddev = forward(params, key, x)
        return tfd.Normal(mu.mean(0), stddev.mean(0))

    def elbo(params, key, x, y):
        log_likelihood = predict(params, key, x).log_prob(y).mean()
        return -log_likelihood + 0.1 / x.shape[0] * kl(params, None)

    @jax.jit
    def update(params, key, opt_state, x, y):
        grads = jax.grad(elbo)(params, key, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    params = bnn.init(next(keys))
    opt_state = opt.init(params)

    for step in range(10000):
        params, opt_state = update(params, next(keys), opt_state, *next(data))

    mus, stddevs = forward(params, next(keys), x_tst)
    yhats = list(map(lambda mu, stddev: tfd.Normal(mu, stddev), mus, stddevs))
    plot(x_range, x, y, x_tst, yhats)


if __name__ == '__main__':
    main()
