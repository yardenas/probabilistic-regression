import haiku as hk
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import optax
import functools
from tensorflow_probability.substrates import jax as tfp

import utils
from f_povi import FunctionalParticleOptimization
from bayes_by_backprop import BayesByBackprop

tfd = tfp.distributions


def load_dataset(x_range, b0, w0, n=150, n_tst=150):
  np.random.seed(43)

  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)

  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1 + np.sin(x)) + b0) + eps
  y = (y - y.mean()) / y.std()
  xm, xs = x.mean(), x.std()
  x = x[..., None]
  x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
  x_tst = x_tst[..., None]

  def normalize(x):
    return (x - xm) / xs

  return (y.astype(np.float32), normalize(x).astype(np.float32),
          normalize(x_tst).astype(np.float32))


def dataset(x, y, batch_size):
  ids = np.arange(len(x))
  while True:
    ids = np.random.choice(ids, batch_size, False)
    yield x[ids], y[ids]


def plot(x, y, x_tst, yhats):
  plt.figure(figsize=[12, 3.0])  # inches
  plt.plot(x, y, 'b.', label='observed')
  avgm = np.zeros_like(x_tst[..., 0])
  for i, yhat in enumerate(yhats):
    m = np.squeeze(yhat.mean())
    s = np.squeeze(yhat.stddev())
    if i < 15:
      plt.plot(
          x_tst,
          m,
          'r',
          label='ensemble means' if i == 0 else None,
          linewidth=1.)
      plt.plot(
          x_tst,
          m + 3 * s,
          'g',
          linewidth=0.5,
          label='ensemble means + 3 ensemble stdev' if i == 0 else None)
      plt.plot(
          x_tst,
          m - 3 * s,
          'g',
          linewidth=0.5,
          label='ensemble means - 3 ensemble stdev' if i == 0 else None)
    avgm += m
  plt.plot(x_tst, avgm / len(yhats), 'r', label='overall mean', linewidth=4)
  ax = plt.gca()
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')
  ax.spines['left'].set_position(('data', 0))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend(
      loc='center left',
      fancybox=True,
      framealpha=0.,
      bbox_to_anchor=(0.95, 0.5))
  plt.tight_layout()
  plt.show()


def main():
  w0 = 0.125
  b0 = 0.
  x_range = [-20, 60]
  batch_size = 32
  y, x, x_tst = load_dataset(x_range, b0, w0)
  data = iter(dataset(x, y, batch_size))
  net_fn = functools.partial(
      utils.net, n_layers=2, n_hidden=50, init_stddev=0.001, sd_scale=0.01)
  model = BayesByBackprop(x[:batch_size], 20, net_fn)
  opt = optax.flatten(optax.adam(1e-3))
  keys = hk.PRNGSequence(jax.random.PRNGKey(42))

  @jax.jit
  def update(x, y, params, key, opt_state):
    grads = model.update_step(params, key, x, y)
    updates, new_opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  opt_state = opt.init(model.params)

  for step in range(20000):
    model.params, opt_state = update(*next(data), model.params, next(keys),
                                     opt_state)
  yhats = model.predict(x_tst)
  plot(x, y, x_tst, yhats)


if __name__ == '__main__':
  main()
