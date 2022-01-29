import unittest

import numpy as np

from f_povi import FunctionalParticleOptimization
from utils import net


class TestFunctionalPovi(unittest.TestCase):
  def setUp(self) -> None:
    self.x = np.array([[0.0, 0.0],
                       [1.0, 1.0],
                       [2.0, 2.0]])
    self.y = np.array([1.0, 2.0, 3.0])
    self.particles = 20
    self.povi = FunctionalParticleOptimization(self.x, self.particles, net)

  def test_grad_shape(self):
    grads = self.povi.grad_step(self.x, self.y)
    # Hardcode 2, since 'net' in utils has two paramters.
    self.assertEqual(grads.shape, (self.particles, 2))
