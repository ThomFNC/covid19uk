from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.internal import test_util
from gemlib.distributions.kcategorical import UniformKCategorical


@test_util.test_all_tf_execution_regimes
class TestUniformInteger(test_util.TestCase):
    def setUp(self):
        self.seed = 10402302
        self.mask = [True, False, False, True, True, False, True]

    def test_sample(self):
        """Sample draws one sample with shape (1,3) ."""
        tf.random.set_seed(self.seed)
        target = tf.convert_to_tensor([[0, 6, 3]])
        X = UniformKCategorical(k=target.shape[-1], mask=self.mask)
        x = X.sample(1, seed=1234)
        x_ = self.evaluate(x)
        target_ = self.evaluate(target)
        self.assertAllEqual(target_, x_)
        self.assertDTypeEqual(x_, np.int32)

    def test_log_prob_float32(self):
        """Log probability of 1 realisations using float32."""
        target = tf.convert_to_tensor([[0, 6, 3]])
        X = UniformKCategorical(k=target.shape[-1], mask=self.mask, float_dtype=tf.float32)
        lp = X.log_prob(target)
        lp_ = self.evaluate(lp)
        print('32', lp)
        self.assertAlmostEqual(lp_, -1.3862944, places=5)
        self.assertDTypeEqual(lp_, np.float32)

    def test_log_prob_float64(self):
        """Log probability of 1 realisations using float32."""
        target = tf.convert_to_tensor([[0, 6, 3]])
        X = UniformKCategorical(k=target.shape[-1], mask=self.mask, float_dtype=tf.float64)
        lp = X.log_prob(target)
        lp_ = self.evaluate(lp)
        print('64', lp)
        self.assertAlmostEqual(lp_, -1.3862944, places=5)
        self.assertDTypeEqual(lp_, np.float64)


if __name__ == '__main__':
    tf.test.main()
