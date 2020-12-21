from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.internal import test_util
from gemlib.distributions.uniform_integer import UniformInteger


@test_util.test_all_tf_execution_regimes
class TestUniformInteger(test_util.TestCase):
    def setUp(self):
        self.seed = 10402302
        self.fixture = [[8, 4, 8],
                        [2, 7, 9],
                        [6, 0, 9]]

    def test_sample_n_int32(self):
        """Sample returning dtype int32."""
        tf.random.set_seed(self.seed)
        X = UniformInteger(0, 10)
        x = X.sample([3, 3], seed=1)
        x_ = self.evaluate(x)
        self.fixture_ = self.evaluate(tf.convert_to_tensor(self.fixture))
        self.assertAllEqual(self.fixture, x_)
        self.assertDTypeEqual(x_, np.int32)

    def test_sample_n_int64(self):
        """Sample returning int64."""
        tf.random.set_seed(self.seed)
        X = UniformInteger(0, 10, dtype=tf.int64)
        x = X.sample([3, 3], seed=1)
        x_ = self.evaluate(x)
        self.fixture_ = self.evaluate(tf.convert_to_tensor(self.fixture))
        self.assertAllEqual(self.fixture_, x_)
        self.assertDTypeEqual(x_, np.int64)

    def test_log_prob_float32(self):
        """log_prob returning float32."""
        X = UniformInteger(0, 10)
        lp = X.log_prob(self.fixture)
        self.assertSequenceEqual(lp.shape, [3, 3])
        lp_ = self.evaluate(lp)
        self.assertAlmostEqual(np.sum(lp_), -20.723265, places=5)
        self.assertDTypeEqual(lp_, np.float32)

    def test_log_prob_float64(self):
        """log_prob returning float64."""
        X = UniformInteger(0, 10, float_dtype=tf.float64)
        lp = X.log_prob(self.fixture)
        self.assertSequenceEqual(lp.shape, [3, 3])
        lp_ = self.evaluate(lp)
        self.assertAlmostEqual(np.sum(lp_), -20.723265, places=5)
        self.assertDTypeEqual(lp_, np.float64)


if __name__ == '__main__':
    tf.test.main()
