from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from gemlib.mcmc.adaptive_random_walk_metropolis import AdaptiveRandomWalkMetropolis


@test_util.test_all_tf_execution_regimes
class TestAdaptiveRandomWalkMetropolis(test_util.TestCase):

    def test_1d_normal(self):
        """Sample from Standard Normal Distribution."""
        dtype = np.float32

        target = tfd.Normal(loc=dtype(0), scale=dtype(1))

        kernel = AdaptiveRandomWalkMetropolis(
            target_log_prob_fn=target.log_prob,
            target_accept_ratio=0.44,
            initial_covariance=dtype(0.001)
        )
        samples = tfp.mcmc.sample_chain(
            num_results=2000,
            current_state=dtype([0.1]),
            kernel=kernel,
            num_burnin_steps=500,
            trace_fn=None)

        sample_mean = tf.math.reduce_mean(samples, axis=0)
        sample_std = tf.math.reduce_std(samples, axis=0)
        [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

        self.assertAllClose([0.], sample_mean_, atol=0.17, rtol=0.)
        self.assertAllClose([1.], sample_std_, atol=0.2, rtol=0.)

    def test_3d_mvn(self):
        """Sample from 3-variate Gaussian Distribution."""
        dtype = np.float32

        true_mean = dtype([1., 2., 3.])
        true_cov = dtype([[0.36, 0.12, 0.06],
                          [0.12, 0.29, -0.13],
                          [0.06, -0.13, 0.26]])
        target = tfd.MultivariateNormalTriL(
            loc=true_mean,
            scale_tril=tf.linalg.cholesky(true_cov)
        )
        kernel = AdaptiveRandomWalkMetropolis(
            target_log_prob_fn=target.log_prob,
            initial_covariance=dtype(0.001) * np.eye(3, dtype=dtype)
        )
        samples = tfp.mcmc.sample_chain(
            num_results=2000,
            current_state=dtype([0.1, 0.1, 0.1]),
            kernel=kernel,
            num_burnin_steps=500,
            trace_fn=None)

        sample_mean = tf.math.reduce_mean(samples, axis=0)
        [sample_mean_] = self.evaluate([sample_mean])
        self.assertAllClose(sample_mean_, true_mean, atol=0.1, rtol=0.1)

        sample_cov = tfp.stats.covariance(samples)
        sample_cov_ = self.evaluate(sample_cov)
        self.assertAllClose(sample_cov_, true_cov, atol=0.1, rtol=0.1)

    def test_float64(self):
        """Sample with dtype float64."""
        dtype = np.float64

        target = tfd.Normal(loc=dtype(0), scale=dtype(1))

        kernel = AdaptiveRandomWalkMetropolis(
            target_log_prob_fn=target.log_prob,
            target_accept_ratio=0.44,
            initial_covariance=dtype(0.001)
        )
        samples = tfp.mcmc.sample_chain(
            num_results=20,
            current_state=dtype([0.1]),
            kernel=kernel,
            trace_fn=None)

    def test_bijector(self):
        """Employ bijector when sampling."""
        dtype = np.float32

        target = tfd.Normal(loc=dtype(0), scale=dtype(1))

        kernel = AdaptiveRandomWalkMetropolis(
            target_log_prob_fn=target.log_prob,
            target_accept_ratio=0.44,
            initial_covariance=dtype(0.001)
        )
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=kernel,
            bijector=tfp.bijectors.Exp()
        )
        samples = tfp.mcmc.sample_chain(
            num_results=20,
            current_state=dtype([0.1]),
            kernel=kernel,
            trace_fn=None)

    def test_is_calibrated(self):
        dtype = np.float32
        kernel = AdaptiveRandomWalkMetropolis(
            target_log_prob_fn=lambda x: -tf.square(x) / 2.,
            initial_covariance=dtype(0.001)
        )
        self.assertTrue(kernel.is_calibrated)


# *** RUN TESTS ***

# ** Method 1 **
# When using tf.test.main() from within jupyter notebook use
#   import sys
#   sys.argv = sys.argv[:1]
# to eliminate the ERROR "unknown command line flag 'f" 
# if __name__ == '__main__':
#    try:
#        tf.test.main()
#    except SystemExit as inst:
#        if inst.args[0] is True: # raised by sys.exit(True) if tests fail
#            raise

# ** Method 2 **
# Althernatively use unittest.main() instead of tf.test.main()
# when running tests within jupyter notebook:
#   import unittest
#   unittest.main(argv=['first-arg-is-ignored'], exit=False)

# ** Method 3 **
# If not using jupyter notebook (or similar environment) the
# following should work:
if __name__ == '__main__':
    tf.test.main()
