import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal.tensor_util import (
    convert_nonref_to_tensor,
)
from tensorflow_probability.python.internal import tensorshape_util

tfd = tfp.distributions


def _log_choose(N, k):
    return (
        tf.math.lgamma(N + 1.0)
        - tf.math.lgamma(k + 1.0)
        - tf.math.lgamma(N - k + 1.0)
    )


class UniformKCategorical(tfd.Distribution):
    def __init__(
        self,
        k,
        mask,
        float_dtype=tf.float32,
        validate_args=False,
        allow_nan_stats=True,
        name="UniformKCategorical",
    ):
        """Uniform K-Categorical distribution.

        Given a set of items indexed $1,...,n$ and a boolean mask of the same shape
        sample $k$ indices without replacement.
        :param k: the number of indices to sample
        :param mask: a boolean mask with True where an element is valid, otherwise False
        :param validate_args: Whether to validate args
        :param allow_nan_stats: allow nan stats
        :param name: name of the distribution

        Example 1: Generate 4 samples of size k given a mask
            import numpy as np
            import tensorflow as tf
            import tensorflow_probability as tfp
            from gemlib.distributions.kcategorical import UniformKCategorical

            # Mask determines which indices are valid and so returned by sample().
            # Below combinations of the indices 0, 3, 4, and 6 will be realised when sampling.
            mask = [True, False, False, True, True, False, True]
            X = UniformKCategorical(k=3, mask=mask)
            x = X.sample(4)
            tf.print(x)

        Example 2: Probability of a given sample
            import numpy as np
            import tensorflow as tf
            import tensorflow_probability as tfp
            from gemlib.distributions.kcategorical import UniformKCategorical

            # Probability of drawing an unordered sample of
            # size k from N items where N equals the number
            # of True states in the mask determines N.
            mask = [True, False, False, True, True, False, True]
            sample = tf.convert_to_tensor([4, 3, 6])
            X = UniformKCategorical(k=sample.shape[-1], mask=mask)
            lp = X.log_prob(sample)
            tf.print('prob:', tf.exp(lp))

        """
        parameters = dict(locals())
        self._mask = convert_nonref_to_tensor(mask, dtype_hint=tf.bool)
        self._k = convert_nonref_to_tensor(k, dtype_hint=tf.int32)
        self._float_dtype = float_dtype
        dtype = self._k.dtype

        with tf.name_scope(name) as name:
            super(UniformKCategorical, self).__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name,
            )

    def _batch_shape(self):
        return tf.TensorShape(self._mask.shape[:-1])

    def _event_shape(self):
        return tensorshape_util.constant_value_as_shape(
            tf.expand_dims(self._k, axis=0)
        )

    def _sample_n(self, n, seed=None):
        u = tfd.Uniform(
            low=tf.zeros(self._mask.shape, dtype=tf.float32),
            high=tf.ones(self._mask.shape, dtype=tf.float32),
        ).sample(n, seed=seed)
        u = u * tf.cast(self._mask, u.dtype)
        _, x = tf.math.top_k(u, k=self._k, sorted=True)
        return x

    def _log_prob(self, x):
        N = tf.math.count_nonzero(self._mask, axis=-1)
        return -_log_choose(
            tf.cast(N, dtype=self._float_dtype),
            tf.cast(self._k, dtype=self._float_dtype),
        )
