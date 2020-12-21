"""Describes a continuous time State Transition Model with discrete event time
approximation.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization

from gemlib.util import batch_gather, transition_coords, compute_state
from gemlib.distributions.discrete_markov import discrete_markov_simulation

tla = tf.linalg
tfd = tfp.distributions


class DiscreteApproxContStateTransitionModel(tfd.Distribution):
    def __init__(
        self,
        transition_rates,
        stoichiometry,
        initial_state,
        initial_step,
        time_delta,
        num_steps,
        validate_args=False,
        allow_nan_stats=True,
        name="StateTransitionMarginalModel",
    ):
        """Implements a discrete-time Markov jump process for a state transition model.

        :param transition_rates: a function of the form `fn(t, state)` taking the
                                 current time `t` and state tensor `state`.  This
                                 function returns a tensor which broadcasts to the first
                                 dimension of `stoichiometry`.  Transition rates are
                                 assumed to be risk ratios, with the baseline hazard
                                 rate marginalised out from the model.
        :param stoichiometry: the stochiometry matrix for the state transition model,
                              with rows representing transitions and columns
                              representing states.
        :param initial_state: an initial state tensor with inner dimension equal to the
                              first dimension of `stoichiometry`.
        :param initial_step: an offset giving the time `t` of the first timestep in the
                             model.
        :param time_delta: the size of the time step to be used.
        :param num_steps: the number of time steps across which the model runs.
        """

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._transition_rates = transition_rates
            self._stoichiometry = tf.convert_to_tensor(
                stoichiometry, dtype=initial_state.dtype
            )
            self._initial_state = initial_state
            self._initial_step = initial_step
            self._time_delta = time_delta
            self._num_steps = num_steps

            super().__init__(
                dtype=initial_state.dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name,
            )

        self.dtype = initial_state.dtype

    @property
    def transition_rates(self):
        return self._transition_rates

    @property
    def baseline_hazard_rate_priors(self):
        return self._baseline_hazard_rate_priors

    @property
    def stoichiometry(self):
        return self._stoichiometry

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def initial_step(self):
        return self._initial_step

    @property
    def time_delta(self):
        return self._time_delta

    @property
    def num_steps(self):
        return self._num_steps

    def _batch_shape(self):
        return tf.TensorShape([])

    def _event_shape(self):
        shape = tf.TensorShape(
            [
                self.initial_state.shape[0],
                tf.get_static_value(self._num_steps),
                self._stoichiometry.shape[0],
            ]
        )
        return shape

    def _sample_n(self, n, seed=None):
        """Runs a simulation from the epidemic model

        :param param: a dictionary of model parameters
        :param state_init: the initial state
        :returns: a tuple of times and simulated states.
        """
        with tf.name_scope("DiscreteTimeStateTransitionModel.log_prob"):

            def hazard_fn(t, state):
                return self.transition_rates(t, state)

            t, sim = discrete_markov_simulation(
                hazard_fn=hazard_fn,
                state=self.initial_state,
                start=self.initial_step,
                end=self.initial_step + self.num_steps * self.time_delta,
                time_step=self.time_delta,
                stoichiometry=self.stoichiometry,
                seed=seed,
            )
            indices = transition_coords(self.stoichiometry)
            sim = batch_gather(sim, indices)
            sim = tf.transpose(sim, perm=(1, 0, 2))
            return tf.expand_dims(sim, 0)

    def _log_prob(self, y, **kwargs):
        """Calculates the log probability of observing epidemic events y
        :param y: a list of tensors.  The first is of shape [n_times] containing times,
                  the second is of shape [n_times, n_states, n_states] containing event
                  matrices.
        :param param: a list of parameters
        :returns: a scalar giving the log probability of the epidemic
        """
        dtype = dtype_util.common_dtype([y, self.initial_state], dtype_hint=self.dtype)
        events = tf.convert_to_tensor(y, dtype)
        with tf.name_scope("DiscreteApproxContStateTransitionModel.log_prob"):

            state_timeseries = compute_state(
                self.initial_state, events, self.stoichiometry, closed=True,
            )

            tms_timeseries = tf.transpose(state_timeseries, perm=(1, 0, 2))
            tmr_events = tf.transpose(events, perm=(1, 0, 2))

            def fn(elems):
                return tf.stack(self.transition_rates(*elems), axis=-1)

            rates = tf.vectorized_map(
                fn=fn,
                elems=(
                    self.initial_step + tf.range(tms_timeseries.shape[0]),
                    tms_timeseries,
                ),
            )

            def integrated_rates():
                """Use mid-point integration to estimate the constant rate 
                   over time
                """
                integrated_rates = tms_timeseries[..., :-1] * rates
                return (integrated_rates[:-1, ...] + integrated_rates[1:, ...]) / 2.0

            log_hazard_rate = tf.reduce_sum(
                tmr_events * tf.math.log(integrated_rates())
            )
            log_survival = tf.reduce_sum(integrated_rates()) * self.time_delta
            log_denom = tf.reduce_sum(tf.math.lgamma(tmr_events + 1.0))

            return log_hazard_rate - log_survival - log_denom
