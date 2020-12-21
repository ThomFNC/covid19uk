"""Describes a State Transition Model with marginalised baseline
hazard rates.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization

from gemlib.util import batch_gather, transition_coords, compute_state
from gemlib.distributions.discrete_markov import discrete_markov_simulation

tla = tf.linalg
tfd = tfp.distributions


class StateTransitionMarginalModel(tfd.Distribution):
    def __init__(
        self,
        transition_rates,
        baseline_hazard_rate_priors,
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
        :param baseline_hazard_rate_priors: a dictionary of `concentration` and `rate`
                                            hyperparameters for implicit Gamma priors
                                            on (marginalised) baseline hazard rates.
                                            Both `concentration` and `rate` should
                                            broadcast with the number of rows in
                                            `stoichiometry`.
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
                stoichiometry, dtype=initial_state.dtype,
            )
            self._initial_state = initial_state
            self._initial_step = initial_step
            self._time_delta = time_delta
            self._num_steps = num_steps
            self._baseline_hazard_rate_priors = baseline_hazard_rate_priors

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
        with tf.name_scope("StateTransitionMarginalModel.log_prob"):

            state_timeseries = compute_state(
                initial_state=self.initial_state,
                events=events,
                stoichiometry=self.stoichiometry,
                closed=True,
            )

            tms_timeseries = tf.transpose(state_timeseries, perm=(1, 0, 2))
            tmr_events = tf.transpose(events, perm=(1, 0, 2))

            def fn(elems):
                return tf.stack(self.transition_rates(*elems), axis=-1)

            rates = tf.vectorized_map(
                fn=fn,
                elems=(
                    self._initial_step + tf.range(tms_timeseries.shape[0]),
                    tms_timeseries,
                ),
            )

            def integrated_rate_fn():
                """Use mid-point integration to estimate the constant rate
                over time.
                """
                integrated_rates = tms_timeseries[..., :-1] * rates
                return (integrated_rates[:-1, ...] + integrated_rates[1:, ...]) / 2.0

            integrated_rates = integrated_rate_fn()

            log_norm_constant = tf.reduce_sum(
                tf.math.multiply_no_nan(tf.math.log(integrated_rates), tmr_events)
                - tf.math.lgamma(tmr_events + 1.0),
                axis=(0, 1),
            )
            pi_concentration = (
                tf.reduce_sum(tmr_events, axis=(0, 1))
                + self.baseline_hazard_rate_priors["concentration"]
            )
            pi_rate = (
                tf.reduce_sum(integrated_rates * self.time_delta, axis=(0, 1))
                + self.baseline_hazard_rate_priors["rate"]
            )

            log_prob = (
                log_norm_constant
                + tf.math.lgamma(pi_concentration)
                - (pi_concentration) * tf.math.log(pi_rate)
            )

            return tf.reduce_sum(log_prob)


class BaselineHazardRateMarginal(tfd.Distribution):
    def __init__(
        self,
        events,
        transition_rate_fn,
        baseline_hazard_rate_priors,
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

        :param events: a [M, T, R] event tensor
        :param transition_rates: a function of the form `fn(t, state)` taking the
                                 current time `t` and state tensor `state`.  This
                                 function returns a tensor which broadcasts to the first
                                 dimension of `stoichiometry`.  Transition rates are
                                 assumed to be risk ratios, with the baseline hazard
                                 rate marginalised out from the model.
        :param baseline_hazard_rate_priors: a dictionary of `concentration` and `rate`
                                            hyperparameters for implicit Gamma priors
                                            on (marginalised) baseline hazard rates.
                                            Both `concentration` and `rate` should
                                            broadcast with the number of rows in
                                            `stoichiometry`.
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
            self._events = events
            self._transition_rate_fn = transition_rate_fn
            self._stoichiometry = tf.convert_to_tensor(
                stoichiometry, dtype=initial_state.dtype,
            )
            self._initial_state = initial_state
            self._initial_step = initial_step
            self._time_delta = time_delta
            self._num_steps = num_steps
            self._baseline_hazard_rate_priors = baseline_hazard_rate_priors

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
    def baseline_hazard_rate_priors(self):
        return self._baseline_hazard_rate_priors

    def _batch_shape(self):
        return tf.TensorShape(())

    def _event_shape(self):
        shape = tf.TensorShape(self._events.shape[-1])
        return shape

    def concentration(self):
        """Calculates the concentration parameter"""
        return (
            tf.reduce_sum(self._events, axis=(0, 1))
            + self.baseline_hazard_rate_priors["concentration"]
        )

    def rate(self):
        """Calculates the rate parameter"""
        state = compute_state(
            self._initial_state, self._events, self._stoichiometry, closed=True
        )
        tms_state = tf.transpose(state, perm=(1, 0, 2))

        def fn(elems):
            return tf.stack(self._transition_rate_fn(*elems), axis=-1)

        rates = tf.vectorized_map(
            fn=fn,
            elems=(self._initial_step + tf.range(tms_state.shape[0]), tms_state,),
        )

        def integrated_rate_fn():
            """Use mid-point integration to estimate the constant rate
            over time.
            """
            integrated_rates = tms_state[..., :-1] * rates
            return (integrated_rates[:-1, ...] + integrated_rates[1:, ...]) / 2.0

        return (
            tf.reduce_sum(integrated_rate_fn(), axis=(-3, -2))
            + self._baseline_hazard_rate_priors["rate"]
        )

    def _sample_n(self, n, seed=None):
        tf.print("Concentration:", self.concentration())
        tf.print("Rate:", self.rate())
        rv = tfd.Gamma(concentration=self.concentration(), rate=self.rate())
        return rv.sample(n, seed=seed)

    def _log_prob(self, y, **kwargs):
        """Calculates the log prob"""
        rv = tfd.Gamma(concentration=self.concentration(), rate=self.rate())
        return tf.reduce_sum(rv.log_prob(y))

    def _mean(self):
        rv = tfd.Gamma(concentration=self.concentration(), rate=self.rate())
        return rv.mean()
