import os
import pickle as pkl
import unittest
import numpy as np
import tensorflow as tf

from covid.impl.event_time_proposal import _abscumdiff, EventTimeProposal, \
    FilteredEventTimeProposal, TransitionTopology

_thisdir = os.path.dirname(__file__)
_testsim = os.path.join(_thisdir, 'fixture/stochastic_sim_small.pkl')


class TestAbsCumDiff(unittest.TestCase):
    def setUp(self):
        with open(_testsim, 'rb') as f:
            sim = pkl.load(f)
        self.events = np.stack([sim['events'][..., 0, 1],
                                sim['events'][..., 1, 2],
                                sim['events'][..., 2, 3]], axis=-1)
        self.events = np.transpose(self.events, axes=(1, 0, 2))
        self.initial_state = sim['state_init']

    def test_fwd_state(self):
        t = 19
        target_events = self.events[..., 1].copy()
        n_max = _abscumdiff(events=self.events,
                            initial_state=self.initial_state, target_id=1,
                            bound_t=t, bound_id=2).numpy()
        # Numpy equivalent
        dNt_np = np.absolute(
            np.cumsum(self.events[..., 1] - self.events[..., 2], axis=-1))
        n_max_np = dNt_np[:, np.squeeze(t)] + self.initial_state[:, 2]
        np.testing.assert_array_equal(n_max.flatten(), n_max_np)

    def test_fwd_state_multi(self):
        t = [19, 20, 21]
        target_events = self.events[..., 1].copy()
        n_max = _abscumdiff(events=self.events,
                            initial_state=self.initial_state, target_id=1,
                            bound_t=t, bound_id=2).numpy()
        dNt_np = np.absolute(
            np.cumsum(self.events[..., 1] - self.events[..., 2], axis=-1))
        n_max_np = dNt_np[:, t] + self.initial_state[:, [2]]
        np.testing.assert_array_equal(n_max, n_max_np)

    def test_fwd_state_multi_multi(self):
        t = tf.broadcast_to([19, 20, 21], [10, 3])
        target_events = self.events[..., 1].copy()
        n_max = _abscumdiff(events=self.events,
                            initial_state=self.initial_state, target_id=1,
                            bound_t=t, bound_id=2).numpy()
        dNt_np = np.absolute(
            np.cumsum(self.events[..., 1] - self.events[..., 2], axis=-1))
        n_max_np = dNt_np[:, 19:22] + self.initial_state[:, [2]]
        np.testing.assert_array_equal(n_max, n_max_np)

    def test_fwd_none(self):
        for t in range(self.events.shape[0]):
            n_max = _abscumdiff(events=self.events,
                                initial_state=self.initial_state, target_id=2,
                                bound_t=t, bound_id=-1).numpy()
            np.testing.assert_array_equal(n_max,
                                          np.full([self.events.shape[0], 1],
                                                  tf.int32.max))

    def test_bwd_state(self):
        t = 26
        n_max = _abscumdiff(events=self.events,
                            initial_state=self.initial_state, target_id=1,
                            bound_t=t - 1, bound_id=0).numpy()
        dNt_np = np.absolute(
            np.cumsum(self.events[..., 1] - self.events[..., 0], axis=-1))
        n_max_np = dNt_np[:, t - 1] + self.initial_state[:, 1]
        np.testing.assert_array_equal(n_max.flatten(), n_max_np)

    def test_bwd_none(self):
        for t in range(self.events.shape[0]):
            n_max = _abscumdiff(events=self.events,
                                initial_state=self.initial_state, target_id=0,
                                bound_t=t, bound_id=-1).numpy()
            np.testing.assert_array_equal(n_max,
                                          np.full([self.events.shape[0], 1],
                                                  tf.int32.max))


class TestEventTimeProposal(unittest.TestCase):
    def setUp(self):
        with open(_testsim, 'rb') as f:
            sim = pkl.load(f)
        self.events = np.stack([sim['events'][..., 0, 1],  # S->E
                                sim['events'][..., 1, 2],  # E->I
                                sim['events'][..., 2, 3]], axis=-1)  # I->R

        self.events = np.transpose(self.events,
                                   axes=(1, 0, 2))  # shape [M, T, K]
        hot_metapop = self.events[..., 1].sum(axis=-1) > 0
        self.events = self.events[hot_metapop]
        self.initial_state = sim['state_init'][hot_metapop]
        self.topology = TransitionTopology(prev=0, target=1, next=2)
        tf.random.set_seed(101020340)
        self.Q = EventTimeProposal(self.events, self.initial_state,
                                   self.topology,
                                   3, 10)

    def test_event_time_proposal_sample(self):
        q = self.Q.sample()
        np.testing.assert_array_equal(q['t'], [37, 36, 38, 22, 11, 33])
        self.assertEqual(2, q['delta_t'])
        np.testing.assert_array_equal(q['x_star'], [1, 0, 1, 0, 0, 1])
        log_prob = tf.reduce_sum(self.Q.log_prob(q))
        self.assertAlmostEqual(log_prob.numpy(), -31.732083, places=6)


class TestFilteredEventTimeProposal(unittest.TestCase):
    def setUp(self):
        with open(_testsim, 'rb') as f:
            sim = pkl.load(f)
        self.events = np.stack([sim['events'][..., 0, 1],  # S->E
                                sim['events'][..., 1, 2],  # E->I
                                sim['events'][..., 2, 3]], axis=-1)  # I->R

        self.events = np.transpose(self.events,
                                   axes=(1, 0, 2))  # shape [M, T, K]
        self.initial_state = sim['state_init']
        self.topology = TransitionTopology(prev=0, target=1, next=2)
        tf.random.set_seed(10202010)
        self.Q = FilteredEventTimeProposal(self.events, self.initial_state,
                                           self.topology,
                                           2, 10)

    def test_filtered_event_time_proposal_sample(self):
        q = self.Q.sample()
        move = q['move']
        np.testing.assert_array_equal(move['x_star'],
                                      [1],
                                      'mismatch in x_star')
        np.testing.assert_array_equal(move['t'], [33],
                                      'mismatch in t')
        self.assertEqual(1, move['delta_t'])
        log_prob = self.Q.log_prob(q)
        self.assertAlmostEqual(log_prob.numpy(), -4.56436819, delta=2.e-5)

    def test_zero_move(self):
        move = dict(m=tf.constant([6], dtype=tf.int32),
                    move=dict(t=tf.constant([33], dtype=tf.int32),
                              delta_t=tf.constant(1, dtype=tf.int32),
                              x_star=tf.constant([1], dtype=tf.int32)))
        lp = self.Q.log_prob(move)
        self.assertAlmostEqual(lp, -4.56436819, delta=2.e-5)

if __name__ == '__main__':
    unittest.main()
