"""MCMC Test Rig for COVID-19 UK model"""
# pylint: disable=E402

import argparse
import os

# Uncomment to block GPU use

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from time import perf_counter

import tqdm
import yaml
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from covid.impl.util import compute_state
from covid.impl.mcmc import UncalibratedLogRandomWalk, random_walk_mvnorm_fn
from covid.impl.event_time_mh import UncalibratedEventTimesUpdate
from covid.impl.occult_events_mh import UncalibratedOccultUpdate, TransitionTopology
from covid.impl.gibbs import DeterministicScanKernel, GibbsStep, flatten_results
from covid.impl.multi_scan_kernel import MultiScanKernel
from covid.data import read_phe_cases

import model_spec

if tf.test.gpu_device_name():
    print("Using GPU")
else:
    print("Using CPU")


tfd = tfp.distributions
tfb = tfp.bijectors
DTYPE = model_spec.DTYPE


if __name__ == "__main__":

    # Read in settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="example_config.yaml",
        help="configuration file",
    )
    args = parser.parse_args()
    print("Loading config file:", args.config)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    inference_period = [
        np.datetime64(x) for x in config["settings"]["inference_period"]
    ]

    covar_data = model_spec.read_covariates(
        config["data"], date_low=inference_period[0], date_high=inference_period[1],
    )

    # We load in cases and impute missing infections first, since this sets the
    # time epoch which we are analysing.
    cases = read_phe_cases(
        config["data"]["reported_cases"],
        date_low=inference_period[0],
        date_high=inference_period[1],
        date_type="report",
    ).astype(DTYPE)

    # Impute censored events, return cases
    events = model_spec.impute_censored_events(cases)

    # Initial conditions are calculated by calculating the state
    # at the beginning of the inference period
    #
    # Imputed censored events that pre-date the first I-R events
    # in the cases dataset are discarded.  They are only used to
    # to set up a sensible initial state.
    state = compute_state(
        initial_state=tf.concat(
            [covar_data["N"][:, tf.newaxis], tf.zeros_like(events[:, 0, :])], axis=-1
        ),
        events=events,
        stoichiometry=model_spec.STOICHIOMETRY,
    )
    start_time = state.shape[1] - cases.shape[1]
    initial_state = state[:, start_time, :]
    events = events[:, start_time:, :]
    num_metapop = covar_data["N"].shape[0]

    ########################################################
    # Build the model, and then construct the MCMC kernels #
    ########################################################
    model = model_spec.CovidUK(
        covariates=covar_data,
        initial_state=initial_state,
        initial_step=0,
        num_steps=events.shape[1],
    )

    # Full joint log posterior distribution
    # $\pi(\theta, \xi, y^{se}, y^{ei} | y^{ir})$
    def logp(theta, xi, events):
        return model.log_prob(
            dict(beta1=theta[0], beta2=theta[1], gamma=theta[2], xi=xi, seir=events,)
        )

    # Build Metropolis within Gibbs sampler
    #
    # Kernels are:
    #     Q(\theta, \theta^\prime)
    #     Q(\xi, \xi^\prime)
    #     Q(Z^{se}, Z^{se\prime}) (partially-censored)
    #     Q(Z^{ei}, Z^{ei\prime}) (partially-censored)
    #     Q(Z^{se}, Z^{se\prime}) (occult)
    #     Q(Z^{ei}, Z^{ei\prime}) (occult)
    def make_theta_kernel(scale, bounded_convergence, name):
        return GibbsStep(
            0,
            tfp.mcmc.MetropolisHastings(
                inner_kernel=UncalibratedLogRandomWalk(
                    target_log_prob_fn=logp,
                    new_state_fn=random_walk_mvnorm_fn(scale, p_u=bounded_convergence),
                )
            ),
            name=name,
        )

    def make_xi_kernel(scale, bounded_convergence, name):
        return GibbsStep(
            1,
            tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=logp,
                new_state_fn=random_walk_mvnorm_fn(scale, p_u=bounded_convergence),
            ),
            name=name,
        )

    def make_partially_observed_step(
        target_event_id, prev_event_id=None, next_event_id=None, name=None
    ):
        return GibbsStep(
            2,
            tfp.mcmc.MetropolisHastings(
                inner_kernel=UncalibratedEventTimesUpdate(
                    target_log_prob_fn=logp,
                    target_event_id=target_event_id,
                    prev_event_id=prev_event_id,
                    next_event_id=next_event_id,
                    initial_state=initial_state,
                    dmax=config["mcmc"]["dmax"],
                    mmax=config["mcmc"]["m"],
                    nmax=config["mcmc"]["nmax"],
                )
            ),
            name=name,
        )

    def make_occults_step(prev_event_id, target_event_id, next_event_id, name):
        return GibbsStep(
            2,
            tfp.mcmc.MetropolisHastings(
                inner_kernel=UncalibratedOccultUpdate(
                    target_log_prob_fn=logp,
                    topology=TransitionTopology(
                        prev_event_id, target_event_id, next_event_id
                    ),
                    cumulative_event_offset=initial_state,
                    nmax=config["mcmc"]["occult_nmax"],
                    t_range=(events.shape[1] - 21, events.shape[1]),
                    name=name,
                )
            ),
            name=name,
        )

    # MCMC tracing functions
    def trace_results_fn(_, results):
        """Returns log_prob, accepted, q_ratio"""

        def is_accepted(result):
            if hasattr(result, "is_accepted"):
                return tf.cast(result.is_accepted, DTYPE)
            return is_accepted(result.inner_results)

        def f(result):
            log_prob = result.proposed_results.target_log_prob
            accepted = is_accepted(result)
            q_ratio = result.proposed_results.log_acceptance_correction
            if hasattr(result.proposed_results, "extra"):
                proposed = tf.cast(result.proposed_results.extra, log_prob.dtype)
                return tf.concat([[log_prob], [accepted], [q_ratio], proposed], axis=0)
            return tf.concat([[log_prob], [accepted], [q_ratio]], axis=0)

        def recurse(f, list_or_atom):
            if isinstance(list_or_atom, list):
                return [recurse(f, x) for x in list_or_atom]
            return f(list_or_atom)

        return recurse(f, results)

    # Build MCMC algorithm here.  This will be run in bursts for memory economy
    @tf.function(autograph=False, experimental_compile=True)
    def sample(n_samples, init_state, sigma_theta, sigma_xi):
        with tf.name_scope("main_mcmc_sample_loop"):

            init_state = init_state.copy()

            kernel = DeterministicScanKernel(
                [
                    make_theta_kernel(sigma_theta, 1.0, "theta_kernel"),
                    make_xi_kernel(sigma_xi, 1.0, "xi_kernel"),
                    MultiScanKernel(
                        config["mcmc"]["num_event_time_updates"],
                        DeterministicScanKernel(
                            [
                                make_partially_observed_step(0, None, 1, "se_events"),
                                make_partially_observed_step(1, 0, 2, "ei_events"),
                                make_occults_step(None, 0, 1, "se_occults"),
                                make_occults_step(0, 1, 2, "ei_occults"),
                            ]
                        ),
                    ),
                ],
                name="gibbs_kernel",
            )

            samples, results = tfp.mcmc.sample_chain(
                n_samples, init_state, kernel=kernel, trace_fn=trace_results_fn
            )

            return samples, results

    ####################################
    # Construct bursted MCMC loop here #
    ####################################

    # MCMC Control
    NUM_BURSTS = config["mcmc"]["num_bursts"]
    NUM_BURST_SAMPLES = config["mcmc"]["num_burst_samples"]
    NUM_EVENT_TIME_UPDATES = config["mcmc"]["num_event_time_updates"]
    THIN_BURST_SAMPLES = NUM_BURST_SAMPLES // config["mcmc"]["thin"]
    NUM_SAVED_SAMPLES = THIN_BURST_SAMPLES * NUM_BURSTS

    # RNG stuff
    tf.random.set_seed(2)

    current_state = [
        np.array([0.85, 0.3, 0.25], dtype=DTYPE),
        np.zeros(model.model["xi"]().event_shape[-1], dtype=DTYPE),
        events,
    ]

    # Output Files
    posterior = h5py.File(
        os.path.expandvars(config["output"]["posterior"]),
        "w",
        rdcc_nbytes=1024 ** 2 * 400,
        rdcc_nslots=100000,
        libver="latest",
    )
    event_size = [NUM_SAVED_SAMPLES] + list(current_state[2].shape)

    posterior.create_dataset("initial_state", data=initial_state)

    # Ideally we insert the inference period into the posterior file
    # as this allows us to post-attribute it to the data.  Maybe better
    # to simply save the data into it as well.
    posterior.create_dataset("config", data=yaml.dump(config))
    theta_samples = posterior.create_dataset(
        "samples/theta",
        [NUM_SAVED_SAMPLES, current_state[0].shape[0]],
        dtype=np.float64,
    )
    xi_samples = posterior.create_dataset(
        "samples/xi", [NUM_SAVED_SAMPLES, current_state[1].shape[0]], dtype=np.float64,
    )
    event_samples = posterior.create_dataset(
        "samples/events",
        event_size,
        dtype=DTYPE,
        chunks=(32, 32, 32, 1),
        compression="szip",
        compression_opts=("nn", 16),
    )

    output_results = [
        posterior.create_dataset("results/theta", (NUM_SAVED_SAMPLES, 3), dtype=DTYPE,),
        posterior.create_dataset("results/xi", (NUM_SAVED_SAMPLES, 3), dtype=DTYPE,),
        posterior.create_dataset(
            "results/move/S->E", (NUM_SAVED_SAMPLES, 3 + num_metapop), dtype=DTYPE,
        ),
        posterior.create_dataset(
            "results/move/E->I", (NUM_SAVED_SAMPLES, 3 + num_metapop), dtype=DTYPE,
        ),
        posterior.create_dataset(
            "results/occult/S->E", (NUM_SAVED_SAMPLES, 6), dtype=DTYPE
        ),
        posterior.create_dataset(
            "results/occult/E->I", (NUM_SAVED_SAMPLES, 6), dtype=DTYPE
        ),
    ]
    posterior.swmr_mode = True

    print("Initial logpi:", logp(*current_state))

    theta_scale = tf.constant(
        [
            [1.12e-3, 1.67e-4, 1.61e-4],
            [1.67e-4, 7.41e-4, 4.68e-5],
            [1.61e-4, 4.68e-5, 1.28e-4],
        ],
        dtype=DTYPE,
    )
    theta_scale = theta_scale * 0.2 / theta_scale.shape[0]

    xi_scale = tf.eye(current_state[1].shape[0], dtype=DTYPE)
    xi_scale = xi_scale * 0.0001 / xi_scale.shape[0]

    # We loop over successive calls to sample because we have to dump results
    #   to disc, or else end OOM (even on a 32GB system).
    # with tf.profiler.experimental.Profile("/tmp/tf_logdir"):
    for i in tqdm.tqdm(range(NUM_BURSTS), unit_scale=NUM_BURST_SAMPLES):
        samples, results = sample(
            NUM_BURST_SAMPLES,
            init_state=current_state,
            sigma_theta=theta_scale,
            sigma_xi=xi_scale,
        )
        current_state = [s[-1] for s in samples]
        s = slice(i * THIN_BURST_SAMPLES, i * THIN_BURST_SAMPLES + THIN_BURST_SAMPLES)
        idx = tf.constant(range(0, NUM_BURST_SAMPLES, config["mcmc"]["thin"]))
        theta_samples[s, ...] = tf.gather(samples[0], idx)
        xi_samples[s, ...] = tf.gather(samples[1], idx)
        # cov = np.cov(
        #     np.log(theta_samples[: (i * NUM_BURST_SAMPLES + NUM_BURST_SAMPLES), ...]),
        #     rowvar=False,
        # )
        print(current_state[0].numpy(), flush=True)
        # print(cov, flush=True)
        # if (i * NUM_BURST_SAMPLES) > 1000 and np.all(np.isfinite(cov)):
        #     theta_scale = 2.38 ** 2 * cov / 2.0

        start = perf_counter()
        event_samples[s, ...] = tf.gather(samples[2], idx)
        end = perf_counter()

        flat_results = flatten_results(results)
        for i, ro in enumerate(output_results):
            ro[s, ...] = tf.gather(flat_results[i], idx)

        posterior.flush()
        print("Storage time:", end - start, "seconds")
        print(
            "Acceptance theta:",
            tf.reduce_mean(tf.cast(flat_results[0][:, 1], tf.float32)),
        )
        print(
            "Acceptance xi:", tf.reduce_mean(tf.cast(flat_results[1][:, 1], tf.float32))
        )
        print(
            "Acceptance move S->E:",
            tf.reduce_mean(tf.cast(flat_results[2][:, 1], tf.float32)),
        )
        print(
            "Acceptance move E->I:",
            tf.reduce_mean(tf.cast(flat_results[3][:, 1], tf.float32)),
        )
        print(
            "Acceptance occult S->E:",
            tf.reduce_mean(tf.cast(flat_results[4][:, 1], tf.float32)),
        )
        print(
            "Acceptance occult E->I:",
            tf.reduce_mean(tf.cast(flat_results[5][:, 1], tf.float32)),
        )

    print(f"Acceptance theta: {output_results[0][:, 1].mean()}")
    print(f"Acceptance xi: {output_results[1][:, 1].mean()}")
    print(f"Acceptance move S->E: {output_results[2][:, 1].mean()}")
    print(f"Acceptance move E->I: {output_results[3][:, 1].mean()}")
    print(f"Acceptance occult S->E: {output_results[4][:, 1].mean()}")
    print(f"Acceptance occult E->I: {output_results[5][:, 1].mean()}")

    posterior.close()
