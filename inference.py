"""MCMC Test Rig for COVID-19 UK model"""
# pylint: disable=E402

import os
from time import perf_counter
import tqdm
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

from gemlib.util import compute_state
from gemlib.mcmc import UncalibratedEventTimesUpdate
from gemlib.mcmc import UncalibratedOccultUpdate, TransitionTopology
from gemlib.mcmc import GibbsKernel
from gemlib.mcmc import MultiScanKernel
from gemlib.mcmc import AdaptiveRandomWalkMetropolis
from gemlib.mcmc import Posterior

from covid.data import read_phe_cases
from covid.cli_arg_parse import cli_args

import model_spec

if tf.test.gpu_device_name():
    print("Using GPU")
else:
    print("Using CPU")


tfd = tfp.distributions
tfb = tfp.bijectors
DTYPE = model_spec.DTYPE

# FNC changes to enable pipelining:
# we load dates out of the config struct (formatted by the initialise node)
# covar data comes from either pipelineData or from GetData

def runInference(pipelineData):

    # Read in settings
    config = pipelineData['config']

    # Extract data
    if 'covar_data' in pipelineData:
        covar_data = pipelineData['covar_data']
    else:
        covar_data, data = GetData.CovarData(config)
        pipelineData['covar_data'] = covar_data
        pipelineData['data'] = data
    # inference_period = config['dates']['inference_period']
    # date_low = config['dates']['low']
    # date_high = config['dates']['high']
    # weekday = config['dates']['weekday']
    
    # We load in cases and impute missing infections first, since this sets the
    # time epoch which we are analysing.
    cases = pipelineData['data']['cases_wide']

    # Impute censored events, return cases
    events = model_spec.impute_censored_events(cases)

    # Initial conditions are calculated by calculating the state
    # at the beginning of the inference period
    #
    # Imputed censored events that pre-date the first I-R events
    # in the cases dataset are discarded.  They are only used to
    # to set up a sensible initial state.
    _initial_state = tf.concat([covar_data["N"], tf.zeros_like(events[:, 0, :])], axis=-1)
    state = compute_state(
            initial_state=_initial_state,
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
    def convert_priors(node):
        if isinstance(node, dict):
            for k, v in node.items():
                node[k] = convert_priors(v)
            return node
        return float(node)

    
    model = model_spec.CovidUK(
        covariates=covar_data,
        initial_state=initial_state,
        initial_step=0,
        num_steps=events.shape[1],
        priors=convert_priors(config["mcmc"]["prior"]),
    )

    # Full joint log posterior distribution
    # $\pi(\theta, \xi, y^{se}, y^{ei} | y^{ir})$
    # FNC NOTE:
    # adding another 0.0 to beta3 as TF complains of dimension mismatch otherwise
    def logp(block0, block1, events):
        return model.log_prob(
            dict(
                beta2=block0[0],
                gamma0=block0[1],
                gamma1=block0[2],
                sigma=block0[3],
                beta3=tf.concat([block0[4:6], [0.0, 0.0]], axis=-1),
                beta1=block1[0],
                xi=block1[1:],
                seir=events,
            )
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
    def make_blk0_kernel(shape, name):
        def fn(target_log_prob_fn, _):
            return tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=AdaptiveRandomWalkMetropolis(
                    target_log_prob_fn=target_log_prob_fn,
                    initial_covariance=np.eye(shape[0], dtype=model_spec.DTYPE)
                    * 1e-1,
                    covariance_burnin=200,
                ),
                bijector=tfp.bijectors.Blockwise(
                    bijectors=[
                        tfp.bijectors.Exp(),
                        tfp.bijectors.Identity(),
                        tfp.bijectors.Exp(),
                        tfp.bijectors.Identity(),
                    ],
                    block_sizes=[1, 2, 1, 2],
                ),
                name=name,
            )

        return fn

    def make_blk1_kernel(shape, name):
        def fn(target_log_prob_fn, _):
            return AdaptiveRandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn,
                initial_covariance=np.eye(shape[0], dtype=model_spec.DTYPE)
                * 1e-1,
                covariance_burnin=200,
                name=name,
            )

        return fn

    def make_partially_observed_step(
        target_event_id, prev_event_id=None, next_event_id=None, name=None
    ):
        def fn(target_log_prob_fn, _):
            return tfp.mcmc.MetropolisHastings(
                inner_kernel=UncalibratedEventTimesUpdate(
                    target_log_prob_fn=target_log_prob_fn,
                    target_event_id=target_event_id,
                    prev_event_id=prev_event_id,
                    next_event_id=next_event_id,
                    initial_state=initial_state,
                    dmax=config["mcmc"]["dmax"],
                    mmax=config["mcmc"]["m"],
                    nmax=config["mcmc"]["nmax"],
                ),
                name=name,
            )

        return fn

    def make_occults_step(prev_event_id, target_event_id, next_event_id, name):
        def fn(target_log_prob_fn, _):
            return tfp.mcmc.MetropolisHastings(
                inner_kernel=UncalibratedOccultUpdate(
                    target_log_prob_fn=target_log_prob_fn,
                    topology=TransitionTopology(
                        prev_event_id, target_event_id, next_event_id
                    ),
                    cumulative_event_offset=initial_state,
                    nmax=config["mcmc"]["occult_nmax"],
                    t_range=(events.shape[1] - 21, events.shape[1]),
                    name=name,
                ),
                name=name,
            )

        return fn

    def make_event_multiscan_kernel(target_log_prob_fn, _):
        return MultiScanKernel(
            config["mcmc"]["num_event_time_updates"],
            GibbsKernel(
                target_log_prob_fn=target_log_prob_fn,
                kernel_list=[
                    (0, make_partially_observed_step(0, None, 1, "se_events")),
                    (0, make_partially_observed_step(1, 0, 2, "ei_events")),
                    (0, make_occults_step(None, 0, 1, "se_occults")),
                    (0, make_occults_step(0, 1, 2, "ei_occults")),
                ],
                name="gibbs1",
            ),
        )

    # MCMC tracing functions
    def trace_results_fn(_, results):
        """Packs results into a dictionary"""
        results_dict = {}
        res0 = results.inner_results

        results_dict["block0"] = {
            "is_accepted": res0[0].inner_results.is_accepted,
            "target_log_prob": res0[
                0
            ].inner_results.accepted_results.target_log_prob,
        }
        results_dict["block1"] = {
            "is_accepted": res0[1].is_accepted,
            "target_log_prob": res0[1].accepted_results.target_log_prob,
        }

        def get_move_results(results):
            return {
                "is_accepted": results.is_accepted,
                "target_log_prob": results.accepted_results.target_log_prob,
                "proposed_delta": tf.stack(
                    [
                        results.accepted_results.m,
                        results.accepted_results.t,
                        results.accepted_results.delta_t,
                        results.accepted_results.x_star,
                    ]
                ),
            }

        res1 = res0[2].inner_results
        results_dict["move/S->E"] = get_move_results(res1[0])
        results_dict["move/E->I"] = get_move_results(res1[1])
        results_dict["occult/S->E"] = get_move_results(res1[2])
        results_dict["occult/E->I"] = get_move_results(res1[3])

        return results_dict

    # Build MCMC algorithm here.  This will be run in bursts for memory economy
    @tf.function(autograph=False, experimental_compile=True)
    def sample(n_samples, init_state, thin=0, previous_results=None):
        with tf.name_scope("main_mcmc_sample_loop"):

            init_state = init_state.copy()

            gibbs_schema = GibbsKernel(
                target_log_prob_fn=logp,
                kernel_list=[
                    (0, make_blk0_kernel(init_state[0].shape, "block0")),
                    (1, make_blk1_kernel(init_state[1].shape, "block1")),
                    (2, make_event_multiscan_kernel),
                ],
                name="gibbs0",
            )

            samples, results, final_results = tfp.mcmc.sample_chain(
                n_samples,
                init_state,
                kernel=gibbs_schema,
                num_steps_between_results=thin,
                previous_kernel_results=previous_results,
                return_final_kernel_results=True,
                trace_fn=trace_results_fn,
            )

            return samples, results, final_results

    ####################################
    # Construct bursted MCMC loop here #
    ####################################

    # MCMC Control
    NUM_BURSTS = int(config["mcmc"]["num_bursts"])
    NUM_BURST_SAMPLES = int(config["mcmc"]["num_burst_samples"])
    NUM_EVENT_TIME_UPDATES = int(config["mcmc"]["num_event_time_updates"])
    NUM_SAVED_SAMPLES = NUM_BURST_SAMPLES * NUM_BURSTS

    # RNG stuff
    tf.random.set_seed(2)

    current_state = [
        np.array([0.2, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=DTYPE),
        np.zeros(
            model.model["xi"](0.0, 0.1).event_shape[-1]
            # + model.model["beta3"]().event_shape[-1]
            + 1,
            dtype=DTYPE,
        ),
        events,
    ]
    print("Initial logpi:", logp(*current_state))

    # Output file
    samples, results, _ = sample(1, current_state)
    print('Storing posterior data at',config["PosteriorData"]["address"])
    posterior = Posterior(config["PosteriorData"]["address"],
        sample_dict={
            "beta2": (samples[0][:, 0], (NUM_BURST_SAMPLES,)),
            "gamma0": (samples[0][:, 1], (NUM_BURST_SAMPLES,)),
            "gamma1": (samples[0][:, 2], (NUM_BURST_SAMPLES,)),
            "sigma": (samples[0][:, 3], (NUM_BURST_SAMPLES,)),
            "beta3": (samples[0][:, 4:], (NUM_BURST_SAMPLES, 2)),
            "beta1": (samples[1][:, 0], (NUM_BURST_SAMPLES,)),
            "xi": (
                samples[1][:, 1:],
                (NUM_BURST_SAMPLES, samples[1].shape[1] - 1),
            ),
            "events": (samples[2], (NUM_BURST_SAMPLES, 43, 43, 1)), # Change so it adapts size correctly
        },
        results_dict=results,
        num_samples=NUM_SAVED_SAMPLES,
    )
    posterior._file.create_dataset("initial_state", data=initial_state)
    posterior._file.create_dataset("config", data=yaml.dump(config))

    # We loop over successive calls to sample because we have to dump results
    #   to disc, or else end OOM (even on a 32GB system).
    # with tf.profiler.experimental.Profile("/tmp/tf_logdir"):
    final_results = None
    for i in tqdm.tqdm(
        range(NUM_BURSTS), unit_scale=NUM_BURST_SAMPLES * config["mcmc"]["thin"]
    ):
        samples, results, final_results = sample(
            NUM_BURST_SAMPLES,
            init_state=current_state,
            thin=config["mcmc"]["thin"] - 1,
            previous_results=final_results,
        )
        current_state = [s[-1] for s in samples]
        print(current_state[0].numpy(), flush=True)

        start = perf_counter()
        posterior.write_samples(
            {
                "beta2": samples[0][:, 0],
                "gamma0": samples[0][:, 1],
                "gamma1": samples[0][:, 2],
                "sigma": samples[0][:, 3],
                "beta3": samples[0][:, 4:],
                "beta1": samples[1][:, 0],
                "xi": samples[1][:, 1:],
                "events": samples[2],
            },
            first_dim_offset=i * NUM_BURST_SAMPLES,
        )
        posterior.write_results(results, first_dim_offset=i * NUM_BURST_SAMPLES)
        end = perf_counter()

        print("Storage time:", end - start, "seconds")
        print(
            "Acceptance theta:",
            tf.reduce_mean(
                tf.cast(results["block0"]["is_accepted"], tf.float32)
            ),
        )
        print(
            "Acceptance xi:",
            tf.reduce_mean(
                tf.cast(results["block1"]["is_accepted"], tf.float32),
            ),
        )
        print(
            "Acceptance move S->E:",
            tf.reduce_mean(
                tf.cast(results["move/S->E"]["is_accepted"], tf.float32)
            ),
        )
        print(
            "Acceptance move E->I:",
            tf.reduce_mean(
                tf.cast(results["move/E->I"]["is_accepted"], tf.float32)
            ),
        )
        print(
            "Acceptance occult S->E:",
            tf.reduce_mean(
                tf.cast(results["occult/S->E"]["is_accepted"], tf.float32)
            ),
        )
        print(
            "Acceptance occult E->I:",
            tf.reduce_mean(
                tf.cast(results["occult/E->I"]["is_accepted"], tf.float32)
            ),
        )

    print(
        f"Acceptance theta: {posterior['results/block0/is_accepted'][:].mean()}"
    )
    print(f"Acceptance xi: {posterior['results/block1/is_accepted'][:].mean()}")
    print(
        f"Acceptance move S->E: {posterior['results/move/S->E/is_accepted'][:].mean()}"
    )
    print(
        f"Acceptance move E->I: {posterior['results/move/E->I/is_accepted'][:].mean()}"
    )
    print(
        f"Acceptance occult S->E: {posterior['results/occult/S->E/is_accepted'][:].mean()}"
    )
    print(
        f"Acceptance occult E->I: {posterior['results/occult/E->I/is_accepted'][:].mean()}"
    )

    del posterior
