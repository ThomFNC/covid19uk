"""Calculate Rt given a posterior"""
import argparse
import os
import yaml
import h5py
import numpy as np
import pandas as pd
import geopandas as gp
import pickle

import tensorflow as tf
from gemlib.util import compute_state

from covid.cli_arg_parse import cli_args
from covid.summary import (
    rayleigh_quotient,
    power_iteration,
)
from covid.summary import mean_and_ci
from FNC import GetData
import model_spec

DTYPE = model_spec.DTYPE


# FNC changes to enable pipelining:
# we load dates out of the config struct (formatted by the initialise node)
# covar data comes from either pipelineData or from GetData
# The file has been broken into runSummary and makeGeopackage methods
# the data required to create the geopackage is saved
def runSummary(pipelineData): 
    # Pipeline data should contain config at a minimium
    config = pipelineData['config']
    settings = config['SummaryData']
    
    if settings['input'] == 'processed':
        summaryData = GetData.SummaryData.process(config)
        pipelineData['summary'] = summaryData
        return pipelineData
    
    # as we're running in a function, we need to assign covar_data before defining
    # functions that call it in order for it to be in scope
    # previously, covar_dict was defined in the __name__ == 'main' portion of this script
    # moving to a pipeline necessitates this change.
    # grab all data from dicts
    # inference_period = config['dates']['inference_period']
    # date_low = config['dates']['low']
    # date_high = config['dates']['high']
    # weekday = config['dates']['weekday']

    if 'covar_data' in pipelineData:
        covar_data = pipelineData['covar_data']
    else:
        covar_data, tmp = GetData.CovarData(config)


    # Reproduction number calculation
    def calc_R_it(param, events, init_state, covar_data, priors):
        """Calculates effective reproduction number for batches of metapopulations
        :param theta: a tensor of batched theta parameters [B] + theta.shape
        :param xi: a tensor of batched xi parameters [B] + xi.shape
        :param events: a [B, M, T, X] batched events tensor
        :param init_state: the initial state of the epidemic at earliest inference date
        :param covar_data: the covariate data
        :return a batched vector of R_it estimates
        """

        def r_fn(args):
            beta1_, beta2_, beta_3, sigma_, xi_, gamma0_, events_ = args
            t = events_.shape[-2] - 1
            state = compute_state(init_state, events_, model_spec.STOICHIOMETRY)
            state = tf.gather(state, t, axis=-2)  # State on final inference day

            model = model_spec.CovidUK(
                covariates=covar_data,
                initial_state=init_state,
                initial_step=0,
                num_steps=events_.shape[-2],
                priors=priors,
            )

            xi_pred = model_spec.conditional_gp(
                model.model["xi"](beta1_, sigma_),
                xi_,
                tf.constant(
                    [events.shape[-2] + model_spec.XI_FREQ], dtype=model_spec.DTYPE
                )[:, tf.newaxis],
            )

            # FNC NOTE:
            # adding another 0.0 to beta3 as TF complains of dimension mismatch otherwise
            par = dict(
                beta1=beta1_,
                beta2=beta2_,
                beta3=tf.concat([beta_3, [0.0, 0.0]], axis=-1),
                sigma=sigma_,
                gamma0=gamma0_,
                xi=xi_,  # tf.reshape(xi_pred.sample(), [1]),
            )
            print("xi shape:", par["xi"].shape)
            ngm_fn = model_spec.next_generation_matrix_fn(covar_data, par)
            ngm = ngm_fn(t, state)
            return ngm

        return tf.vectorized_map(
            r_fn,
            elems=(
                param["beta1"],
                param["beta2"],
                param["beta3"],
                param["sigma"],
                param["xi"],
                param["gamma0"],
                events,
            ),
        )


    @tf.function
    def predicted_incidence(param, init_state, init_step, num_steps, priors):
        """Runs the simulation forward in time from `init_state` at time `init_time`
        for `num_steps`.
        :param theta: a tensor of batched theta parameters [B] + theta.shape
        :param xi: a tensor of batched xi parameters [B] + xi.shape
        :param events: a [B, M, S] batched state tensor
        :param init_step: the initial time step
        :param num_steps: the number of steps to simulate
        :param priors: the priors for gamma
        :returns: a tensor of srt_quhape [B, M, num_steps, X] where X is the number of state
                transitions
        """

        def sim_fn(args):
            beta1_, beta2_, beta3_, sigma_, xi_, gamma0_, gamma1_, init_ = args

            # FNC NOTE:
            # adding another 0.0 to beta3 as TF complains of dimension mismatch otherwise
            par = dict(
                beta1=beta1_,
                beta2=beta2_,
                beta3=tf.concat([beta3_, [0.0,0.0]], axis=-1),
                gamma0=gamma0_,
                gamma1=gamma1_,
                xi=xi_,
            )

            model = model_spec.CovidUK(
                covar_data,
                initial_state=init_,
                initial_step=init_step,
                num_steps=num_steps,
                priors=priors,
            )
            sim = model.sample(**par)
            return sim["seir"]

        events = tf.map_fn(
            sim_fn,
            elems=(
                param["beta1"],
                param["beta2"],
                param["beta3"],
                param["sigma"],
                param["xi"],
                param["gamma0"],
                param["gamma1"],
                init_state,
            ),
            fn_output_signature=(tf.float64),
        )
        return events


    # Today's prevalence
    def prevalence(predicted_state, population_size, name=None):
        """Computes prevalence of E and I individuals

        :param state: the state at a particular timepoint [batch, M, S]
        :param population_size: the size of the population
        :returns: a dict of mean and 95% credibility intervals for prevalence
                in units of infections per person
        """
        prev = tf.reduce_sum(predicted_state[:, :, 1:3], axis=-1) / tf.squeeze(
            population_size
        )
        return mean_and_ci(prev, name=name)


    def predicted_events(events, name=None):
        num_events = tf.reduce_sum(events, axis=-1)
        return mean_and_ci(num_events, name=name)




    # Load posterior file
    posterior_path = config['PosteriorData']['address']
    print("Using posterior:", posterior_path)
    posterior = h5py.File(
        os.path.expandvars(
            posterior_path,
        ),
        "r",
        rdcc_nbytes=1024 ** 3,
        rdcc_nslots=1e6,
    )

    # Pre-determined thinning of posterior (better done in MCMC?)
    if posterior["samples/beta1"].size >= 10000:
        idx = range(6000, 10000, 10)
    else:
        print('Using smaller MCMC sample range')
        print('Size of posterior["samples/beta1"] is',posterior["samples/beta1"].size)
        idx = range(600, 1000, 10)
    param = dict(
        beta1=posterior["samples/beta1"][idx],
        beta2=posterior["samples/beta2"][idx],
        beta3=posterior["samples/beta3"][
            idx,
        ],
        sigma=posterior["samples/sigma"][
            idx,
        ],
        xi=posterior["samples/xi"][idx],
        gamma0=posterior["samples/gamma0"][idx],
        gamma1=posterior["samples/gamma1"][idx],
    )
    events = posterior["samples/events"][idx]
    init_state = posterior["initial_state"][:]
    state_timeseries = compute_state(
        init_state, events, model_spec.STOICHIOMETRY
    )

    # Build model
    model = model_spec.CovidUK(
        covar_data,
        initial_state=init_state,
        initial_step=0,
        num_steps=events.shape[1],
        priors=config["mcmc"]["prior"],
    )

    ngms = calc_R_it(
        param, events, init_state, covar_data, config["mcmc"]["prior"]
    )
    b, _ = power_iteration(ngms)
    rt = rayleigh_quotient(ngms, b)
    q = np.arange(0.05, 1.0, 0.05)
    
    # FNC Note: removed dict from this and 
    # instead added Rt as a sheet name in the excel writer
    rt_quantiles = pd.DataFrame(
        np.quantile(rt, q, axis=-1), index=q
    ).T
    rt_quantiles.to_excel(
        config['RtQuantileData']['address'],sheet_name='Rt'
    )

    # Prediction requires simulation from the last available timepoint for 28 + 4 + 1 days
    # Note a 4 day recording lag in the case timeseries data requires that
    # now = state_timeseries.shape[-2] + 4
    prediction = predicted_incidence(
        param,
        init_state=state_timeseries[..., -1, :],
        init_step=state_timeseries.shape[-2] - 1,
        num_steps=70,
        priors=config["mcmc"]["prior"],
    )
    predicted_state = compute_state(
        state_timeseries[..., -1, :], prediction, model_spec.STOICHIOMETRY
    )

    # Prevalence now
    prev_now = prevalence(
        predicted_state[..., 4, :], covar_data["N"], name="prev"
    )

    # Incidence of detections now
    cases_now = predicted_events(prediction[..., 4:5, 2], name="cases")

    # Incidence from now to now+7
    cases_7 = predicted_events(prediction[..., 4:11, 2], name="cases7")
    cases_14 = predicted_events(prediction[..., 4:18, 2], name="cases14")
    cases_21 = predicted_events(prediction[..., 4:25, 2], name="cases21")
    cases_28 = predicted_events(prediction[..., 4:32, 2], name="cases28")
    cases_56 = predicted_events(prediction[..., 4:60, 2], name="cases56")

    # Prevalence at day 7
    prev_7 = prevalence(
        predicted_state[..., 11, :], covar_data["N"], name="prev7"
    )
    prev_14 = prevalence(
        predicted_state[..., 18, :], covar_data["N"], name="prev14"
    )
    prev_21 = prevalence(
        predicted_state[..., 25, :], covar_data["N"], name="prev21"
    )
    prev_28 = prevalence(
        predicted_state[..., 32, :], covar_data["N"], name="prev28"
    )
    prev_56 = prevalence(
        predicted_state[..., 60, :], covar_data["N"], name="prev56"
    )

    # Package up summary data
    # this will be saved into a pickle
    # Add LADs in for later reference
    summaryData = {'cases': {'now':cases_now,
                             '7':cases_7,
                             '14':cases_14,
                             '21':cases_21,
                             '28':cases_28,
                             '56':cases_56},
                   'prev':{'now':prev_now,
                           '7':prev_7,
                           '14':prev_14,
                           '21':prev_21,
                           '28':prev_28,
                           '56':prev_56},
                   'metrics':{'ngms':ngms,
                              'b':b,
                              'rt':rt,
                              'q':q},
                    'LADs': config['lad19cds']}
    
    # Save and pass on the output data
    if config['GenerateOutput']['summary']:
        settings = config['SummaryData']
        if settings['format'] == 'pickle':
            fn = settings['address']
            with open(fn,'wb') as file:
                pickle.dump(summaryData,file)
    pipelineData['summary'] = summaryData
    return pipelineData

# GeoPackage Functions

def geosummary(geodata, summaries):
    for summary in summaries:
        for k, v in summary.items():
            arr = v
            if isinstance(v, tf.Tensor):
                arr = v.numpy()
            geodata[k] = arr

def makeGeopackage(pipelineData):
    # This function makes the geopackage from the summary data
    config = pipelineData['config']
    settings = config['GeoSummary']
    
    GIS_TEMPLATE = settings['template']
    GIS_OUTPUT = settings['address']
    
    # use the pipelineData if available
    # otherwise try to load the processed output
    if 'summary' in pipelineData:
        summaryData = pipelineData['summary']
    else:
        print('SummaryData not present in pipelineData')
        print('Trying to load processed file instead')
        print('Looking for file at',config['SummaryData']['address'])
        config['SummaryData']['input'] = 'processed'
        summaryData = GetData.SummaryData.process(config)

    
    try:
        ltla = gp.read_file(GIS_TEMPLATE, layer="UK2019mod_pop_xgen")
    except:
        print("Layer UK2019mod_pop_xgen doesn't exist in",GIS_TEMPLATE)
        ltla = gp.read_file(GIS_TEMPLATE)
    ltla = ltla[ltla["lad19cd"].str.startswith("E")]  # England only, for now.
    ltla = ltla.sort_values("lad19cd")

    # FNC: These likely need to be downselected to just the LADs in the summary data
    # do some data checks and throw a warning for now if they're not the same
    # we didn't have an example file with more than 43 LADs
    geo_lads = ltla.lad19cd.values
    config_lads = np.array(summaryData['LADs'])
    if geo_lads.size != config_lads.size:
        print('GEOSUMMARY: Different number of LADs in GIS_TEMPLATE vs summaryData')
        print('GIS_Template:',geo_lads.size)
        print('summaryData:',config_lads.size)

    con_in_geo = np.array([x in geo_lads for x in config_lads]).sum()
    geo_in_con = np.array([x in config_lads for x in geo_lads]).sum()
    if con_in_geo != geo_lads.size:
        print('GEOSUMMARY:',con_in_geo,'/',geo_lads.size,'summary data LADs in GIS Template')
    if geo_in_con != config_lads.size:
        print('GEOSUMMARY:',geo_in_con,'/',config_lads.size,'GIS Template LADs in summary data')
    
    rti = tf.reduce_sum(summaryData['metrics']['ngms'], axis=-2)

    geosummary(
        ltla,
        (
            mean_and_ci(rti, name="Rt"),
            summaryData['prev']['now'],
            summaryData['cases']['now'],
            summaryData['prev']['7'],
            summaryData['prev']['14'],
            summaryData['prev']['21'],
            summaryData['prev']['28'],
            summaryData['prev']['56'],
            summaryData['cases']['7'],
            summaryData['cases']['14'],
            summaryData['cases']['21'],
            summaryData['cases']['28'],
            summaryData['cases']['56'],
        ),
    )

    ltla["Rt_exceed"] = np.mean(rti > 1.0, axis=0)
    ltla = ltla.loc[
        :,
        ltla.columns.str.contains(
            "(lad19cd|lad19nm$|prev|cases|Rt|popsize|geometry)", regex=True
        ),
    ]
    ltla.to_file(
        GIS_OUTPUT,
        driver="GPKG",
    )
