# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:13:04 2020

This file details the various different pipeline nodes.
Nodes can be added pretty simply - at the very least, they need a run node which takes 
self and another argument - we pass pipelineData from node to node.
They then emit pipelineData to the next node.

The first node requires a setup method as well as a run method.

Here, we use an Initialise node to simply setup the pipeline.
That way nodes can be called in any order (although at present it only makes sense to start 
with GatherData)

PyPiper seems to loop back round to the first node's run method upon completion.
This happens even when gthe final node emits nothing and calls close().
To finalise the pipeline correctly, we call an End node which emits nothing.
This triggers the Initialise node again, which calls close() and finishes the pipeline.


@authors: Thom Kirwan-Evans, Jeremy Revell
"""

from pyPiper import Node 
from FNC import GetData
from FNC.GetData import format_output_filename
from FNC import COVIDInspect
from inference import runInference
from summary import runSummary, makeGeopackage
import pandas as pd
from covid import data as COVIDData
import numpy as np
import model_spec

DTYPE = model_spec.DTYPE

class Initialise(Node):
    # This node sets up all common calculations
    # Then passes pipeline Data to the next node
    def setup(self,pipelineData):
        print('PIPELINE: Initialising')
        config = pipelineData['config']

        # Configure the pipeline to run
        self.RunCOVID = True

        # Update output addresses to include date and strings
        outputs = ['SummaryData','PosteriorData','RtQuantileData','GeoSummary']
        for O in outputs:
            config[O]['address'] = format_output_filename(config[O]['address'],config)

        # calculate date periods
        # do this once and pass around
        inference_period = [np.datetime64(x) for x in config["Global"]["inference_period"]]
        date_low = inference_period[0]
        date_high = inference_period[1]
        weekday = pd.date_range(date_low, date_high).weekday < 5
        prediction_period = [np.datetime64(x) for x in config["Global"]["prediction_period"]]
        date_low_pred = prediction_period[0]
        date_high_pred = prediction_period[1]
        pipelineData['config']['dates'] = {
            'inference_period':inference_period,
            'prediction_period':prediction_period,
            'low':date_low,
            'high':date_high,
            'low_pred':date_low_pred,
            'high_pred':date_high_pred,
            'weekday':weekday
        }
        self.pipelineData = pipelineData

    def run(self,tmp):
        # tmp is needed, as one argument is always passed in

        if self.RunCOVID:
            print('PIPELINE: Running')
            # configure the pipeline to stop running next time we get here
            self.RunCOVID = False
            self.emit(self.pipelineData)
        else:
            print('PIPELINE: Stopping')
            self.close()


class GatherData(Node):
    # Node to load initial data
    def run(self,pipelineData):
        print('PIPELINE: Gathering data')
        # tmp is None unless emited from something else, but needs to be there
        
        # Use example data - for testing only
        config = pipelineData['config']
        if config['Global']['useExampleData']:
            print("---------------------")
            print('USING EXAMPLE DATA FILES')
            print('OUTPUTS SHOULD NOT BE USED')
            print("---------------------")
            mobility = pd.read_csv(config['ExampleFiles']['MobilityTrendData'], index_col=0)
            popsize = pd.read_csv(config['ExampleFiles']['PopulationData'], index_col=0)
            commute_volume = pd.read_csv(config['ExampleFiles']['InterLadCommuteData'], index_col=0)
            cases_tidy = pd.read_csv(config['ExampleFiles']['CasesData'])
            cases_wide = cases_tidy.reset_index().pivot(index="lad19cd", columns="date", values="cases")
            
            date_low = config['dates']['low']
            date_high = config['dates']['high']
            allAreaCodes = GetData.AreaCodeData.process(config) 
            ix = np.unique(cases_tidy.lad19cd.values)
            smallAreaCodes = allAreaCodes[allAreaCodes.lad19cd.isin(ix)]
            config['lad19cds'] = list(smallAreaCodes.lad19cd)
            tiers = COVIDData.read_challen_tier_restriction(config['TierData']['address'],date_low,date_high,config['lad19cds'])
            
            pipelineData['data']['commute'] = commute_volume
            pipelineData['data']['cases_tidy'] = cases_tidy
            pipelineData['data']['cases_wide'] = cases_wide
            pipelineData['data']['mobility'] = mobility
            pipelineData['data']['population'] = popsize
            pipelineData['data']['tier'] = tiers
            pipelineData['data']['areas'] = smallAreaCodes.values
            pipelineData['covar_data'] = dict(
                      C=pipelineData['data']['commute'].to_numpy().astype(DTYPE),
                      W=pipelineData['data']['mobility'].to_numpy().astype(DTYPE),
                      N=pipelineData['data']['population'].to_numpy().astype(DTYPE),
                      L=pipelineData['data']['tier'].astype(DTYPE),
                      weekday=config['dates']['weekday'].astype(DTYPE),
                    )
        
        
        # use GetData functions to load the specified data
        else:
            pipelineData['data'], pipelineData['covar_data'] = GetData.CovarData(config)

        # Print some statistics
        print('--------------')
        print('GatherData stats:')
        print('Inter-LAD commute')
        print('Shape:',pipelineData['data']['commute'].shape)
        print('-')
        print('Cases:')
        print('LADs:',np.unique(pipelineData['data']['cases_tidy'].lad19cd.values).size)
        print('Dates:',np.unique(pipelineData['data']['cases_tidy'].date.values).size)
        print('-')
        print('Mobility multiplier:')
        print('Dates:',pipelineData['data']['mobility'].shape[0])
        print('-')
        print('Population:')
        print('LADs:',np.unique(pipelineData['data']['cases_tidy'].lad19cd.values).size)
        print('-')
        print('Restrictions:')
        print('LADs:',np.unique(pipelineData['data']['tier'].code).size)
        print('Dates:',np.unique(pipelineData['data']['tier'].date).size)
        print('Tiers:',np.unique(pipelineData['data']['tier'].tier))
        print('--------------')




        self.emit(pipelineData)
        
        
class Infer(Node):
    # Run the inference model
    # Generates the posterior data
    def run(self,pipelineData):
        print('PIPELINE: Running inference')
        runInference(pipelineData)
        self.emit(pipelineData)
        
    
class Summarise(Node):
    # Run the Summary module
    # Samples the posterior data  to forecast cases, prevalence, and metrics
    def run(self,pipelineData):
        print('PIPELINE: Running summary')
        runSummary(pipelineData)
        self.emit(pipelineData)
        
class MakeGPKG(Node):
    # Generates the geopackage
    def run(self,pipelineData):
        print('PIPELINE: Creating geopackage')
        makeGeopackage(pipelineData)
        self.emit(pipelineData)
        print('Done')
    
class InspectData(Node):
    # Class to plot performance of a model
    def run(self,pipelineData):
        print('PIPELINE: Inspecting data')
        config = pipelineData['config']
        COVIDInspect.setBackend(config)

        if 'exampleLAD' in config['Inspect'] and 'data' in pipelineData:
            try:
                LAD = (pipelineData['data']['areas'].lad19cd.values==config['Inspect']['exampleLAD']).argmax()
            except:
                config['Inspect']['exampleLAD'] = 'Unknown'
                LAD = 0 
        else: 
            config['Inspect']['exampleLAD'] = 'Unknown'
            LAD = 0
        config['Inspect']['exampleLADix'] = LAD

        if 'data' in pipelineData:
            COVIDInspect.InspectInputData(pipelineData['data'],config)
        if 'summary' in pipelineData:
            COVIDInspect.InspectSummaryData(pipelineData['summary'],config)
        self.emit(pipelineData)

class PlotResults(Node):
    # Class to create result plots
    # Node is currently empty to allow for future results to be added
    def run(self,pipelineData):
        pass
            
class PipelineEnd(Node):
    # Class that finishes the pipeline
    def run(self,tmp):
        print('PIPELINE: Finished')
        # self.close()
        
        