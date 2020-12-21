# -*- coding: utf-8 -*-
"""
This file runs the COVID pipeline using pyPiper.
pipelineData is pass to each of the nodes in turn.
It starts with config, and is populated as it passes through each node.
If necessary data isn't in the structure (i.e. a prior node hasn't been run),
    the pipeline will attempt to load it. See makeGeopackage() in summary.py 
    for an example
Each stage should be able to store the data necessary for the next stage,
    as well as pass it on directly using pipelineData

@authors: Thom Kirwan-Evans, Jeremy Revell
"""
from pyPiper import Pipeline
from FNC.COVIDPipeline import *
import yaml

# specify the config file to load
with open("./configs/config_pipeline.yml", 'r') as f:
    config = yaml.safe_load(f)

# Create the pipelineData object
# This will be passed from node to node
pipelineData = {'config':config,'data':{}}

# Run the pipeline
pipeline = Pipeline(
        Initialise("Initialise", pipelineData=pipelineData) # adds dates to config
        | GatherData("Gather") # adds 'data' and 'covar_data'
        | Infer("Infer") # generates posterior and saves it
        | Summarise("Summary") # simulates and adds 'summaryData'
        | MakeGPKG("GPKG") # creates a geopackage output
        | InspectData("Inspector") # creates plots for data checks
        # | PlotResults("Plotter") # creates plots of results
        | PipelineEnd('End') # stops the pipeline
        )

pipeline.run()
print('Finished')

