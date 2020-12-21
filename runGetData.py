# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:44:10 2020

@author: tak
"""

from FNC.GetData import *
import pandas as pd
import yaml
from pandas.util.testing import assert_frame_equal
import geopandas as gp
import numpy as np

from FNC import GetData, COVIDInspect

#print('Scraping data...')
#with open("./configs/config_scrape.yml", 'r') as f:
#    config = yaml.safe_load(f)
#df_transport_1 = TransportData.process(config)
#df_pop_1 = PopulationData.process(config)
#df_commute_1 = CommuteData.process(config)
#
#print('Testing local data...')
#with open("./configs/config_local.yml", 'r') as f:
#    config = yaml.safe_load(f)
#df_transport_2 = TransportData.process(config)
#df_pop_2 = PopulationData.process(config)
#df_commute_2 = CommuteData.process(config)

#print('Testing processed data...')
#with open("./configs/config_processed.yml", 'r') as f:
#    config = yaml.safe_load(f)
#df_transport_3 = TransportData.process(config)
#df_pop_3 = PopulationData.process(config)
#df_commute_3 = CommuteData.process(config)
#
## Check all are equal
#assert_frame_equal(df_transport_1,df_transport_2)
#assert_frame_equal(df_transport_1,df_transport_3)
#assert_frame_equal(df_pop_1,df_pop_2)
#assert_frame_equal(df_pop_1,df_pop_3)
#assert_frame_equal(df_commute_1,df_commute_2)
#assert_frame_equal(df_commute_1,df_commute_3)
#

#with open("./configs/config_pipeline.yml", 'r') as f:
#    config = yaml.safe_load(f)
#pipelineData = {'data': {}}
#pipelineData['data']['areas'] = GetData.AreaCodeData.process(config) 
#pipelineData['data']['commute'] = GetData.CommuteData.process(config)
#pipelineData['data']['cases_tidy'] = GetData.CasesData.process(config)
#pipelineData['data']['cases_wide'] = pipelineData['data']['cases_tidy'].pivot(index="lad19cd", columns="date", values="cases")
#pipelineData['data']['mobility'] = GetData.TransportData.process(config)
#pipelineData['data']['population'] = GetData.PopulationData.process(config)


with open("./configs/config_pipeline.yml", 'r') as f:
    config = yaml.safe_load(f)

inference_period = [np.datetime64(x) for x in config["Global"]["inference_period"]]
date_low = inference_period[0]
date_high = inference_period[1]
weekday = pd.date_range(date_low, date_high).weekday < 5
config['dates'] = {
    'inference_period':inference_period,
    'low':date_low,
    'high':date_high,
    'weekday':weekday
}

data = {}
data['areas'] = AreaCodeData.process(config) 
config['lad19cds'] = data['areas'].lad19cd.values
data['commute'] = InterLadCommuteData.process(config)
data['cases_tidy'] = CasesData.process(config)
data['cases_wide'] = data['cases_tidy'].pivot(index="lad19cd", columns="date", values="cases")
data['mobility'] = MobilityTrendData.process(config)
data['population'] = PopulationData.process(config)
data['tier'] = TierData.process(config)

covar_data = dict(
        C=data['commute'].to_numpy().astype(DTYPE),
        W=data['mobility'].to_numpy().astype(DTYPE),
        N=data['population'].to_numpy().astype(DTYPE),
        L=data['tier'].astype(DTYPE),
        weekday=config['dates']['weekday'].astype(DTYPE),
    )

print('Read it!')

# mobility = pd.read_csv(config['ExampleFiles']['MobilityTrendData'], index_col=0)
# popsize = pd.read_csv(config['ExampleFiles']['PopulationData'], index_col=0)
# commute_volume = pd.read_csv(config['ExampleFiles']['InterLadCommuteData'], index_col=0)
# cases_tidy = pd.read_csv(config['ExampleFiles']['CasesData'])
# cases_wide = cases_tidy.pivot(index="lad19cd", columns="date", values="cases")

# pipelineData = {'data': {}}
# pipelineData['data']['commute'] = commute_volume
# pipelineData['data']['cases_tidy'] = cases_tidy
# pipelineData['data']['cases_wide'] = cases_wide
# pipelineData['data']['mobility'] = mobility
# pipelineData['data']['population'] = popsize

# # inspect.InspectInputData(pipelineData['data'],config)

# ## fix areacode / TIER stuff
# # areacode = AreaCodeData.process(config)
# # geo = gp.read_file('data/UK2019mod_pop.gpkg')
# # geo = geo.loc[geo["lad19cd"].str.startswith("E")]

# import covid.data as data

# inference_period = [np.datetime64(x) for x in config["Global"]["inference_period_long"]]
# date_low=inference_period[0]
# date_high=inference_period[1]
# #tier_restriction = data.read_challen_tier_restriction(config['TierData']['address'],date_low,date_high)



# tier_restriction_csv = config['TierData']['address']
# tiers = pd.read_csv(tier_restriction_csv)
# tiers["date"] = pd.to_datetime(tiers["date"], format="%Y-%m-%d")
# tiers["code"] = _merge_ltla(tiers["code"])

# # Separate out December tiers
# tiers.loc[
#     (tiers["date"] > np.datetime64("2020-12-02"))
#     & (tiers["tier"] == "three"),
#     "tier",
# ] = "dec_three"
# tiers.loc[
#     (tiers["date"] > np.datetime64("2020-12-02"))
#     & (tiers["tier"] == "two"),
#     "tier",
# ] = "dec_two"
# tiers.loc[
#     (tiers["date"] > np.datetime64("2020-12-02"))
#     & (tiers["tier"] == "one"),
#     "tier",
# ] = "dec_one"

# # add in fake LADs to ensure all lockdown tiers are present for filtering
# lockdown_states = ["two", "three", "dec_two", "dec_three"]
# for t in lockdown_states:
#     tiers.loc[tiers.shape[0]+1] = ['TESTLAD','TEST','LAD',date_low,t]

# index = pd.MultiIndex.from_frame(tiers[["date", "code", "tier"]])
# index = index.sort_values()
# index = index[~index.duplicated()]
# ser = pd.Series(1.0, index=index, name="value")
# ser = ser[date_low : (date_high - np.timedelta64(1, "D"))]
# xarr = ser.to_xarray()
# xarr.data[np.isnan(xarr.data)] = 0.0
# # this needs to be handled better if these values aren't in the array
# xarr_filt = xarr.loc[..., ["two", "three", "dec_two", "dec_three"]]
# xarr_filt = xarr_filt.drop_sel({'code':'TESTLAD'})

# import xarray as xr
# tiers_in_timeframe = list(tiers.tier.unique())
# null_array = xarr.loc[..., [tiers_in_timeframe[0]]]*0
# null_coords = null_array.coords
# lockdown_states = ["two", "three", "dec_two", "dec_three"]
# xarr_filt
# xr.concat([xarr.loc[..., ["two"]],xarr.loc[..., ["three"]]],dim='tier')
# xarrs = []
# for tier in lockdown_states:
#     if tier in 
# xarr_two = xarr.loc[..., ["two"]]+1
# xarr_three = xarr.loc[..., ["three"]]

#
#print("Opening Anon Line List file")
## Reduce memory footprint
#columns = ["pillar", "LTLA_code", "specimen_date", "lab_report_date"]
##    dfs = pd.read_csv("//fnc.domain//project/BRS/ProjB/010144/GLOBAL/TECHNICAL/3_Received_Data/Anonymised Combined Line List 20201016.csv", 
##        chunksize=50000, iterator=True, usecols=columns)
#dfs = pd.read_csv("C:/Scratch/Projects/SAGE/Pipeline/data/Anonymised Combined Line List 20201016.csv", 
#    chunksize=50000, iterator=True, usecols=columns)
#df = pd.concat(dfs)
#print("Process Anon Line List data")
#df_cases = CasesData.adapt(df, config)
#
