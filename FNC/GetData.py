# -*- coding: utf-8 -*-
"""
This file combines all data loading methods into a central location.
Each type of data has a class that retrieves, processes, and checks it.

Each class has the following methods:
    get - retrieves raw data from a source
    adapt - transforms from the raw data to the common processed format
    check - performs some format checking to see if the processed data looks right
    process - does all the above

Additionally, each class then has source specific handlers.
    E.g. there might be a get_url and a get_csv for a given class
    and then an adapt_phe and an adapt_hps method to format the data

If pulled from an external source (e.g. url), the raw data can be stored 
by setting the config['GenerateOutput']['storeInputs'] flag to be True.
These will be stored in the data/ folder

The processed output can be stored by setting the config['GenerateOutput']['storeProcessedInputs']
flag to be true, which will store the data in processed_data/

@authors:  Jeremy Revell, Thom Kirwan-Evans
"""
import os
import sys

import yaml
import pandas as pd
import re
import requests
import io
import json
import zipfile
from http import HTTPStatus
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime
import pickle
import h5py
import numpy as np
from covid import data as LancsData

# import model_spec
# DTYPE = model_spec.DTYPE
DTYPE = np.float64

def CovarData(config):
    # Return data and covar data structs
    data = {}
    data['areas'] = AreaCodeData.process(config) 
    data['commute'] = InterLadCommuteData.process(config)
    data['cases_tidy'] = CasesData.process(config)
    data['cases_wide'] = data['cases_tidy'].pivot(index="lad19cd", columns="date", values="cases")
    data['mobility'] = MobilityTrendData.process(config)
    data['population'] = PopulationData.process(config)
    data['tier'] = TierData.process(config)

    # Check dimensions are consistent
    check_aligned(data)
    print('Data passes allignment check')
        
    # put it into covar data form
    covar_data = dict(
        C=data['commute'].to_numpy().astype(DTYPE),
        W=data['mobility'].to_numpy().astype(DTYPE),
        N=data['population'].to_numpy().astype(DTYPE),
        L=data['tier'].astype(DTYPE),
        weekday=config['dates']['weekday'].astype(DTYPE),
    )
    return data, covar_data

class TierData:
    def get(config):
        """
        Retrieve an xarray DataArray of the tier data
        """
        settings = config['TierData']
        if settings['input'] == 'csv':
            df = TierData.getCSV(settings['address'])   
        else:
            invalidInput(settings['input'])

        return df

    def getCSV(file):
        """
        Read TierData CSV from file
        """
        return pd.read_csv(file)

    def check(xarray, config):
        """
        Check the data format
        """
        return True

    def adapt(df, config):
        """
        Adapt the dataframe to the desired format. 
        """
        global_settings = config["Global"]
        settings = config["TierData"]

        # this key might not be stored in the config file
        # if it's not, we need to grab it using AreaCodeData
        if 'lad19cds' not in config:
            _df = AreaCodeData.process(config)
        areacodes = config["lad19cds"]

        # Below is assuming inference_period dates
        date_low, date_high = get_date_low_high(config)

        if settings['format'].lower() == 'tidy':
            xarray = TierData.adapt_xarray(df, date_low, date_high, areacodes, settings)

        return xarray

        
    def adapt_xarray(tiers, date_low, date_high, lads, settings):
        """
        Adapt to a filtered xarray object
        """
        tiers["date"] = pd.to_datetime(tiers["date"], format="%Y-%m-%d")
        tiers["code"] = merge_lad_codes(tiers["code"])

        # Separate out December tiers
        date_mask = tiers["date"] > np.datetime64("2020-12-02")
        tiers.loc[
            date_mask & (tiers["tier"] == "three"),
            "tier",
        ] = "dec_three"
        tiers.loc[
            date_mask & (tiers["tier"] == "two"),
            "tier",
        ] = "dec_two"
        tiers.loc[
            date_mask & (tiers["tier"] == "one"),
            "tier",
        ] = "dec_one"

        # filter down to the lads
        if len(lads) > 0:
            tiers = tiers[tiers.code.isin(lads)]

        # add in fake LADs to ensure all lockdown tiers are present for filtering
        # xarray.loc does not like it when the values aren't present
        # this seems to be the cleanest way
        # we drop TESTLAD after filtering down
        #lockdown_states = ["two", "three", "dec_two", "dec_three"]
        lockdown_states = settings['lockdown_states']

        for (i, t) in enumerate(lockdown_states):
            tiers.loc[tiers.shape[0]+i+1] = ['TESTLAD','TEST','LAD',date_low,t]

        index = pd.MultiIndex.from_frame(tiers[["date", "code", "tier"]])
        index = index.sort_values()
        index = index[~index.duplicated()]
        ser = pd.Series(1.0, index=index, name="value")
        ser = ser[date_low : (date_high - np.timedelta64(1, "D"))]
        xarr = ser.to_xarray()
        xarr.data[np.isnan(xarr.data)] = 0.0
        xarr_filt = xarr.loc[..., lockdown_states]
        xarr_filt = xarr_filt.drop_sel({'code':'TESTLAD'})
        return xarr_filt

    def process(config):
        if config['TierData']['format'].lower()[0:5] == 'lancs':
            xarray = TierData.process_lancs(config)
        else:
            df = TierData.get(config)
            xarray = TierData.adapt(df, config)
        if TierData.check(xarray, config):
            return xarray
    
    def process_lancs(config):
        global_settings = config["Global"]
        settings = config["TierData"]
        if 'lad19cds' not in config:
            _df = AreaCodeData.process(config)
        areacodes = config["lad19cds"]
        date_low, date_high = get_date_low_high(config)
        if config['TierData']['format'].lower() == 'lancs_raw':
            return LancsData.read_tier_restriction_data(settings['address'], areacodes, date_low, date_high)
        elif config['TierData']['format'].lower() == 'lancs_tidy':
            return LancsData.read_challen_tier_restriction(settings['address'], date_low, date_high, areacodes)
        else:
            raise NotImplementedError(f'Format type {config["TierData"]["format"]} not implemented')


class CasesData:
    def get(config):
        """
        Retrieve a pandas DataFrame containing the cases/line list data. 
        """
        settings = config['CasesData']
        if settings['input'] == 'url':
            df = CasesData.getURL(settings['address'],config)
        elif settings['input'] == 'csv':
            print('Reading case data from local CSV file at',settings['address'])
            df = CasesData.getCSV(settings['address'])
        elif settings['input'] == 'processed':
            print('Reading case data from preprocessed CSV at', settings['address'])
            df = pd.read_csv(settings['address'],index_col=0)
        else:
            invalidInput(settings['input'])
            
        return df


    def getURL(url, config):
        """
        Placeholder, in case we wish to interface with an API. 
        """
        pass
    

    def getCSV(file):
        """
        Format as per linelisting
        """
        columns = ["pillar", "LTLA_code", "specimen_date", "lab_report_date"]
        dfs = pd.read_csv(file, chunksize=50000, iterator=True, usecols=columns)
        df = pd.concat(dfs)
        return df
    
    def check(df, config):
        """
        Check that data format seems correct
        """
        dims = df.shape
        nareas = len(config["lad19cds"])
        date_low, date_high = get_date_low_high(config)
        dates = pd.date_range(start=date_low,end=date_high,closed="left")
        days = len(dates)
        entries = days * nareas

        if not (((dims[1] >= 3) & (dims[0] == entries)) | ((dims[1] == days) & (dims[0] == nareas))):
            raise ValueError("Incorrect CasesData dimensions")

        if 'date' in df:
            _df = df
        elif df.columns.name == 'date':
            _df = pd.DataFrame({"date":df.columns})
        else:
            raise ValueError("Cannot determine date axis")

        check_date_bounds(df, date_low, date_high)
        check_date_format(df)
        check_lad19cd_format(df)
        return True
    
    def adapt(df, config):
        """
        Adapt the line listing data to the desired dataframe format. 
        """
        # Extract the yaml config settings
        global_settings = config["Global"]
        output_settings = config['GenerateOutput']
        date_low, date_high = get_date_low_high(config)
        settings = config["CasesData"]
        pillars = settings["pillars"]
        measure = settings["measure"].casefold()
        output = settings["output"]
        # this key might not be stored in the config file
        # if it's not, we need to grab it using AreaCodeData
        if 'lad19cds' not in config:
            _df = AreaCodeData.process(config)
        areacodes = config["lad19cds"]

        if settings['input'] == 'processed':
            return df

        if settings['format'].lower() == 'phe':
            df = CasesData.adapt_phe(df, date_low, date_high, pillars, measure, areacodes, output)

        if output_settings['storeProcessedInputs'] and output != "None":
            output = format_output_filename(output,config)
            df.to_csv(output, index=True)

        return df


    def adapt_phe(df, date_low, date_high, pillars, measure, areacodes, output):
        """
        Adapt the line listing data to the desired dataframe format. 
        """
        # Clean missing values
        df.dropna(inplace=True)
        df = df.rename(columns = {"LTLA_code":"lad19cd"})

        # Clean time formats
        df["specimen_date"] = pd.to_datetime(df["specimen_date"], dayfirst=True)
        df["lab_report_date"] = pd.to_datetime(df["lab_report_date"], dayfirst=True)
        
        df["lad19cd"] = merge_lad_codes(df["lad19cd"])

        # filters for pillars, date ranges, and areacodes if given
        filters = df["pillar"].isin(pillars)
        filters &= df["lad19cd"].isin(areacodes)
        if measure == "specimen":
            filters &= (date_low <= df["specimen_date"]) & (df["specimen_date"] < date_high)
        else:
            filters &= (date_low <= df["lab_report_date"]) & (df["lab_report_date"] < date_high)
        df = df[filters]
        df = df.drop(columns="pillar") # No longer need pillar column

        # Aggregate counts
        if measure == "specimen":
            df = df.groupby(["specimen_date", "lad19cd"]).count()
            df = df.rename(columns = {"lab_report_date":"cases"})
        else:
            df = df.groupby(["lab_report_date", "lad19cd"]).count()
            df = df.rename(columns = {"specimen_date":"cases"})

        df.index.names = ["date", "lad19cd"]

        # Fill in all dates, and add 0s for empty counts
        dates = pd.date_range(date_low, date_high, closed="left")
        indexes = [(date, lad19) for date in dates for lad19 in areacodes]
        multi_indexes = pd.MultiIndex.from_tuples(indexes, names=["date", "lad19cd"])
        results = pd.DataFrame(0, index=multi_indexes, columns=["cases"])
        results = results.add(df, axis=0, fill_value=0)
        results = results.reset_index()

        return results

    def process(config):
        if config["CasesData"]["format"].lower() == "lancs":
            df = CasesData.process_lancs(config)
        else:
            df = CasesData.get(config)
            df = CasesData.adapt(df, config)
        if CasesData.check(df, config):
            return df

    def process_lancs(config):
        global_settings = config["Global"]
        settings = config["CasesData"]
        if 'lad19cds' not in config:
            _df = AreaCodeData.process(config)
        areacodes = config["lad19cds"]
        inference_period = [np.datetime64(x) for x in global_settings["inference_period"]]
        date_low = inference_period[0]
        date_high = inference_period[1]
        if ("Pillar 1" in settings["pillars"]) and ("Pillar 2" in settings["pillars"]):
            pillars = "both"
        elif ("Pillar 1" in settings["pillars"]):
            pillars = "1"
        elif ("Pillar 2" in settings["pillars"]):
            pillars = "2"
        dtype = settings["measure"]

        df = LancsData.read_phe_cases(settings['address'], date_low, date_high, 
            pillar=pillars, date_type=dtype, ltlas = areacodes)

        return df.reset_index().melt(['lad19cd']).rename(columns={"value":"cases"})


class MobilityTrendData:
    """
    This is the transport data. The fraction of travel compared to normal levels. 
    """
    
    def get(config):
        """
        Retrieve a response containing the .ods transport data as content.
        """
        settings = config['MobilityTrendData']
        if settings['input'] == 'url':
            df = MobilityTrendData.getURL(settings['address'],config)
        elif settings['input'] == 'ods':
            print('Reading Transport data from local CSV file at',settings['address'])
            df = MobilityTrendData.getODS(settings['address'])
        elif settings['input'] == 'processed':
            print('Reading Transport data from preprocessed CSV at', settings['address'])
            df = pd.read_csv(settings['address'],index_col=0)
            df.date = pd.to_datetime(df.date)
        else:
            invalidInput(settings['input'])

        return df
    
    def getURL(url,config):
        """
        Utility to extract the URL to the DFT transport .ods data.
        """
        settings = config['MobilityTrendData']
        response = requests.get(url)
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise RuntimeError(f'Request failed: {response.text}')

        if settings['format'].lower() == 'dft':
            print("Retrieving transport data from the DfT")
            soup = BeautifulSoup(response.text, "html.parser")
            href = soup.find("a", {"href":re.compile("COVID-19-transport-use-statistics.ods")}).get("href")
            
            response = requests.get(href, timeout=5)
            if response.status_code >= HTTPStatus.BAD_REQUEST:
                raise RuntimeError(f'Request failed: {response.text}')
                
            data = io.BytesIO(response.content)
            
            # store the base data
            if config['GenerateOutput']['storeInputs']:
                fn = format_output_filename(config['GenerateOutput']['scrapedDataDir'] + '/MobilityTrendData_DFT.ods',config)
                with open(fn,'wb') as f: 
                    f.write(data.getvalue())
            
            df = MobilityTrendData.getODS(data)

        return df

    def getODS(file):
        """
        Read DfT ODS file
        """
        return pd.read_excel(file, sheet_name='Transport_use_(GB)', header=6, engine='odf',
            converters={"All motor vehicles2": MobilityTrendData.clean})

    
    
    def check(df, config):
        """
        Check that data format seems correct
        Return True if passes
        Error if not
        """
        dims = df.shape
        date_low, date_high = get_date_low_high(config)
        
        dates = pd.date_range(start=date_low,end=date_high,closed="left")
        days = len(dates)

        if not ((dims[1] >= 1) & (dims[0] == days)): # number of entries
            raise ValueError("Incorrect MobilityData dimensions")

        # our dates are stored in the index column
        # create a new df with just the dates to see
        df_date = pd.DataFrame(df.index)
        check_date_bounds(df_date, date_low, date_high)
        check_date_format(df_date)
        return True


    def clean(x):
        """
        Utility to clean formatting from the table where data has been revised.
        """
        if type(x) == str:
            return float(x.strip("r%"))/100
        else:
            return x


    def adapt(df, config):
        """
        Adapt the transport data to the desired dataframe format.
        """
        global_settings = config["Global"]
        output_settings = config['GenerateOutput']
        date_low, date_high = get_date_low_high(config)
        
        settings = config["MobilityTrendData"]
        output = settings["output"]
        
        if settings['input'] == 'processed':
            return df
        
        if settings['format'].lower() == 'dft':
            df = MobilityTrendData.adapt_dft(df,date_low,date_high,output,config)
        
        if output_settings['storeProcessedInputs'] and output != "None":
            output = format_output_filename(output,config)
            df.to_csv(output, index=True)
            
        return df
    
    
    def adapt_dft(df,date_low,date_high,output,config):
        """
        Adapt the department for Transport data format to a clean Dataframe
        """
        columns = [
            "Date1(weekends and bank holidays in grey)", 
            "All motor vehicles2"
            ]
        colnames = ["date", "percent"]
        df = df[columns]
        df = df.dropna(0)
        df.columns = colnames

        df["date"] = df["date"].apply(lambda x: pd.to_datetime(x, dayfirst=True))

        mask = (df["date"] >= date_low) & (df["date"] < date_high)
        df = df.loc[mask]

        # change the index
        df.set_index('date',inplace=True)

        # set dtype
        df.percent = pd.to_numeric(df.percent)
        
        return df

    def process(config):
        if config['MobilityTrendData']['format'].lower() == "lancs":
            df = MobilityTrendData.process_lancs(config)
        else:
            df = MobilityTrendData.get(config)
            df = MobilityTrendData.adapt(df, config)
        if MobilityTrendData.check(df, config):
            return df

    def process_lancs(config):
        date_low, date_high = get_date_low_high(config)
        return LancsData.read_traffic_flow(
            config['MobilityTrendData']['address'], 
            date_low, date_high)



class PopulationData:
    def get(config):
        """
        Retrieve a response containing the population data from the ONS.
        """
        settings = config['PopulationData']
        if settings['input'] == 'url':
            df = PopulationData.getURL(settings['address'],config)
        elif settings['input'] == 'xls':
            print('Reading Pop. data from local XLS file at',settings['address'])
            df = PopulationData.getXLS(settings['address'])
        elif settings['input'] == 'processed':
            print('Reading Pop. data from preprocessed CSV at', settings['address'])
            df = pd.read_csv(settings['address'],index_col=0)
        else:
            invalidInput(settings['input'])

        return df


    def getURL(url, config):
        """
        Utility to extract the URL to the ONS population .xls data.
        """
        settings = config['PopulationData']
        response = requests.get(url, timeout=5)
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise RuntimeError(f'Request failed: {response.text}')

        if settings['format'] == 'ons':
            print("Retrieving population data from the ONS")
            data = io.BytesIO(response.content)
            
            # store the base data
            if config['GenerateOutput']['storeInputs']:
                fn = format_output_filename(config['GenerateOutput']['scrapedDataDir'] + '/PopulationData_ONS.xls',config)
                with open(fn,'wb') as f: 
                    f.write(data.getvalue())
            
            df = PopulationData.getXLS(data)

        return df


    def getXLS(file):
        """
        Read ONS XLS file
        """
        return pd.read_excel(file, sheet_name='MYE2 - Persons', header=4)
    
    
    def check(df, config):
        """
        Check that data format seems correct
        """
        dims = df.shape
        nareas = len(config["lad19cds"])
        
        if not ((dims[1] >= 1) & (dims[0] == nareas)): # number of entries
            raise ValueError("PopData: Incorrect dimensions")

        check_lad19cd_format(df)
        return True

    def adapt(df, config):
        """
        Adapt the population data to the desired dataframe format.
        """
        output_settings = config['GenerateOutput']
        settings = config["PopulationData"]
        output = settings["output"]
        
        if settings['input'] == 'processed':
            return df

        if settings['format'].lower() == 'ons':
            df = PopulationData.adapt_ons(df,output,config)
        
        if output_settings['storeProcessedInputs'] and output != "None":
            output = format_output_filename(output,config)
            df.to_csv(output, index=True)
            
        return df

    def adapt_ons(df, output, config):
        """
        Adapt the ONS data format to a clean Pandas DataFrame
        """
        lads = [
            'Metropolitan District',
            'Non-metropolitan District',
            'Unitary Authority',
            'London Borough',
            'Council Area',
            'Local Government District'
        ]

        
        if 'lad19cds' not in config:
            _df = AreaCodeData.process(config)
        areacodes = config["lad19cds"]

        columns = ["Code", "All ages"]
        colnames = ["lad19cd", "n"]

        df = df[df.Geography1.isin(lads)]
        df = df[columns]
        df.columns = colnames
        df["lad19cd"] = merge_lad_codes(df["lad19cd"])
        df = merge_lad_values(df)
        filters = df["lad19cd"].isin(areacodes)
        df = df[filters]
        df.set_index('lad19cd',inplace=True)
        

        return df

    def process(config):
        if config['PopulationData']['format'].lower() == "lancs":
            df = PopulationData.process_lancs(config)
        else:
            df = PopulationData.get(config)
            df = PopulationData.adapt(df, config)
        try:
            PopulationData.check(df, config)
            return df
        except:
            print('Population data failed check')
            return df

    def process_lancs(config):
        return LancsData.read_population(config['PopulationData']['address'])



class InterLadCommuteData:
    def get(config):
        """
        Retrieve a response containing the commuting data from Nomisweb.
        """
        settings = config['InterLadCommuteData']
        if settings['input'] == 'url':
            df = InterLadCommuteData.getURL(settings['address'],config)
        elif settings['input'] == 'csv':
            print('Reading Commute data from local CSV file at',settings['address'])
            df = InterLadCommuteData.getCSV(settings['address'])
        elif settings['input'] == 'processed':
            print('Reading Commute data from preprocessed CSV at', settings['address'])
            df = pd.read_csv(settings['address'],index_col=0)
        else:
            invalidInput(settings['input'])

        return df

    def getURL(url, config):
        """
        Utility to extract the URL to the Nomis commuting csv (zipped) data.
        """
        settings = config['InterLadCommuteData']
        response = requests.get(url, timeout=5)
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise RuntimeError(f'Request failed: {response.text}')

        
        if settings['format'].lower() == 'nomis':
            print("Retrieving commute data from NomisWeb")
            with zipfile.ZipFile(io.BytesIO(response.content)) as csvzip: 
                with csvzip.open(csvzip.namelist()[0]) as csv:
                    data = io.BytesIO(csv.read())
    
            if config['GenerateOutput']['storeInputs']:
                fn = format_output_filename(config['GenerateOutput']['scrapedDataDir'] + '/InterLadCommuteData_Nomis.csv',config)
                with open(fn,'wb') as f: 
                    f.write(data.getvalue())
    
            df = InterLadCommuteData.getCSV(data)

        return df

    def getCSV(file):
        """
        Read Nomisweb CSV file
        """
        return pd.read_csv(file)

    
    def check(df, config):
        """
        Check that data format seems correct
        """
        check_lad19cd_format(df)

        dims = df.shape
        nareas = len(config["lad19cds"])
        
        if not ((dims[1] == nareas) & (dims[0] == nareas)): # shape
            raise ValueError("Incorrect dimensions")

        return True

    def adapt(df, config):
        """
        Adapt the commuting data to the desired dataframe matrix format. 
        """
        output_settings = config['GenerateOutput']
        settings = config["InterLadCommuteData"]
        output = settings["output"]
        
        if settings['input'] == 'processed':
            return df

        if settings['format'].lower() == 'nomis':
            df = InterLadCommuteData.adapt_nomis(df,output,config)
        
        if output_settings['storeProcessedInputs'] and output != "None":
            output = format_output_filename(output,config)
            df.to_csv(output, index=True)

        return df

    def adapt_nomis(df, output, config):
        """
        Adapt the Nomis data format to a clean DataFrame
        """
        # Get unique LAD codes
        area_codes = config["lad19cds"]
        df["Area of usual residence"] = AreaCodeData.cmlad11_to_lad19(df["Area of usual residence"])
        df["Area of workplace"] = AreaCodeData.cmlad11_to_lad19(df["Area of workplace"])
        df["Area of usual residence"] = merge_lad_codes(df["Area of usual residence"])
        df["Area of workplace"] = merge_lad_codes(df["Area of workplace"])
        df = df.groupby(["Area of usual residence", "Area of workplace"]).sum().reset_index()
        residence = df["Area of usual residence"].to_list()
        workplace = df["Area of workplace"].to_list()

        # construct a flow matrix
        data_hash = dict(zip(zip(residence, workplace), df["All categories: Method of travel to work"]))
        matrix = [[data_hash.get((residence_code, workplace_code), 0) for workplace_code in area_codes] for residence_code in area_codes]
        df = pd.DataFrame(matrix, columns=area_codes, index=area_codes)
        df.index.rename("lad19cd", inplace=True)

        return df

    def process(config):
        if config['InterLadCommuteData']['format'].lower() == "lancs":
            df = InterLadCommuteData.process_lancs(config)
        else:
            df = InterLadCommuteData.get(config)
            df = InterLadCommuteData.adapt(df, config)
        if InterLadCommuteData.check(df, config):
            return df

    def process_lancs(config):
        return LancsData.read_mobility(config['InterLadCommuteData']['address'])
        
        
"""
Summary Data
This is the prediction data generated by runSummary in summary.py
"""
class SummaryData:
    def get(config):
        settings = config['SummaryData']
        if settings['input'] == 'processed':
            if settings['format'] == 'pickle':
                print('Reading processed summary pickle file')
                data = SummaryData.getPickle(config)
        return data
                
    def getPickle(config):
        settings = config['SummaryData']
        with open(settings['address'],'rb') as file:
            data = pickle.load(file)
        return data
    
    def check(data,config):
        return True
        
    def process(config):
        data = SummaryData.get(config)
        if SummaryData.check(data, config):
            return data
        
"""
Posterior Data
"""

class PosteriorData:
    def get(config):
        settings = config['PosteriorData']
        if settings['format'] == 'h5':
            print('Reading posterior h5 file')
            data = PosteriorData.getH5(config)
        return data
                
    def getH5(config):
        settings = config['PosteriorData']
        data = h5py.File(settings["address"], "r", rdcc_nbytes=1024 ** 3, rdcc_nslots=1e6,
        )
        return data
    
    def check(data,config):
        return True
        
    def process(config):
        data = PosteriorData.get(config)
        if PosteriorData.check(data, config):
            return data
        


"""
Area code data
"""


class AreaCodeData:
    def get(config):
        """
        Retrieve a response containing a list of all the LAD codes
        """

        settings = config['AreaCodeData']
        if settings['input'] == 'url':
            df = AreaCodeData.getURL(settings['address'],config)
        elif settings['input'] == 'json':
            print('Reading Area Code data from local JSON file at',settings['address'])
            df = AreaCodeData.getJSON(settings['address'])
        elif settings['input'] == 'csv':
            print('Reading Area Code data from local CSV file at',settings['address'])
            df = AreaCodeData.getCSV(settings['address'])
        elif settings['input'] == 'processed':
            print('Reading Area Code data from preprocessed CSV at', settings['address'])
            df = pd.read_csv(settings['address'])
        else:
            invalidInput(settings['input'])

        return df

    def getConfig(config):
        # Create a dataframe from the LADs specified in config
        df = pd.DataFrame(config['lad19cds'],columns=['lad19cd'])
        df['name'] = 'n/a' # placeholder names for now. 
        return df

    def getURL(url, config):
        settings = config["AreaCodeData"]

        fields = [
            "LAD19CD",
            "LAD19NM"
        ]

        api_params = {
        'outFields': str.join(',', fields),
        'f': 'json'
        }

        response = requests.get(url, params=api_params, timeout=5)
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise RuntimeError(f'Request failed: {response.text}')
        
        if settings['format'] == 'ons':
            print("Retrieving Area Code data from the ONS")
            data = response.json()
            
            if config['GenerateOutput']['storeInputs']:
                fn = format_output_filename(config['GenerateOutput']['scrapedDataDir'] + 'AreaCodeData_ONS.json',config)
                with open(fn,'w') as f: 
                    json.dump(data,f)

            df = AreaCodeData.getJSON(json.dumps(data))

        return df

    def cmlad11_to_lad19(cmlad11):
        """
        Converts CM (census merged) 2011 codes to LAD 2019 codes
        """
        # The below URL converts from CMLAD2011CD to LAD11CD
#        url = "http://infuse.ukdataservice.ac.uk/showcase/mergedgeographies/Merging-Local-Authorities-Lookup.xlsx"
#        response = requests.get(url, timeout=5)
#        if response.status_code >= HTTPStatus.BAD_REQUEST:
#            raise RuntimeError(f'Request failed: {response.text}')
#
#        data = io.BytesIO(response.content)
#
#        cm11_to_lad11_map = pd.read_excel(data)
        
        # cached
        cm11_to_lad11_map = pd.read_excel('data/Merging-Local-Authorities-Lookup.xlsx')
        
        cm11_to_lad11_dict = dict(zip(cm11_to_lad11_map["Merging Local Authority Code"], cm11_to_lad11_map["Standard Local Authority Code"]))
        
        lad19cds = cmlad11.apply(lambda x: cm11_to_lad11_dict[x] if x in cm11_to_lad11_dict.keys() else x) 

        mapping = {
            "E06000028" : "E06000058", # "Bournemouth" : "Bournemouth, Christchurch and Poole",
            "E06000029" : "E06000058", # "Poole" : "Bournemouth, Christchurch and Poole",
            "E07000048" : "E06000058", # "Christchurch" : "Bournemouth, Christchurch and Poole",
            "E07000050" : "E06000059", # "North Dorset" : "Dorset",
            "E07000049" : "E06000059", # "East Dorset" : "Dorset",
            "E07000052" : "E06000059", # "West Dorset" : "Dorset",
            "E07000051" : "E06000059", # "Purbeck" : "Dorset",
            "E07000053" : "E06000059", # "Weymouth and Portland" : "Dorset",
            "E07000191" : "E07000246", # "West Somerset" : "Somerset West and Taunton",
            "E07000190" : "E07000246", # "Taunton Deane" : "Somerset West and Taunton",
            "E07000205" : "E07000244", # "Suffolk Coastal" : "East Suffolk",
            "E07000206" : "E07000244", # "Waveney" : "East Suffolk",
            "E07000204" : "E07000245", # "St Edmundsbury" : "West Suffolk",
            "E07000201" : "E07000245", # "Forest Heath" : "West Suffolk",

            "E07000097" : "E07000242", # East Hertforshire
            "E07000101" : "E07000243", # Stevenage
            "E07000100" : "E07000240", # St Albans
            "E08000020" : "E08000037", # Gateshead
            "E06000048" : "E06000057", # Northumberland
            "E07000104" : "E07000241", # Welwyn Hatfield
            }

        lad19cds = lad19cds.apply(lambda x: mapping[x] if x in mapping.keys() else x)
        lad19cds = merge_lad_codes(lad19cds)

        return lad19cds


    def getJSON(file):
        data = pd.read_json(file, orient="index").T["features"][0]
        data = [record["attributes"] for record in data]
        df = pd.DataFrame.from_records(data)
        return df

    def getCSV(file):
        return pd.read_csv(file)

    
    def check(df, config):
        """
        Check that data format seems correct
        """
        check_lad19cd_format(df)
        return True

    
    def adapt(df, config):
        """
        Adapt the area codes to the desired dataframe format
        """
        output_settings = config['GenerateOutput']
        settings = config["AreaCodeData"]
        output = settings["output"]
        regions = settings["regions"]

        if settings['input'] == 'processed':
            return df

        if settings['format'].lower() == 'ons':
            df = AreaCodeData.adapt_ons(df, regions, output, config)

        # if we have a predefined list of LADs, filter them down
        if 'lad19cds' in config:
            df = df[[x in config['lad19cds'] for x in df.lad19cd.values]]
        
        if output_settings['storeProcessedInputs'] and output != "None":
            output = format_output_filename(output, config)
            df.to_csv(output, index=False)

        return df

    def adapt_ons(df, regions, output, config):
        colnames = ["lad19cd", "name"]
        df.columns = colnames
        filters = df["lad19cd"].str.contains(str.join("|", regions))
        df = df[filters]
        df["lad19cd"] = merge_lad_codes(df["lad19cd"])
        df = df.drop_duplicates(subset="lad19cd")

        return df

    def process(config):
        df = AreaCodeData.get(config)
        df = AreaCodeData.adapt(df, config)
        if AreaCodeData.check(df, config):
            config["lad19cds"] = df["lad19cd"].tolist()
            return df


#%%  Helper functions
def prependDate(filename):
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d")
    return date_time + '_' + filename
    
def prependID(filename,config):
    return config['Global']['prependID_Str'] + '_' + filename

def format_input_filename(filename,config):
    # prepend with a set string
    # to load a specific date, this should be in the string
    p,f = os.path.split(filename)
    if config['Global']['prependID']:
        f = prependID(f,config)
    filename = p + '/' + f
    return filename

def format_output_filename(filename,config):
    p,f = os.path.split(filename)
    if config['Global']['prependID']:
        f = prependID(f,config)
    if config['Global']['prependDate']:
        f = prependDate(f)
    filename = p + '/' + f
    return filename

def merge_lad_codes(lad19cd):
    merging = {
            "E06000052" : "E06000052,E06000053", # City of London & Westminster
            "E06000053" : "E06000052,E06000053", # City of London & Westminster
            "E09000001" : "E09000001,E09000033", # Cornwall & Isles of Scilly
            "E09000033" : "E09000001,E09000033"  # Cornwall & Isles of Scilly
        }
    lad19cd = lad19cd.apply(lambda x: merging[x] if x in merging.keys() else x)

    return lad19cd

def merge_lad_values(df):
    df = df.groupby("lad19cd").sum().reset_index()
    return df

def get_date_low_high(config):
    if 'dates' in config:
        low = config['dates']['low']
        high = config['dates']['high']
    else:
        inference_period = [np.datetime64(x) for x in config['Global']['inference_period']]
        low = inference_period[0]
        high = inference_period[1]
    return (low, high)


#%%  Single data testing functions
def check_date_format(df):
    df = df.reset_index()

    if not pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').notnull().all():
        raise ValueError("Invalid date format")

    return True

def check_date_bounds(df, date_low, date_high):
    if not ((date_low <= df["date"]) & (df["date"] < date_high)).all():
        raise ValueError("Date out of bounds")
    return True


def check_lad19cd_format(df):
    df = df.reset_index()

    # Must contain 9 characters, 1 region letter followed by 8 numbers
    split_code = df["lad19cd"].apply(lambda x: re.split('(\d+)',x))
    if not split_code.apply(lambda x: \
        (len(x[0])==1) & \
        (x[0] in "ENSW") & \
        (len(x[1])==8) \
        ).all():
        raise ValueError("Invalid lad19cd format")

    return True

def invalidInput(input):
    raise NotImplementedError(f'Input type "{input}" mode not implemented')

#%% Multi data testing functions
def check_aligned(dfs):
    # First check we have all the data, assuming dfs is a dict
    sources = ['areas', 'commute', 'cases_tidy', 'cases_wide', 
        'mobility', 'population', 'tier']

    if not all(source in dfs.keys() for source in sources):
        missing = set(sources) - set(dfs.keys())
        raise ValueError(f"Missing {missing} data source(s)")

    # Check LAD dims align
    lad_check = len(dfs['areas']) == len(dfs['cases_wide']) == \
        len(dfs['population']) == dfs['commute'].shape[0] == \
        dfs['commute'].shape[1] == len(dfs['tier'].code) == \
        len(dfs['cases_tidy']['lad19cd'].unique())

    if not lad_check:
        raise ValueError(f"Mismatching LAD dimensions")

    # Check date dims align
    date_check = len(dfs['cases_wide'].columns) == \
        len(dfs['mobility'].index) == len(dfs['tier'].date) == \
        len(dfs['cases_tidy']['date'].unique())

    if not date_check:
        raise ValueError(f"Mismatching date dimensions")


if __name__ == "__main__":
    with open("./configs/config_pseudo_scrape.yml", 'r') as f:
        config = yaml.safe_load(f)

    # with open("./configs/config_lancs.yml", 'r') as f:
    #     config = yaml.safe_load(f)

    # with open("./configs/config_local.yml", 'r') as f:
    #     config = yaml.safe_load(f)

    print("Retrieving area code data from the ONS")
    df_areas = AreaCodeData.process(config)
    print(df_areas)

    print("Retrieving Tiers data")
    xarray = TierData.process(config)
    print(xarray)

    print("Retrieving transport data from the DfT")
    df_mobility = MobilityTrendData.process(config)
    print(df_mobility)

    print("Retrieving population data from the ONS")
    df_population = PopulationData.process(config)
    print(df_population)

    print("Retrieving commuting data from Nomisweb")
    df_commute = InterLadCommuteData.process(config)
    print(df_commute)

    print("Opening Anon Line List file")
    df_cases = CasesData.process(config)
    print(df_cases)

    dfs = {'areas':df_areas, 
        'commute':df_commute,
        'cases_tidy':df_cases,
        'cases_wide':df_cases.pivot(index="lad19cd", columns="date", values="cases"), 
        'mobility':df_mobility, 
        'population':df_population, 
        'tier':xarray}

    check_aligned(dfs)
