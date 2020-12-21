# -*- coding: utf-8 -*-
"""

Functions to perform inspection of both the input data and predictions
They can be run at any stage of the pipeline to ensure that everything is being handled correctly.
Not limited to plotting routines, but that's what we've implemented at present.

@authors: Thom Kirwan-Evans, Jeremy Revell
"""
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from FNC.GetData import format_output_filename
import numpy as np
import os
import datetime as dt

def setBackend(config):
    # If we aren't showing plots (recommended), we need to 
    # set matplotlib to only buffer the plots
    if not config['Inspect']['show_plots']:
        matplotlib.use('Agg')
        
def InspectInputData(data, config):
    # Check the inout covar data
    print('Inspecting Input Data')
    plotCases(data['cases_tidy'],config)
    plotMobilityTrend(data['mobility'],config) 
    plotTiers(data['tier'],config)

def plotCases(cases, config):
    # Plots all total daily cases
    x = np.unique(cases.date)
    y = []
    for d in x:
        y.append(cases.cases[cases.date==d].sum())
    # x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in x]

    plotTimeSeries(x, y, config, 'daily_cases.png')

def plotMobilityTrend(mobility, config):
    # Plots mobility time series
    mobility = mobility.reset_index()

    try:
        x = mobility['date']
    except:
        x = mobility['index']
        
    y = mobility['percent']
    # try:
    #     x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in x]
    # except:
    #     # Workaround for example data date format
    #     x = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in x]

    plotTimeSeries(x, y, config, 'mobility_trend.png')

def plotTiers(tiers, config):
    # Plots the change in tier status for a given LAD
    LAD = config['Inspect']['exampleLADix']
    tv_all = tiers.values
    states = config['TierData']['lockdown_states']
    states = np.hstack(('Other',states))
    dates = tiers.date.to_dataframe().date.values
    tv = tv_all[:,LAD,:].squeeze()
    ld = tv.argmax(axis=1)+1
    less_than_lockdownstates = tv.sum(axis=1)==0
    ld[less_than_lockdownstates] = 0
    x = dates
    y = ld
    fig, ax = plt.subplots()
    ax.plot(x, y)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.yticks(range(len(states)),states)
    ax.grid(True)
    plt.title('LAD ' + config['Inspect']['exampleLAD'] + ' Lockdown state')
    if config['Inspect']['show_plots']:
        plt.show()
    if config['Inspect']['save_plots']:
        saveFigure('Tier_state.png',config)

def plotInterLADCommuteMatrix(commute,config):
    pass

def InspectSummaryData(summaryData,config):
    print('Inspecting Summary Data')
    plotSummaryAllLADs(summaryData,config)
    plotSummarySingleLAD(summaryData,config)


def plotSummaryAllLADs(summaryData,config):
    # Plot predicted prevalence and cases for all LADs
    states = list(summaryData['cases']) # this should be in order
    casedata = summaryData['cases']['now']['cases_mean'].numpy()
    prevdata = summaryData['prev']['now']['prev_mean'].numpy()
    dates = np.zeros((len(states),casedata.size))
    for i,s in enumerate(states[1:]):
        key = 'cases' + s + '_mean'
        casedata = np.vstack((casedata, summaryData['cases'][s][key].numpy()))
        key = 'prev' + s + '_mean'
        prevdata = np.vstack((prevdata, summaryData['prev'][s][key].numpy()))
        dates[i+1,:] = int(s)


    # plot Cases
    plt.figure()
    plt.plot(dates,casedata,color='lightgray')
    plt.grid()
    plt.xlabel('Days from ' + str(config['Global']['prediction_period'][0]))
    plt.ylabel('Cases (Mean)')
    plt.title('LAD Case Predictions')
    if config['Inspect']['show_plots']:
        plt.show()
    if config['Inspect']['save_plots']:
        saveFigure('Predicted_Cases_All_LADs.png',config)

    # plot prevalence
    plt.figure()
    plt.plot(dates,prevdata,color='lightgray')
    plt.grid()
    plt.xlabel('Days from ' + str(config['Global']['prediction_period'][0]))
    plt.ylabel('Prev (Mean)')
    plt.title('LAD Prev Predictions')
    if config['Inspect']['show_plots']:
        plt.show()
    if config['Inspect']['save_plots']:
        saveFigure('Predicted_Prev_All_LADs.png',config)

def plotSummarySingleLAD(summaryData,config):
    # Plot an inidividual LADs mean and range
    LAD = config['Inspect']['exampleLADix']
    low_key = '0.025'
    high_key = '0.975'
    states = list(summaryData['cases']) # this should be in order
    casedata = summaryData['cases']['now']['cases_mean'][LAD].numpy()
    prevdata = summaryData['prev']['now']['prev_mean'][LAD].numpy()
    casedata_low = summaryData['cases']['now']['cases_' + low_key][LAD].numpy()
    prevdata_low = summaryData['prev']['now']['prev_' + low_key][LAD].numpy()
    casedata_high = summaryData['cases']['now']['cases_' + high_key][LAD].numpy()
    prevdata_high = summaryData['prev']['now']['prev_' + high_key][LAD].numpy()
    dates = np.zeros((len(states),casedata.size))
    for i,s in enumerate(states[1:]):
        key = 'cases' + s + '_'
        casedata = np.vstack((casedata, summaryData['cases'][s][key + 'mean'][LAD].numpy()))
        casedata_low = np.vstack((casedata_low, summaryData['cases'][s][key + low_key][LAD].numpy()))
        casedata_high = np.vstack((casedata_high, summaryData['cases'][s][key + high_key][LAD].numpy()))
        key = 'prev' + s + '_'
        prevdata = np.vstack((prevdata, summaryData['prev'][s][key + 'mean'][LAD].numpy()))
        prevdata_low = np.vstack((prevdata_low, summaryData['prev'][s][key + low_key][LAD].numpy()))
        prevdata_high = np.vstack((prevdata_high, summaryData['prev'][s][key + high_key][LAD].numpy()))
        dates[i+1,:] = int(s)

    # plot Cases
    plt.figure()
    plt.plot(dates,casedata,'-o',color='k')
    plt.plot(dates,casedata_low,'--',color='lightgray')
    plt.plot(dates,casedata_high,'--',color='lightgray')
    plt.grid()
    plt.xlabel('Days from ' + str(config['Global']['inference_period'][-1]))
    plt.ylabel('Cases (Mean)')
    plt.title('LAD Case Predictions')
    plt.legend(['Mean',low_key,high_key])
    if config['Inspect']['show_plots']:
        plt.show()
    if config['Inspect']['save_plots']:
        saveFigure('Predicted_Cases_Single_LAD.png',config)

    # Plot prevalence
    plt.figure()
    plt.plot(dates,prevdata,'-o',color='k')
    plt.plot(dates,prevdata_low,'--',color='lightgray')
    plt.plot(dates,prevdata_high,'--',color='lightgray')
    plt.grid()
    plt.xlabel('Days from ' + str(config['Global']['inference_period'][-1]))
    plt.ylabel('Prev (Mean)')
    plt.title('LAD Prev Predictions')
    plt.legend(['Mean',low_key,high_key])
    if config['Inspect']['show_plots']:
        plt.show()
    if config['Inspect']['save_plots']:
        saveFigure('Predicted_Prev_Single_LAD.png',config)

def plotTimeSeries(x, y, config, filename):
    # Common functionality for plotting timeseries data.
    # x is a dt.datetime list
    fig, ax = plt.subplots()
    ax.plot(x, y)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True)

    if config['Inspect']['show_plots']:
        plt.show()
    
    if config['Inspect']['save_plots']:
        saveFigure(filename,config)

def saveFigure(filename,config):
    # Save the figure into the plot directory
    fn = format_output_filename(
            os.path.join(config['Inspect']['plot_dir'], filename),
            config)
    print(fn)
    plt.savefig(fn, dpi=600)

