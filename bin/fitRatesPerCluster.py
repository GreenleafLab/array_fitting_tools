#!/usr/bin/env python
""" Fit on or off rates.

Using the binned times & binding series from binTimes.py,
bootstraps the medians to fit the on or off rates.

Sarah Denny """

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seqfun
import IMlibs
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import lmfit
from scikits.bootstrap import bootstrap
import fitFun
from fitFun import objectiveFunctionOffRates, objectiveFunctionOnRates
import itertools
import warnings
import fileFun
import singleClusterFits
import findFmaxDist

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap on off rate fits')
parser.add_argument('-cs', '--cpseries', metavar="CPseries.pkl",
                   help='CPseries file containining the fluorescence information')
parser.add_argument('-t', '--tiles', metavar="CPtiles.pkl",
                   help='CPtiles file giving the tile per cluster')
parser.add_argument('-td', '--time_dict', metavar="timeDict.p",
                   help='file containining the per-tile times')
parser.add_argument('-ft', '--fittype', default='off', metavar="[off | on]",
                   help='fittype ["off" | "on"]. Default is "off" for off rates')

group = parser.add_argument_group('additional option arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of -cs (CPseries) filename')
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores. default = 20')
group.add_argument('--subset',action="store_true", default=False,
                    help='if flagged, will only do a subset of the data for test purposes')

 

def getInitialParameters(times, fittype=None):
    """ Get standard set of fit parameters across all variants depending on fittype."""
    
    # default fittype is 'off'
    if fittype is None: fittype = 'off'
    
    # different lifetime name for association vs dissociation
    if fittype == 'off':
        param_name = 'koff'
    elif fittype == 'on':
        param_name = 'kobs'
    else:
        print 'Error: "fittype" not recognized. Valid options are "on" or "off".'
        sys.exit()
        
    # fit parameters structure has three parameters (fmax, lifeftime, fmin),
    # each has a lowerbound, upperbound, and initialization
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', param_name, 'fmin'])
    
    # fmin cannot be less than zero. Initial condition will be set based on
    # min median fluorescence of variant
    param = 'fmin'
    fitParameters.loc[:, param] = [0, 0, np.inf]
    
    # fmax cannot be less than zero. Initial condition will be set based on
    # max median fluorescence of variant
    param = 'fmax'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    
    # rate parameter constraints.
    # (1) min lifetime (slowest) is that given by only 1% dropoff after the full time span.
    # (2) initial lifteime is that given by 50% dropoff within half the time span.
    # (3) max lifetime (fastest) is that given by 99% dropoff after 1/10 of the first
    # time interval.
    # note that 'min fraction bound' is 1 - the dropoff amount quoted above.
    fitParameters.loc[:, param_name] = [-np.log(min_fraction_decreased)/t_delta
     for t_delta, min_fraction_decreased in itertools.izip(
        [times.max()-times.min(), (times.max()-times.min())/2, (times[1]-times[0])/10.],
        [0.99,                    0.5,                         0.01])]

    return fitParameters

def splitAndFit(bindingSeries, timeDict, tileSeries, fitParameters, numCores,
                index=None, change_params=None, func=None):
    """ Given a table, split by tile and fit. """
    if index is None:
        index = bindingSeries.index
    
    # split into parts
    print 'Splitting clusters into %d groups:'%numCores
    tiles = timeDict.keys()
    bindingSeriesSplit = {}
    for tile in tiles:
        mat = bindingSeries.loc[index].loc[tileSeries.loc[index]==tile].dropna(axis=0, thresh=5).copy()
        if len(mat) > 0:
            bindingSeriesSplit[tile] = mat

    printBools = [True] + [False]*(numCores-1)

    print 'Fitting binding curves:'
    fits = (Parallel(n_jobs=numCores, verbose=10)
            (delayed(singleClusterFits.fitSetClusters)
             (np.array(timeDict[tile]), bindingSeriesSplit[tile], fitParameters, printBools[i], change_params, func)
             for i, tile in enumerate(bindingSeriesSplit.keys())))

    return pd.concat(fits)

def getMeanTimes(timeDict):
    binwidth = np.min([(np.sort(times)[1:] - np.sort(times)[:-1]).min() for times in timeDict.values()])
    binstart = np.min([np.min(times) for times in timeDict.values()])
    binend = np.max([np.max(times) for times in timeDict.values()])
    return np.arange(binstart, binend+binwidth, binwidth)

##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()
    outFile  = args.out_file
    tileFile = args.tiles
    timeDeltaFile = args.time_dict
    bindingSeriesFile = args.cpseries
    fittype = args.fittype
    numCores = args.numCores
    
    # find out file
    if outFile is None:
        outFile = fileFun.stripExtension(bindingSeriesFile)
        
    # depending on fittype, use one of two provided objective functions
    if fittype == 'off':
        func = objectiveFunctionOffRates
    elif fittype == 'on':
        func = objectiveFunctionOnRates
    else:
        print ('Error: fittype "%s" not recognized. Valid options are '
               '"on" or "off".')%fittype
        sys.exit()
        
    # laod data
    print 'Loading data..'
    # load time series data
    bindingSeries = fileFun.loadFile(bindingSeriesFile)
    timeDict = fileFun.loadFile(timeDeltaFile)
    tileSeries = fileFun.loadFile(tileFile)
    
    # find fitParameters
    mean_times = getMeanTimes(timeDict)
    fitParameters = getInitialParameters(mean_times,
                                         fittype=fittype)
    # fit gamm distribution for last binding point
    fitParameters.loc['initial', 'fmin'] = (
        findFmaxDist.fitGammaDistribution(bindingSeries.iloc[:, -1].dropna(), plot=True, set_offset=0).loc['mean'])
    
    # subset if option is given
    if args.subset:
        num = 5E3
        index = bindingSeries.index[::num]
        outFile = outFile + '_subset'
    else:
        index = bindingSeries.index

    # fit single clusters
    cluster_data = splitAndFit(bindingSeries, timeDict, tileSeries, fitParameters, numCores,
                index=index, change_params=True, func=func)
    cluster_data.to_pickle(outFile+'.CPfitted.pkl')




    
