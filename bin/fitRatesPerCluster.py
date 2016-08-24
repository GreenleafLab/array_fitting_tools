#!/usr/bin/env python
""" Fit on or off rates.

Sarah Denny """

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import itertools
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import lmfit
from fittinglibs import fitting, distribution, fileio

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
group.add_argument('--pb_correct',default = 1, type=float, metavar="N",
                    help='use this value for photobleaching amount per image')
group.add_argument('-id', '--image_n_dict', metavar="imageNDict.p",
                   help='file containining the per-tile image number to use for pb correction')

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

def findDefaultImageNs(timeDict):
    imageNDict = {}
    for tile, times in  timeDict.items():
        imageNDict[tile] = np.arange(len(times))
    return imageNDict

def splitAndFit(bindingSeries, timeDict, tileSeries, fitParameters, numCores,
                index=None, change_params=None, func=None, bleach_fraction=None,
                imageNDict=None):
    """ Given a table, split by tile and fit. """
    if index is None:
        index = bindingSeries.index
    
    if imageNDict is None:
        imageNDict = findDefaultImageNs(timeDict)

    # split into parts
    print 'Splitting clusters into %d groups:'%numCores
    tiles = timeDict.keys()
    numTiles = len(tiles)
    sizePerTile = tileSeries.loc[index].value_counts()
    avgSplitPerTile = sizePerTile/sizePerTile.sum()*numCores
    splitPerTile = np.around(avgSplitPerTile)
    
    # change number of splits per tile to make sure number of splits corresponds to number of cores
    if splitPerTile.sum() > numCores:
        diff = splitPerTile.sum() - numCores
        index_to_subtract = avgSplitPerTile.loc[splitPerTile > 1].order().index[:diff]
        splitPerTile.loc[index_to_subtract] -= 1
    elif splitPerTile.sum() < numCores:
        diff = numCores - splitPerTile.sum()
        index_to_add = avgSplitPerTile.order(ascending=False).index[:diff]
        splitPerTile.loc[index_to_add] += 1

    # split CPseries    
    bindingSeriesSplit = {}
    for tile, numSplits in splitPerTile.iteritems():
        mat = bindingSeries.loc[index].loc[tileSeries.loc[index]==tile].dropna(axis=0, thresh=5).copy()
        indicesSplit = np.array_split(mat.index.tolist(), numSplits)
        for i, indices in enumerate(indicesSplit):
            bindingSeriesSplit[(tile, i)] = mat.loc[indices]
            
    printBools = [True] + [False]*(numCores-1)

    print 'Fitting binding curves:'
    fits = (Parallel(n_jobs=numCores, verbose=10)
            (delayed(fitting.fitSetClusters)
             (np.array(timeDict[tile]), bindingSeriesSplit[(tile, idx)], fitParameters,
              printBools[i], change_params, func, kwargs={'bleach_fraction':bleach_fraction,
                                                          'image_ns':imageNDict[tile]})
             for i, (tile, idx) in enumerate(bindingSeriesSplit.keys())))

    return pd.concat(fits)

def getMeanTimes(timeDict):
    binwidth = np.min([(np.sort(times)[1:] - np.sort(times)[:-1]).min()
        for times in timeDict.values() if len(times)>1])
    binstart = np.min([np.min(times) for times in timeDict.values() if len(times)>0])
    binend = np.max([np.max(times) for times in timeDict.values() if len(times)>0])
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
    bleach_fraction = args.pb_correct
    imageNFile = args.image_n_dict
    
    # find out file
    if outFile is None:
        outFile = fileio.stripExtension(bindingSeriesFile)
        
    if fittype == 'off':
        func = fitting.objectiveFunctionOffRates
    elif fittype == 'on':
        func = fitting.objectiveFunctionOnRates
    else:
        print ('Error: fittype "%s" not recognized. Valid options are '
               '"on" or "off".')%fittype
        sys.exit()
        
    # laod data
    print 'Loading data..'
    # load time series data
    bindingSeries = fileio.loadFile(bindingSeriesFile)
    timeDict = fileio.loadFile(timeDeltaFile)
    tileSeries = fileio.loadFile(tileFile)
    if imageNFile is not None:
        imageNDict = fileio.loadFile(imageNFile)
    else:
        imageNDict = None
    # find fitParameters
    mean_times = getMeanTimes(timeDict)
    fitParameters = getInitialParameters(mean_times,
                                         fittype=fittype)
    # fit gamm distribution for last binding point
    if fittype=="off":
        fitParameters.loc['initial', 'fmin'] = (
            distribution.fitGammaDistribution(bindingSeries.dropna(how='all', axis=1).iloc[:, -1].dropna(), plot=True, set_offset=0).loc['mean'])
    else:
        fitParameters.loc['initial', 'fmin'] = (
            distribution.fitGammaDistribution(bindingSeries.iloc[:, 0].dropna(), plot=True, set_offset=0).loc['mean'])
      
    # subset if option is given
    if args.subset:
        num = 5E3
        index = bindingSeries.index[::num]
        outFile = outFile + '_subset'
    else:
        index = bindingSeries.index
    
    # fit single clusters
    cluster_data = splitAndFit(bindingSeries, timeDict, tileSeries, fitParameters, numCores,
                index=index, change_params=True, func=func, bleach_fraction=bleach_fraction, imageNDict=imageNDict)
    cluster_data.to_pickle(outFile+'.CPfitted.pkl')

    sys.exit()
