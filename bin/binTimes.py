#!/usr/bin/env python
#
# bin times into time bins with time delta equal to
# minimum time between measurements on any single tile.
#
# Sarah Denny

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
import fitFun
import itertools
import plotFun
import fileFun


### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bin time series by time')

parser.add_argument('-cs', '--cpseries', metavar="CPseries.pkl",
                   help='CPseries file containining the time series information')
parser.add_argument('-t', '--tiles', metavar="CPtiles.pkl",
                   help='CPtiles file containining the tile per cluster')
parser.add_argument('-td', '--time_dict', metavar="timeDict.p",
                   help='file containining the timing information per tile')
parser.add_argument('-o', '--out_file', 
                   help='output basename')

group = parser.add_argument_group('optional arguments for binning')
parser.add_argument('-tau', '--tau', type=float, metavar="n",
                   help='number of seconds separating bins. Default is to find '
                   'min time between tiles.')


##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()

    outFile = args.out_file
    timeSeriesFile = args.cpseries
    timeDelta = fileFun.loadFile(args.time_dict)
    tileFile = args.tiles
    
    # find out file
    if outFile is None:
        outFile = fileFun.stripExtension(timeSeriesFile)
    figDirectory = os.path.join(os.path.dirname(outFile), fileFun.returnFigDirectory())
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    
    # process output filenames
    timesFilename = outFile + '.times'

    # load time Series and tiles
    
    # time series is organized with rows=clusterIDs and columns 0-number of time points.
    # columns in different tiles can mean very different times.
    timeSeries = fileFun.loadFile(timeSeriesFile)
    tiles = fileFun.loadFile(tileFile)
    
    # now find times and do the best you can to bin times and effectively
    # fill in data from different tiles measured at slightly different times
    

    # make array of universal times: i.e. bins in which you will test binding
    # default time delta is the minimum time delta within a single tile
    if args.tau is None:
        time_deltas = np.hstack([np.sort(times)[1:] - np.sort(times)[:-1] for times in timeDelta.values() if len(times) > 1])
        min_time_delta = time_deltas.min()
        plotFun.plotTimeDeltaDist(time_deltas, min_time_delta)
        plt.savefig(os.path.join(figDirectory, 'time_delta_hist.pdf'))

    else:
        min_time_delta = args.tau 
    
    universalTimes = np.arange(0, np.hstack(timeDelta.values()).max()+min_time_delta,
                               min_time_delta)
    
    # find which times in universal times have variants
    
    # timeBinsDict has keys = tile and values equal to the time bins that tile occupies
    timeBinsDict = {}
    for tile, times in timeDelta.items():
        timeBinsDict[tile] = pd.Series(np.searchsorted(universalTimes, times, side='right'))
    timeBins = np.unique(np.hstack(timeBinsDict.values()))
           
    # tile Map will contain bool values for whether that tile shows up in that bin
    tileMap = pd.DataFrame(data=0, index=np.sort(timeBinsDict.keys()), columns=timeBins)
    
    # time Map will contain the time stamp of that tile, orgnaized by which bin it is in
    timeMap = pd.DataFrame(index=np.sort(timeBinsDict.keys()), columns=timeBins)
    
    # timeSeries is organized with rows = clusters and columns = time bins
    timeSeriesNorm = []
    
    # for times in timeDelta dict, 
    for tile, times in timeDelta.items():
        print 'adding tile %s'%tile
        # find data for a particular tile
        index = tiles == tile
        tileSeries = timeSeries.loc[index, timeBinsDict[tile].index]
        
        # rename columns with new time binds
        tileSeries.columns = timeBinsDict[tile]
                
        # if there are any duplicates, take mean and save to list
        timeSeriesNorm.append(tileSeries.groupby(axis=1, level=0).mean())
        
        # fill in maps
        counts = timeBinsDict[tile].value_counts()
        tileMap.loc[tile, counts.index] = counts # set to True
        timeMap.loc[tile, timeBinsDict[tile]] = pd.Series(timeDelta[tile], index=timeBinsDict[tile]).groupby(level=0).mean()
    
    timeSeriesNorm = pd.concat(timeSeriesNorm)
    finalTimes = timeMap.mean()
    
    # save times and 
    timeSeriesNorm.to_pickle(outFile + '.CPtimeseries.pkl')
    np.savetxt(timesFilename, finalTimes.values)
    
    # plot the number of tiles and times
    plotFun.plotNumberOfTilesFitRates(tileMap, finalTimes)
    plt.savefig(os.path.join(figDirectory, 'number_tiles_fit.pdf'))
    
    plotFun.plotTimesScatter(timeMap, finalTimes)
    plt.savefig(os.path.join(figDirectory,  'times_fit.pdf'))
        
    # plot timeDelta original
    plotFun.plotTimesOriginal(timeDelta)
    plt.savefig(os.path.join(figDirectory, 'times_per_tile_original.pdf'))
