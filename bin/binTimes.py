"""
Sarah Denny
Stanford University

Using the compressed barcode file (from Lauren) and the list of designed variants,
figure out how many barcodes per designed library variant (d.l.v.), and plot histograms.
"""

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


### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bin time series by time')

parser.add_argument('-cs', '--cpsignal', metavar="CPsignal.pkl",
                   help='CPsignal file containining the binding curve information'
                   ' and tile information.')
parser.add_argument('-td', '--timeDeltaDict', metavar="timeDict.pkl",
                   help='file containining the timing information per tile')
parser.add_argument('-o', '--out_file', 
                   help='output basename')


##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()

    outFile                = args.out_file
    pickleCPsignalFilename = args.cpsignal
    timeDelta = IMlibs.loadTimeDeltaDict(args.timeDeltaDict)
    
    # find out file
    if outFile is None:
        outFile = os.path.splitext(
            pickleCPsignalFilename[:pickleCPsignalFilename.find('.pkl')])[0]
    
    # process output filenames
    timesFilename = outFile + '.times.txt'
    bindingCurveFilename = outFile + '.bindingCurve.pkl'

    # laod timing info and fluorescence info

    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
            pickleCPsignalFilename)
    
    allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # now find times and do the best you can to bin times and effectively
    # fill in data from different tiles measured at slightly different times
    
    # read tiles
    tiles = pd.read_pickle(pickleCPsignalFilename).loc[:, 'tile']
    
    # make array of universal times: i.e. bins in which you will test binding
    # default time delta is the minimum time delta within a single tile
    min_time_delta = np.min([(np.array(times[1:]) - np.array(times[:-1])).min()
                             for times in timeDelta.values()])
    
    universalTimes = np.arange(0, np.hstack(timeDelta.values()).max()+min_time_delta,
                               min_time_delta)
    
    # find which times in universal times have variants
    cols = []
    for tile, times in timeDelta.items():
        cols.append(np.searchsorted(universalTimes, times))
    finalCols = np.unique(np.hstack(cols))
        
    # remake binding series file
    timeSeriesNorm = pd.DataFrame(index=bindingSeriesNorm.index,
                                  columns=finalCols)
    for tile, times in timeDelta.items():
        print 'adding tile %s'%tile
        index = tiles == tile
        old_cols = np.arange(len(times))
        cols = np.searchsorted(universalTimes, times)
        timeSeriesNorm.loc[index, cols] = bindingSeriesNorm.loc[index,old_cols].values
    
    # save times and 
    finalTimes = pd.Series(universalTimes[finalCols], index=finalCols)
    timeSeriesNorm.to_pickle(bindingCurveFilename)
    np.savetxt(timesFilename, finalTimes.values)
    
    