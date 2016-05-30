#!/usr/bin/env python
""" Extracts data from merged, CPseries file for red and green and normalizes.

Sarah Denny

"""

import os
import numpy as np
import pandas as pd
import argparse
import sys
import itertools
import scipy.stats as st
import matplotlib.pyplot as plt
import fileFun
import plotFun
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='fit single clusters to binding curve')

group = parser.add_argument_group('required arguments for fitting single clusters')
group.add_argument('-b', '--binding_series', metavar="CPseries.pkl",
                    help='reduced CPseries file of binding signal.')
group.add_argument('-a', '--all_cluster', metavar='CPseries.pkl',
                   help='reduced CPseries file of any RNA signal.')

group = parser.add_argument_group('optional arguments')
group.add_argument('--no_bounds', action="store_true",
                   help='By default, all cluster signal is bounded to prevent '
                   'dividing by zero in cases where signal is low. Flag to prevent this.')
group.add_argument('--bounds', nargs=2, metavar='N N',
                   help='use these lower and upper bounds if provided. ',
                   type=float)
group.add_argument('-out', '--out_file', 
                   help='output filename. includes extension not pkl')


def boundFluorescence(signal, plot=False, bounds=None):
    # take i.e. all cluster signal and bound it     
    signal = signal.copy()
    
    # check if at least one element of signal is not nan
    if np.isfinite(signal).sum() > 0:
        if bounds is None:
            lowerbound = np.percentile(signal.dropna(), 1)
            upperbound = signal.median() + 5*signal.std()
            print 'Found bounds: %4.3f, %4.3f'%(lowerbound, upperbound)
        else:
            lowerbound = bounds[0]
            upperbound = bounds[1]
            print 'Using given bounds: %4.3f, %4.3f'%(lowerbound, upperbound)
        
        if plot:
            plotFun.plotBoundFluorescence(signal, [lowerbound, upperbound])
        signal.loc[signal < lowerbound] = lowerbound
        signal.loc[signal > upperbound] = upperbound
    
    else:
        #if they are all nan, set to 1 for division
        signal.loc[:] = 1
    return signal


if __name__=="__main__":    
    args = parser.parse_args()
    
    # find out file and fig directory
    if args.out_file is None:
        args.out_file = fileFun.stripExtension(args.binding_series) + '_normalized.CPseries'
    
    figDirectory = os.path.join(os.path.dirname(args.out_file), fileFun.returnFigDirectory())
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    
    # use only first columns of allCluster signal file
    print "Loading all RNA signal..."
    allClusterSignal = fileFun.loadFile(args.all_cluster).iloc[:, 0]
    
    # laod whole binding Series
    print "Loading series..."
    bindingSeries = fileFun.loadFile(args.binding_series)

    # make normalized binding series
    print "Normalizing..."
    if not args.no_bounds:
        allClusterSignal = boundFluorescence(allClusterSignal, plot=True, bounds=args.bounds)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # save
    print "Saving..."
    bindingSeriesNorm.to_pickle(args.out_file + '.pkl')
    plt.savefig(os.path.join(figDirectory, 'bound_all_cluster_signal.pdf'))
    