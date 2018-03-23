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
import logging
from fittinglibs import (plotting, fileio, processing)

##### PARSE ARGUMENTS #####
parser = argparse.ArgumentParser(description='normalize fluorescence series in green channel '
                                 '(associated with binding signal) by the signal in the red channel '
                                 '(associated with total transcribed RNA)')
processing.add_common_args(parser.add_argument_group('common arguments'))

group = parser.add_argument_group('optional arguments for normalize series')
group.add_argument('--no_bounds', action="store_true",
                   help='By default, all cluster signal is bounded to prevent '
                   'dividing by zero in cases where signal is low. Flag to prevent this.')
group.add_argument('--bounds', nargs=2, metavar='N N',
                   help='use these lower and upper bounds if provided. ',
                   type=float)


##### FUNCITONS #####
def boundFluorescence(signal, plot=False, bounds=None):
    # take i.e. all cluster signal and bound it     
    signal = signal.copy()
    
    # check if at least one element of signal is not nan
    if np.isfinite(signal).sum() > 0:
        if bounds is None:
            lowerbound = np.percentile(signal.dropna(), 1)
            upperbound = signal.median() + 5*signal.std()
            logging.info('Found bounds: %4.3f, %4.3f'%(lowerbound, upperbound))
        else:
            lowerbound = bounds[0]
            upperbound = bounds[1]
            logging.info('Using given bounds: %4.3f, %4.3f'%(lowerbound, upperbound))
        
        if plot:
            plotting.plotBoundFluorescence(signal, [lowerbound, upperbound])
        signal.loc[signal < lowerbound] = lowerbound
        signal.loc[signal > upperbound] = upperbound
    
    else:
        #if they are all nan, set to 1 for division
        signal.loc[:] = 1
    return signal

##### MAIN #####
if __name__=="__main__":


    args = parser.parse_args()
    processing.update_logger(logging, args.log)
    
    ##### DEFINE OUTPUT FILE AND DIRECTORY #####
    # find out file and fig directory
    if args.out_file is None:
        args.out_file = fileio.stripExtension(args.binding_series) + '_normalized.CPseries.gz'
    
    figDirectory = os.path.join(os.path.dirname(args.out_file), fileio.returnFigDirectory())
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    
    # use only first columns of allCluster signal file
    logging.info("Loading all RNA signal")
    allClusterSignal = fileio.loadFile(args.ref_fluor_series).iloc[:, 0]
    
    # laod whole binding Series
    logging.info("Loading series...")
    bindingSeries = fileio.loadFile(args.binding_series)

    # make normalized binding series
    logging.info("Normalizing...")
    if not args.no_bounds:
        allClusterSignal = boundFluorescence(allClusterSignal, plot=True, bounds=args.bounds)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # save
    logging.info( "Saving...")
    bindingSeriesNorm.to_csv(args.out_file, sep='\t', compression='gzip')
    plotting.savefig(os.path.join(figDirectory, 'bound_all_cluster_signal.pdf'))
    