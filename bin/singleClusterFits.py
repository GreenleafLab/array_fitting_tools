#!/usr/bin/env python
""" Fit all single clusters with minimal constraints.

Fits all single clusters.

Input:
CPsignal file
concentrations file

Output:
normalized binding series file
fit results

Sarah Denny

"""

import os
import numpy as np
import pandas as pd
import argparse
import sys
import itertools
import scipy.stats as st
from joblib import Parallel, delayed
import lmfit
from fittinglibs import (plotting, fitting, fileio, seqfun, distribution)
from fittinglibs.fitting import fittingParameters

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='fit single clusters to binding curve')

group = parser.add_argument_group('required arguments for fitting single clusters')
group.add_argument('-b', '--binding_series', metavar="CPseries.pkl",
                    help='reduced [normalized] CPseries file.')
group.add_argument('-c', '--concentrations', required=True, metavar="concentrations.txt",
                    help='text file giving the associated concentrations')

group = parser.add_argument_group('optional arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')
group.add_argument('-ft', '--fit_parameters',
                    help='fitParameters file. If file is given, use these '
                    'upperbound/lowerbounds')
group.add_argument('--slope', default=0, type=float,
                    help='if provided, use this value for the slope of a linear fit.'
                    'upperbound/lowerbounds')
group.add_argument('-n','--numCores', type=int, default=20, metavar="N",
                    help='maximum number of cores to use. default=20')
group.add_argument('--subset',action="store_true", default=False,
                    help='if flagged, will only do a subset of the data for test purposes')
#group.add_argument('--ft_only',action="store_true", default=False,
#                    help='if flagged, do not fit, but save the fit parameters')


def splitAndFit(bindingSeries, concentrations, fitParameters, numCores,
                index=None, change_params=None, func=None, kwargs=None):
    """ Given a table of binding curves, split and parallelize fit. """
    if index is None:
        index = bindingSeries.index
    
    # split into parts
    numCores = min(len(index), numCores)
    print 'Splitting clusters into %d groups:'%numCores
    # assume that list is sorted somehow
    indicesSplit = [index[i::numCores] for i in range(numCores)]
    bindingSeriesSplit = [bindingSeries.loc[indices] for indices in indicesSplit]
    printBools = [True] + [False]*(numCores-1)

    print 'Fitting binding curves:'
    fits = (Parallel(n_jobs=numCores, verbose=10)
            (delayed(fitting.fitSetClusters)(concentrations, subBindingSeries,
                                     fitParameters, print_bool=print_bool, change_params=change_params, kwargs=kwargs)
             for subBindingSeries, print_bool in itertools.izip(bindingSeriesSplit, printBools)))

    return pd.concat(fits)

# define functions
def bindingSeriesByCluster(concentrations, bindingSeries, numCores, subset=False,
                           fitParameters=None, func=None, kwargs=None, index_all=None):
    """ Initialize fitting. """
    # find initial parameters
    if fitParameters is None:
        fitParameters = fitting.getInitialFitParameters(concentrations)
        
        # change fmin initial
        all_indices = bindingSeries.dropna(subset=[bindingSeries.columns[0]]).index.tolist()
        index_sub = np.random.choice(all_indices, size=min(1000, len(all_indices)), replace=False)
        initial_fluorescence = bindingSeries.loc[index_sub].iloc[:, 0]
        print "Fitting fmin initial..."
        fitParameters.loc['initial', 'fmin'] = distribution.fitGammaDistribution(
            initial_fluorescence, plot=False, set_offset=0).loc['mean']

    # sort by fluorescence in null_column to try to get groups of equal
    # distributions of binders/nonbinders
    if not index_all:
        fluorescence = bindingSeries.iloc[:, -1].copy()
        # 20180228: .sort() has been depreciated completely now. sort_values() should be equivalent.
        #fluorescence.sort()
        fluorescence.sort_values(inplace=True)
        index_all = bindingSeries.loc[fluorescence.index].dropna(axis=0, thresh=4).index
    
        if subset:
            num = 2E3
            index_all = index_all[np.linspace(0, len(index_all)-1, num).astype(int)]
    
    
    fitResults = pd.DataFrame(index=bindingSeries.index,
                              columns=fitting.fitSingleCurve(concentrations,
                                                            None, fitParameters, func,
                                                            do_not_fit=True).index)
    fitResults.loc[index_all] = splitAndFit(bindingSeries, concentrations,
                                        fitParameters, numCores, index=index_all,
                                        change_params=True, kwargs=kwargs)


    return fitResults, fitParameters
    
def checkFitResults(fitResults):
    # did any of the stde work?
    param_names = ['fmax', 'dG', 'fmin']
    numClusters = fitResults.dropna(subset=param_names).shape[0]
    print ('%4.2f%% clusters have rsq>50%%'
           %(100*(fitResults.rsq > 0.5).sum()/float(numClusters)))
    print ('%4.2f%% clusters have stde in dG < 1'
           %(100*(fitResults.dG_stde < 1).sum()/float(numClusters)))
    print ('%4.2f%% clusters have stde in fmax < fmax'
           %(100*(fitResults.fmax_stde < fitResults.fmax).sum()/float(numClusters)))
    print ('%4.2f%% clusters have stde != 0 for at least one fit parameters'
           %(100 -100*(fitResults.loc[:, ['%s_stde'%param for param in param_names]]==0).all(axis=1).sum()/float(numClusters)))
    

if __name__=="__main__":    
    args = parser.parse_args()
    
    bindingSeriesFilename = args.binding_series
    fitParametersFilename  = args.fit_parameters
    outFile  = args.out_file
    numCores = args.numCores
    concentrations = np.loadtxt(args.concentrations)
    
    # load fit parameters if given
    if fitParametersFilename is not None:
        fitParameters = pd.read_table(fitParametersFilename, index_col=0)
    else:
        fitParameters = None
    
    # define out file
    if outFile is None:
        outFile = fileio.stripExtension(bindingSeriesFilename)
    if args.subset:
        outFile = '%s.subset'%outFile
    print "Loading binding series..."
    bindingSeries = fileio.loadFile(bindingSeriesFilename)  
    fitResults, fitParameters = bindingSeriesByCluster(concentrations, bindingSeries, 
                           numCores=numCores, subset=args.subset, 
                           fitParameters=fitParameters, kwargs={'slope':args.slope})
    
    fitResults.to_pickle(outFile+'.CPfitted.pkl')
    fitParameters.to_csv(outFile+'.fitParameters', sep='\t')
    
    checkFitResults(fitResults)