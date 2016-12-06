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
import ipdb
from fittinglibs import (plotting, fitting, fileio, seqfun, distribution, objfunctions)

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
group.add_argument('-n','--numCores', type=int, default=20, metavar="N",
                    help='maximum number of cores to use. default=20')
group.add_argument('--subset',action="store_true", default=False,
                    help='if flagged, will only do a subset of the data for test purposes')

group = parser.add_argument_group('arguments about fitting function')
group.add_argument('--func', default = 'binding_curve',
                   help='fitting function. default is "binding_curve", referring to module names in fittinglibs.objfunctions.')
group.add_argument('--params', nargs='+',
                   help='list of param names in fitting function.')
group.add_argument('--params_init', nargs='+', type=float,
                   help='initial values for params in params init. Has presets for most param types.')
group.add_argument('--params_ub', nargs='+', type=float,
                   help='upper bounds for params in params init. Has presets for most param types.')
group.add_argument('--params_lb', nargs='+', type=float,
                   help='lower bounds for params in params init. Has presets for most param types.')
group.add_argument('--params_vary', nargs='+', type=int,
                   help='Whether to vary params in params init. Has presets for most param types.')
#group.add_argument('--ft_only',action="store_true", default=False,
#                    help='if flagged, do not fit, but save the fit parameters')


def splitAndFit(bindingSeries, concentrations, fitParameters, numCores,
                index=None, change_params=None, func=None):
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
            (delayed(fitting.fitSetClusters)(concentrations, subBindingSeries, fitParameters, func=func,
                                             print_bool=print_bool, change_params=change_params)
             for subBindingSeries, print_bool in itertools.izip(bindingSeriesSplit, printBools)))

    return pd.concat(fits)

# define functions
def bindingSeriesByCluster(concentrations, bindingSeries, fitParameters, numCores, subset=False,
                           func=None, index_all=None):
    """ Initialize fitting. """

    # sort by fluorescence in null_column to try to get groups of equal
    # distributions of binders/nonbinders
    if not index_all:
        fluorescence = bindingSeries.iloc[:, -1].copy()
        fluorescence.sort()
        index_all = bindingSeries.loc[fluorescence.index].dropna(axis=0, thresh=4).index
    
        if subset:
            num = 5E3
            index_all = index_all[np.linspace(0, len(index_all)-1, num).astype(int)]
    
    
    fitResults = pd.DataFrame(index=bindingSeries.index,
                              columns=fitting.fitSingleCurve(concentrations, None, fitParameters, func, do_not_fit=True).index)
    fitResults.loc[index_all] = splitAndFit(bindingSeries, concentrations, fitParameters, numCores, func=func,
                                            index=index_all, change_params=True)


    return fitResults
    
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

def getFmin(bindingSeries):
    """Given the initial binding fluorescence, estimate fmin."""
    # change fmin initial
    initial_fluorescence = bindingSeries.iloc[:, 0]
    all_indices = initial_fluorescence.dropna().index.tolist()
    index_sub = np.random.choice(all_indices, size=min(1000, len(all_indices)), replace=False)
    print "Fitting fmin initial..."
    fmin_fixed = distribution.fitGammaDistribution(
        initial_fluorescence.loc[index_sub], plot=True, set_offset=0).loc['mean']
    return fmin_fixed

if __name__=="__main__":    
    args = parser.parse_args()
    
    bindingSeriesFilename = args.binding_series
    outFile  = args.out_file
    numCores = args.numCores
    concentrations = np.loadtxt(args.concentrations)

    # define out file
    if outFile is None:
        outFile = fileio.stripExtension(bindingSeriesFilename)
    if args.subset:
        outFile = '%s.subset'%outFile
    print "Loading binding series..."
    bindingSeries = fileio.loadFile(bindingSeriesFilename)  
    
    # parse input
    func = getattr(objfunctions, args.func)
    fitParameters = objfunctions.processFuncInputs(args.func, concentrations, args.params, args.params_init, args.params_lb, args.params_ub, args.params_vary)
    
    # change fitParameters fmin to a fixed value.
    fitParameters.loc['initial', 'fmin'] = getFmin(bindingSeries)

    fitResults = bindingSeriesByCluster(concentrations, bindingSeries, fitParameters,
                           numCores=numCores, subset=args.subset, func=func)
    
    fitResults.to_pickle(outFile+'.CPfitted.pkl')
    fitParameters.to_csv(outFile+'.fitParameters', sep='\t')
    
    checkFitResults(fitResults)