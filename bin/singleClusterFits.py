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
import seqfun
import itertools
import scipy.stats as st
from joblib import Parallel, delayed
import lmfit
import fileFun
import fitFun
from fitFun import fittingParameters
import findFmaxDist
import plotFun
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
group.add_argument('-n','--numCores', type=int, default=20, metavar="N",
                    help='maximum number of cores to use. default=20')
group.add_argument('--subset',action="store_true", default=False,
                    help='if flagged, will only do a subset of the data for test purposes')


def getInitialFitParameters(concentrations):
    """ Return fitParameters object with minimal constraints.
    
    Input: concentrations
    Uses concencetration to provide constraints on dG
    """
    parameters = fittingParameters(concentrations=concentrations)
    
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    
    # find fmin
    param = 'fmin'
    fitParameters.loc[:, param] = [0, 0, np.inf]

    # find fmax
    param = 'fmax'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    
    # find dG
    fitParameters.loc[:, 'dG'] = [parameters.find_dG_from_Kd(
        parameters.find_Kd_from_frac_bound_concentration(frac_bound, concentration))
             for frac_bound, concentration in itertools.izip(
                                    [0.99, 0.5, 0.01],
                                    [concentrations[0], concentrations[-1], concentrations[-1]])]
 
    return fitParameters


def splitAndFit(bindingSeries, concentrations, fitParameters, numCores,
                index=None, change_params=None, func=None):
    """ Given a table of binding curves, split and parallelize fit. """
    if index is None:
        index = bindingSeries.index
    
    # split into parts
    print 'Splitting clusters into %d groups:'%numCores
    # assume that list is sorted somehow
    indicesSplit = [index[i::numCores] for i in range(numCores)]
    bindingSeriesSplit = [bindingSeries.loc[indices] for indices in indicesSplit]
    printBools = [True] + [False]*(numCores-1)

    print 'Fitting binding curves:'
    fits = (Parallel(n_jobs=numCores, verbose=10)
            (delayed(fitSetClusters)(concentrations, subBindingSeries,
                                     fitParameters, print_bool, change_params)
             for subBindingSeries, print_bool in itertools.izip(bindingSeriesSplit, printBools)))

    return pd.concat(fits)

def fitSetClusters(concentrations, subBindingSeries, fitParameters, print_bool=None,
                   change_params=None, func=None, kwargs=None):
    """ Fit a set of binding curves. """
    if print_bool is None: print_bool = True

    #print print_bool
    singles = []
    for i, idx in enumerate(subBindingSeries.index):
        if print_bool:
            num_steps = max(min(100, (int(len(subBindingSeries)/100.))), 1)
            if (i+1)%num_steps == 0:
                print ('working on %d out of %d iterations (%d%%)'
                       %(i+1, len(subBindingSeries.index), 100*(i+1)/
                         float(len(subBindingSeries.index))))
                sys.stdout.flush()
        fluorescence = subBindingSeries.loc[idx]
        singles.append(perCluster(concentrations, fluorescence, fitParameters,
                                  change_params=change_params, func=func, kwargs=kwargs))

    return pd.concat(singles)

def perCluster(concentrations, fluorescence, fitParameters, plot=None, change_params=None, func=None,
               fittype=None, kwargs=None, verbose=False):
    """ Fit a single binding curve. """
    if plot is None:
        plot = False
    if change_params is None:
        change_params = True
    try:
        if change_params:
            a, b = np.percentile(fluorescence.dropna(), [0, 100])
            fitParameters = fitParameters.copy()
            fitParameters.loc['initial', 'fmax'] = b
        #index = np.isfinite(fluorescence)
        fluorescence = fluorescence[:len(concentrations)]
        single = fitFun.fitSingleCurve(concentrations,
                                                       fluorescence,
                                                       fitParameters, func=func, kwargs=kwargs)
    except IndexError as e:
        if verbose: print e
        print 'Error with %s'%fluorescence.name
        single = fitFun.fitSingleCurve(concentrations,
                                                       fluorescence,
                                                       fitParameters,
                                                       do_not_fit=True, func=func)
    if plot:
        if fittype == 'off':
            plotFun.plotFitCurve(concentrations, fluorescence, single, fitParameters,
                                 log_axis=False, func=fitFun.objectiveFunctionOffRates, fittype='off', kwargs=kwargs)
        else:
            plotFun.plotFitCurve(concentrations, fluorescence, single, fitParameters, func=func, kwargs=kwargs) 
              
    return pd.DataFrame(columns=[fluorescence.name],
                        data=single).transpose()


# define functions
def bindingSeriesByCluster(concentrations, bindingSeries, 
                           numCores=None,  subset=None, fitParameters=None, func=None):
    """ Initialize fitting. """
    if subset is None:
        subset = False

    # find initial parameters
    if fitParameters is None:
        fitParameters = getInitialFitParameters(concentrations)
        
        # change fmin initial
        initial_fluorescence = bindingSeries.sort([0]).dropna().iloc[::100, 0]
        print "Fitting fmin initial..."
        fitParameters.loc['initial', 'fmin'] = findFmaxDist.fitGammaDistribution(
            initial_fluorescence, plot=True, set_offset=0).loc['mean']

    # sort by fluorescence in null_column to try to get groups of equal
    # distributions of binders/nonbinders
    fluorescence = bindingSeries.iloc[:, -1].copy()
    fluorescence.sort()
    index_all = bindingSeries.loc[fluorescence.index].dropna(axis=0, thresh=4).index

    if subset:
        num = 5E3
        index_all = index_all[np.linspace(0, len(index_all)-1, num).astype(int)]
    
    fitResults = pd.DataFrame(index=bindingSeries.index,
                              columns=fitFun.fitSingleCurve(concentrations,
                                                            None, fitParameters,
                                                            do_not_fit=True).index)
    fitResults.loc[index_all] = splitAndFit(bindingSeries, concentrations,
                                        fitParameters, numCores, index=index_all,
                                        change_params=True)


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
        outFile = fileFun.stripExtension(bindingSeriesFilename)
    print "Loading binding series..."
    bindingSeries = fileFun.loadFile(bindingSeriesFilename)
        
    fitResults, fitParameters = bindingSeriesByCluster(concentrations, bindingSeries, 
                           numCores=numCores, subset=args.subset,
                           fitParameters=fitParameters)
    
    fitResults.to_pickle(outFile+'.CPfitted.pkl')
    fitParameters.to_csv(outFile+'.fitParameters', sep='\t')
    
    checkFitResults(fitResults)