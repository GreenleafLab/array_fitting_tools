#!/usr/bin/env python
""" Fit all single clusters with minimal constraints.

Extracts data from merged, CPsignal file.
Normalizes by all cluster signal if this was given.
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
import IMlibs
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF               
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import lmfit
import fitFun
from fitFun import fittingParameters
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='fit single clusters to binding curve')

group = parser.add_argument_group('required arguments for fitting single clusters')
group.add_argument('-cs', '--cpsignal', metavar="CPsignal.pkl",
                    help='reduced CPsignal file. Use this if binding curves file not given')
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
                index=None, change_params=None):
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
                   change_params=None):
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
                                  change_params=change_params))

    return pd.concat(singles)

def perCluster(concentrations, fluorescence, fitParameters, plot=None, change_params=None):
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
        
        index = np.isfinite(fluorescence)
        single = fitFun.fitSingleCurve(concentrations[index.values],
                                                       fluorescence.loc[index],
                                                       fitParameters)
    except:
        print 'Error with %s'%fluorescence.name
        single = fitFun.fitSingleCurve(concentrations,
                                                       fluorescence,
                                                       fitParameters,
                                                       do_not_fit=True)
    if plot:
        fitFun.plotFitCurve(concentrations, fluorescence, single, fitParameters) 
              
    return pd.DataFrame(columns=[fluorescence.name],
                        data=single).transpose()


# define functions
def bindingSeriesByCluster(concentrations, bindingSeries, 
                           numCores=None,  subset=None, fitParameters=None):
    """ Initialize fitting. """
    if subset is None:
        subset = False

    # find initial parameters
    if fitParameters is None:
        fitParameters = getInitialFitParameters(concentrations)

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


    return fitResults
    



if __name__=="__main__":    
    args = parser.parse_args()
    
    pickleCPsignalFilename = args.cpsignal
    fitParametersFilename  = args.fit_parameters
    outFile  = args.out_file
    numCores = args.numCores
    concentrations = np.loadtxt(args.concentrations)
    
    if fitParametersFilename is not None:
        fitParameters = pd.read_table(fitParametersFilename, index_col=0)
    else:
        fitParameters = None
        
    #  check proper inputs
    if outFile is None:
        outFile = os.path.splitext(
                    bindingCurveFilename[:bindingCurveFilename.find('.pkl')])[0]
    
    print '\tLoading binding series and all RNA signal:'; sys.stdout.flush()
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
        pickleCPsignalFilename, concentrations=concentrations)
    
    # make normalized binding series
    allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # save
    bindingSeriesNorm.to_pickle(bindingCurveFilename)
    
    fitResults = bindingSeriesByCluster(concentrations, bindingSeriesNorm, 
                           numCores=numCores, subset=args.subset,
                           fitParameters=fitParameters)
    
    fitResults.to_pickle(outFile+'.CPfitted.pkl')
    
