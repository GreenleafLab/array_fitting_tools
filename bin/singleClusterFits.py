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
from fittinglibs import (plotting, fitting, fileio, seqfun, distribution, objfunctions, initfits)

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
group.add_argument('--subset_num', default=5000,
                    help='do at most this many single clusters when the subset flag is true. default=5000')
group = parser.add_argument_group('arguments about fitting function')
group.add_argument('--func', default = 'binding_curve',
                   help='fitting function. default is "binding_curve", referring to module names in fittinglibs.objfunctions.')
group.add_argument('--params_name', nargs='+', help='name of param(s) to edit.')
group.add_argument('--params_init', nargs='+', type=float, help='new initial val(s) of param(s) to edit.')
group.add_argument('--params_vary', nargs='+', type=int, help='whether to vary val(s) of param(s) to edit.')
group.add_argument('--params_lb', nargs='+', type=float, help='new lowerbound val(s) of param(s) to edit.')
group.add_argument('--params_ub', nargs='+', type=float, help='new upperbound val(s) of param(s) to edit.')
#group.add_argument('--ft_only',action="store_true", default=False,
#                    help='if flagged, do not fit, but save the fit parameters')


def splitAndFit(fitParams, bindingSeries, numCores, index=None):
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
    
    # fit each set of split clusters
    print 'Fitting binding curves:'
    fits = (Parallel(n_jobs=numCores, verbose=10)
            (delayed(fitting.fitSetClusters)(fitParams, subBindingSeries, print_bool=print_bool)
             for subBindingSeries, print_bool in itertools.izip(bindingSeriesSplit, printBools)))

    return pd.concat(fits)

# define functions

    
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
    idx_min_concentration = pd.Series(concentrations, index=bindingSeries.columns).idxmin()
    idx_max_concentration = pd.Series(concentrations, index=bindingSeries.columns).idxmax()
    
    # parse input
    fitParams = initfits.FitParams(args.func, concentrations, before_fit_ops=[('fmax', 'initial', np.max)])
    fitParams.update_init_params(fmin={'initial':bindingSeries.loc[:, idx_min_concentration].median()})

    # process input args
    args = initfits.process_new_params(args)
    for param_name, param_init, param_lb, param_ub, param_vary in zip(args.params_name, args.params_init, args.params_lb, args.params_ub, args.params_vary):
        if param_name:
            fitParams.update_init_params(**{param_name:{'initial':param_init, 'lowerbound':param_lb, 'upperbound':param_ub, 'vary':bool(param_vary)}})

    # sort by fluorescence in null_column to try to get groups of equal
    # distributions of binders/nonbinders
    index_all = bindingSeries.sort_values(idx_max_concentration).dropna(axis=0, thresh=4).index.tolist()
    if args.subset:
        if len(index_all) > args.subset_num:
            index_all = [index_all[i] for i in np.linspace(0, len(index_all)-1, args.subset_num).astype(int)]

    # split into nCore number of clusters and fit each set    
    fitResults = splitAndFit(fitParams, bindingSeries, args.numCores, index=index_all)
    
    # save
    fitResults.to_pickle(outFile+'.CPfitted.pkl')
    fileio.saveFile(outFile+'.fitParameters.p', fitParams)
    
    checkFitResults(fitResults)