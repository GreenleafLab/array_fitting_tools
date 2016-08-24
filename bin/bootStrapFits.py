#!/usr/bin/env python
#
# will find fmax distribution given the single cluster fits on 
# good binders and good fitters.
#
# fmax distribtuion is then enforced for weak binders, allowing
# good measurements even when no reaching saturation.
# 
# Median of cluster fits are bootstrapped to obtain errors on
# fit parameters.
#
# Sarah Denny
# Updated by Anthony Ho
# Aug 2016

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import datetime
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import lmfit
import itertools
from fittinglibs import fitting, plotting, fileio, distribution


### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-v', '--variant_file', required=True, metavar=".CPvariant.pkl",
                   help='file with single cluster fits')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')
parser.add_argument('-b', '--binding_curves', required=True, metavar=".bindingSeries.pkl",
                   help='file containining the binding curve information')
parser.add_argument('-c', '--concentrations', required=True, metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
parser.add_argument('-f', '--fmax_dist_file', required=True, metavar="fmaxdist.p",
                    help='text file giving the associated concentrations')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')


group = parser.add_argument_group('additional option arguments')
group.add_argument('--n_samples', default=100, type=int, metavar="N",
                   help='number of times to bootstrap samples')
group.add_argument('--enforce_fmax', type=int,  
                   help='set to 0 or 1 if you want to always enforce fmax (1) or'
                   'never enforce it (0). Otherwise the program will decide. ')
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores')
group.add_argument('--subset',action="store_true", default=False,
                    help='if flagged, will only do a subset of the data for test purposes')
group.add_argument('--slope', default=0, type=float,
                    help='if provided, use this value for the slope of a linear fit.'
                    'upperbound/lowerbounds')
group.add_argument('--min_error',type=float, default=0,
                   help='set this value for the minimum amount of error per fluorescence point, in units of percent of fmax. default=0')
group.add_argument('--no_weights', action="store_true",
                   help='Flag if you would like to not weight the fit')
group.add_argument('--fmin_float', action="store_true",
                   help='Flag if you would like to allow the fmin to float')
group.add_argument('--fit_slope', action="store_true",
                   help='Flag if you would like to fit a linear slope')

##### functions #####
def loadGroupDict(bindingCurveFilename, annotatedClusterFile):
    """ Return the fluorescence values, split by variant. """
    
    # load binding series information with variant numbers
    print "Loading binding fluorescence..."
    fluorescenceMat = (pd.concat([pd.read_pickle(annotatedClusterFile),
                                  pd.read_pickle(bindingCurveFilename).astype(float)], axis=1).
        sort('variant_number'))

    # fit all labeled variants
    fluorescenceMat.dropna(axis=0, subset=['variant_number'], inplace=True)

    # fit only clusters that are not all NaN
    fluorescenceMat.dropna(axis=0, subset=fluorescenceMat.columns[1:], how='all',inplace=True)
    
    print "\tSplitting..."
    fluorescenceMatSplit = {}
    for name, group in fluorescenceMat.groupby('variant_number'):
        fluorescenceMatSplit[name] = group.iloc[:, 1:]
    return fluorescenceMat, fluorescenceMatSplit

def initiateFitting(variant_table, fluorescenceMat, fluorescenceMatSplit, concentrations, fmaxDistObject,
                    fmin_float=False):
    """Get fit parameters and initial points."""
    # get parameters
    parameters = fitting.fittingParameters(concentrations)

    # make sure initial points have all of keys that table does
    initialPoints = distribution.findInitialPoints(variant_table)
    missing_variants = (np.array(fluorescenceMatSplit.keys())
                        [np.logical_not(np.in1d(fluorescenceMatSplit.keys(), initialPoints.index))])
    initialPoints = pd.concat([initialPoints, pd.DataFrame(index=missing_variants,
                                                           columns=initialPoints.columns)])
    
    # find fmin
    fmin_fixed = distribution.returnFminFromFluorescence(initialPoints, fluorescenceMat, parameters.mindG)
    fitParameters = []
    fitParameters.append(fitting.getFitParam('fmax', init_val=fmaxDistObject.getDist(1).stats(moments='m'), vary=True))
    fitParameters.append(fitting.getFitParam('dG', concentrations, vary=True))
    fitParameters.append(fitting.getFitParam('fmin', init_val=fmin_fixed, vary=fmin_float))
        
    return initialPoints, pd.concat(fitParameters, axis=1)


def parseResults(variant_table, results):
    """Parse the results table into the variant table final. """

    # save final results as one dataframe
    variant_final = pd.DataFrame(
        index  =np.unique(variant_table.index.tolist() + results.index.tolist()),
        columns=np.unique(variant_table.columns.tolist() + results.columns.tolist()))
    variant_final.loc[variant_table.index, variant_table.columns] = variant_table
    variant_final.loc[results.index, results.columns] = results
    
    col_order = variant_table.columns.tolist()
    """
    to return all columns:
    col_order = variant_table.columns.tolist() + [col for col in results if col not in variant_table]
    """
    return variant_final.loc[:, col_order].astype(float)



##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()
    
    variantFile = args.variant_file
    annotatedClusterFile  = args.annotated_clusters
    bindingCurveFilename  = args.binding_curves
    fmaxDistFile = args.fmax_dist_file
    n_samples = args.n_samples
    numCores = args.numCores
    outFile  = args.out_file
    enforce_fmax = args.enforce_fmax
    concentrations = np.loadtxt(args.concentrations)
    fmin_float = args.fmin_float
    weighted_fit = not args.no_weights
    
    # find out file
    if outFile is None:
        outFile = fileio.stripExtension(bindingCurveFilename)
    
    # load data
    fmaxDistObject = fileio.loadFile(fmaxDistFile)
    variant_table = fileio.loadFile(variantFile)
    fluorescenceMat, fluorescenceMatSplit = loadGroupDict(bindingCurveFilename, annotatedClusterFile)
    func_kwargs = {'slope':args.slope, 'fit_slope':args.fit_slope}
    min_error = fmaxDistObject.getDist(1).mean()*args.min_error/100.
        
    # subset
    variants = variant_table.index.tolist()
    if args.subset:
        variants = variants[-100:]
        outFile = outFile + '_subset'

    # process before fitting
    initialPoints, fitParameters = initiateFitting(variant_table, fluorescenceMat,
                                                   fluorescenceMatSplit, concentrations,
                                                   fmaxDistObject, fmin_float=fmin_float)
    
    # add slope to fitParameters if supposed to fit it 
    if 'fit_slope' in func_kwargs.keys():
        fitParameters.loc[:, 'slope'] = fitting.getFitParam('slope', vary=func_kwargs['fit_slope'])
    
    # adjust min error
    if min_error > 0:
        outFile = outFile + '_error%.1e'%min_error

    print '\tMultiprocessing bootstrapping...'
    # parallelize fitting
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(fitting.perVariant)(concentrations,
                                        fluorescenceMatSplit[variant],
                                        fitParameters,
                                        fmaxDistObject,
                                        initial_points=initialPoints.loc[variant],
                                        n_samples=n_samples,
                                        enforce_fmax=enforce_fmax,
                                        weighted_fit=weighted_fit,
                                        min_error=min_error,
                                        func_kwargs=func_kwargs)
                 for variant in variants if variant in fluorescenceMatSplit.keys()))      
     
    results = pd.concat(results, axis=1).transpose()
    results.index = [variant for variant in variants if variant in fluorescenceMatSplit.keys()]
        
    # fit
    variant_table = parseResults(variant_table, results)

    # save
    variant_table.to_csv(outFile + '.CPvariant', sep='\t', index=True)
    
    
