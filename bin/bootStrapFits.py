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
import ipdb
from fittinglibs import fitting, plotting, fileio, distribution, objfunctions


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
group.add_argument('--variants',
                    help='fit only variants listed in this text file')
group.add_argument('--no_weights',action="store_true", default=False,
                    help="if flagged, won't weight the fit by error bars on median fluorescence")

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

def findInitialPoints(variant_table, variants):
    """Return initial points from variant table."""
    # make sure initial points have all of keys that table does
    initialPoints = distribution.findInitialPoints(variant_table)
    missing_variants = (np.array(variants)
                        [np.logical_not(np.in1d(np.array(variants).astype(str), np.array(initialPoints.index.tolist()).astype(str)))])
    initialPoints = pd.concat([initialPoints, pd.DataFrame(index=missing_variants,
                                                           columns=initialPoints.columns)])
    return initialPoints

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
    """
    col_order = variant_table.columns.tolist()

    to return all columns:
    """
    col_order = variant_table.columns.tolist() + [col for col in results if col not in variant_table]
    
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
    weighted_fit = not args.no_weights
    
    # find out file
    if outFile is None:
        outFile = fileio.stripExtension(bindingCurveFilename)

    # load data
    fmaxDistObject = fileio.loadFile(fmaxDistFile)
    variant_table = fileio.loadFile(variantFile)
    fluorescenceMat, fluorescenceMatSplit = loadGroupDict(bindingCurveFilename, annotatedClusterFile)

    # find table of initial points
    parameters = fitting.fittingParameters(concentrations)
    initialPoints = findInitialPoints(variant_table, fluorescenceMatSplit.keys())
    fmin_fixed = distribution.returnFminFromFluorescence(initialPoints, fluorescenceMat, parameters.mindG)
        
    # subset
    if args.variants is None:
        variants = variant_table.index.tolist()
    else:
        variants = np.loadtxt(args.variants)
    if args.subset:
        variants = variants[-100:]
        outFile = outFile + '_subset'
    
    # parse input
    func = getattr(objfunctions, args.func)
    fitParameters = objfunctions.processFuncInputs(args.func, concentrations, args.params, args.params_init, args.params_lb, args.params_ub, args.params_vary)
    fitParameters.loc['vary'] = True
    fitParameters.loc['vary', 'fmin'] = False
    fitParameters.loc['initial', 'fmin'] = fmin_fixed
    fitParameters.loc['initial', 'fmax'] = fmaxDistObject.getDist(1).stats(moments='m')
    fitParameters.to_csv(outFile + '.fitParameters', sep='\t', index=True)

    print '\tMultiprocessing bootstrapping...'
    # parallelize fitting
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(fitting.perVariant)(concentrations,
                                        fluorescenceMatSplit[variant],
                                        fitParameters,
                                        fmaxDistObject,
                                        func,
                                        initial_points=initialPoints.loc[variant],
                                        n_samples=n_samples,
                                        enforce_fmax=enforce_fmax,
                                        weighted_fit=weighted_fit)
                 for variant in variants if variant in fluorescenceMatSplit.keys()))      
     
    results = pd.concat(results, axis=1).transpose()
    results.index = [variant for variant in variants if variant in fluorescenceMatSplit.keys()]
        
    # fit
    variant_final = parseResults(variant_table, results)

    # save
    variant_final.to_csv(outFile + '.CPvariant', sep='\t', index=True)
    
    
