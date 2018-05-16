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
from joblib import Parallel, delayed
import lmfit
import itertools
import ipdb
from fittinglibs import fitting, plotting, fileio, distribution, variables, initfits, processing


### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
processing.add_common_args(parser.add_argument_group('common arguments'),
                           required_x=True, required_v=True, required_a=True)
parser.add_argument('-m', '--fmax_dist_file', required=True, metavar="fmaxdist.p",
                    help='text file giving the associated concentrations')

group = parser.add_argument_group('additional option arguments')
group.add_argument('--n_samples', default=100, type=int, metavar="N",
                   help='number of times to bootstrap samples')
group.add_argument('--enforce_fmax', type=int,  
                   help='set to 0 or 1 if you want to always enforce fmax (1) or'
                   'never enforce it (0). Otherwise the program will decide. ')
group.add_argument('--subset',action="store_true", default=False,
                    help='if flagged, will only do a subset of the data for test purposes')
group.add_argument('--variants',
                    help='fit only variants listed in this text file')
group.add_argument('--no_weights',action="store_true", default=False,
                    help="if flagged, won't weight the fit by error bars on median fluorescence")

group = parser.add_argument_group('arguments about fitting function')
group.add_argument('--func', default = 'binding_curve',
                   help='fitting function. default is "binding_curve", referring to module names in fittinglibs.objfunctions.')
group.add_argument('--params_name', nargs='+', help='name of param(s) to edit.')
group.add_argument('--params_init', nargs='+', type=float, help='new initial val(s) of param(s) to edit.')
group.add_argument('--params_vary', nargs='+', type=int, help='whether to vary val(s) of param(s) to edit.')
group.add_argument('--params_lb', nargs='+', type=float, help='new lowerbound val(s) of param(s) to edit.')
group.add_argument('--params_ub', nargs='+', type=float, help='new upperbound val(s) of param(s) to edit.')
##### functions #####
def makeGroupDict(bindingSeries, annotatedClusters):
    """ Return the fluorescence values, split by variant. """
    
    # load binding series information with variant numbers
    cols = bindingSeries.columns.tolist()
    fluorescenceMat = pd.concat([annotatedClusters, bindingSeries.astype(float)], axis=1)
    return fluorescenceMat.dropna(subset=['variant_number']).dropna(subset=cols, thresh=4).set_index('variant_number', append=True).swaplevel(0, 1).sort_index()
"""
    # fit all labeled variants
    fluorescenceMat.dropna(subset=['variant_number'], inplace=True)

    # fit only clusters that are not all NaN
    fluorescenceMat.dropna(subset=bindingSeries.columns.tolist(), how='all',inplace=True)
    
    print "\tSplitting..."
    bindingSeriesDict = {}
    for name, group in fluorescenceMat.groupby('variant_number'):
        bindingSeriesDict[name] = group.loc[:, bindingSeries.columns]
    return bindingSeriesDict
"""

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
    bindingCurveFilename  = args.binding_series
    fmaxDistFile = args.fmax_dist_file
    n_samples = args.n_samples
    numCores = args.numCores
    outFile  = args.out_file
    
    enforce_fmax = args.enforce_fmax
    concentrations = np.loadtxt(args.xvalues)
    weighted_fit = not args.no_weights
    parameters = variables.fittingParameters()

    # find out file
    if outFile is None:
        basename = fileio.stripExtension(bindingCurveFilename)
        if args.subset:
            outFile = basename + '_subset.CPvariant.gz'
        else:
            outFile = basename + '.CPvariant.gz'

    # load data
    print "Loading binding fluorescence..."
    fmaxDistObject = fileio.loadFile(fmaxDistFile)
    initialPoints = fileio.loadFile(variantFile)
    bindingSeries = fileio.loadFile(bindingCurveFilename)
    annotatedClusters = fileio.loadFile(annotatedClusterFile)

    # subset
    if args.variants is None:
        variants = initialPoints.index.tolist()
        if args.subset:
            variants = variants[-100:]
    else:
        variants = list(np.loadtxt(args.variants))

    # process bindign series into per-variant dict
    bindingSeriesDict = makeGroupDict(bindingSeries, annotatedClusters)
    medianBindingSeries = bindingSeriesDict.groupby(level=0).median()
    
    # find initial points
    #initialPoints = findInitialPoints(variantTable, bindingSeriesDict.keys())
    
    # find fmin based on initial fits and some thresholds on dG
    maxFracBound = 0.01 # at most 1% bound in first concentration
    mindG = parameters.find_dG_from_frac_bound(maxFracBound, np.min(concentrations))
    idxMin = pd.Series(concentrations, index=bindingSeries.columns).idxmin()
    # the fixed fmin is the median fluorescence at the lowest concentration for those dGs that are less than 1% bound at the lowest concentration
    fminFixed = medianBindingSeries.loc[initialPoints.dG > mindG, idxMin].median()
    fitParams = initfits.FitParams(args.func, concentrations)
    fitParams.update_init_params(fmin={'initial':fminFixed, 'vary':False},
                                 fmax={'initial':fmaxDistObject.getDist(1).mean()})
    
    # process input args
    args = initfits.process_new_params(args)
    for param_name, param_init, param_lb, param_ub, param_vary in zip(args.params_name, args.params_init, args.params_lb, args.params_ub, args.params_vary):
        if param_name:
            fitParams.update_init_params(**{param_name:{'initial':param_init, 'lowerbound':param_lb, 'upperbound':param_ub, 'vary':bool(param_vary)}})

        
    # initiate fits

    # actually split by cores
    print '\tSplitting data into %d pieces...'%numCores
    variantsSplit = [list(vec) for vec in np.array_split(variants, numCores)]
    variantParamsSplit = [initfits.MoreFitParams(fitParams, initial_points=initialPoints.loc[variantSet],
                                                 binding_series_dict=bindingSeriesDict.loc[variantSet],
                                                 fmax_dist_obj=fmaxDistObject)
                          for variantSet in variantsSplit]
    printBools = [True] + [False]*(numCores-1)

    print '\tMultiprocessing bootstrapping...'

    # parallelize fitting
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(fitting.fitSetVariants)(variantParams,
                                        n_samples=n_samples,
                                        enforce_fmax=enforce_fmax,
                                        weighted_fit=weighted_fit,
                                        print_bool=printbool)
                 for variantParams, printbool  in zip(variantParamsSplit, printBools)))      
     
    results = pd.concat(results).sort_index()

        
    # fit
    #variantFinal = parseResults(variantTable, results)

    # save
    results.to_csv(outFile , sep='\t', index=True, compression='gzip')

    variantParams = initfits.MoreFitParams(fitParams, initial_points=initialPoints, binding_series_dict=bindingSeriesDict, fmax_dist_obj=fmaxDistObject)
    
    variantParams.results_all = results
    fileio.saveFile(fileio.stripExtension(outFile) + '.variantParams.p', variantParams)