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
# July 2015

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seqfun
import datetime
import IMlibs
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import lmfit
import itertools
import fitFun
import plotFun
import fileFun
import findFmaxDist
import singleClusterFits
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


##### functions #####
def getInitialFitParameters(concentrations):
    """ Return initial fit parameters from single cluster fits.
    
    Add a row 'vary' that indicates whether parameter should vary or not. """
    fitParameters = singleClusterFits.getInitialFitParameters(concentrations)
    return pd.concat([fitParameters.astype(object),
                      pd.DataFrame(True, columns=fitParameters.columns,
                                         index=['vary'], dtype=bool)])


def perVariant(concentrations, subSeries, fitParameters, fmaxDistObject, initial_points=None,
               plot=None, n_samples=None, enforce_fmax=None, func=None, fittype=None):
    """ Fit a variant to objective function by bootstrapping median fluorescence. """
    
    # default id to not plot results
    if plot is None:
        plot = False
    
    # define fitting function
    if func is None:
        func = fitFun.bindingCurveObjectiveFunction

    # change initial guess on fit parameters if given previous fit
    fitParametersPer = fitParameters.copy()
    if initial_points is not None:
        params_to_change = fitParameters.loc[:, fitParametersPer.loc['vary'].astype(bool)].columns
        fitParametersPer.loc['initial', params_to_change] = (initial_points.loc[params_to_change])
    
    # find actual distribution of fmax given number of measurements
    fmaxDist = fmaxDistObject.getDist(len(subSeries))
    
    # fit variant
    results, singles = fitFun.bootstrapCurves(concentrations, subSeries, fitParameters,
                                              func=func,
                                              enforce_fmax=enforce_fmax,
                                              fmaxDist=fmaxDist,
                                              n_samples=n_samples,
                                              verbose=plot)
    # plot
    if plot:
        plotFun.plotFitCurve(concentrations,
                                     subSeries,
                                     results,
                                     fitParameters,
                                     log_axis=True,
                                     func=func, fittype=fittype)
        # also plot initial
        if initial_points is not None:
            try:
                more_concentrations = np.logspace(0, 4)
                fit = func(fitFun.returnParamsFromResults(initial_points),
                                                           more_concentrations)
                plt.plot(more_concentrations, fit, 'k--')
            except: pass

    return results

def getMedianFirstBindingPoint(table):
    """ Return the median fluoresence in first binding point of each variant. """
    return table.groupby('variant_number').median().iloc[:, 0]

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


def findInitialPoints(variant_table):
    """ Return initial points with different column names. """
    initialPoints = variant_table.loc[:, ['fmax_init', 'dG_init', 'fmin_init', 'numTests']]
    initialPoints.columns = ['fmax', 'dG', 'fmin', 'numTests']  
    return initialPoints

def returnFminFromFluorescence(initialPoints, fluorescenceMat, cutoff):
    """ Return the estimated fixed fmin based on affinity and fluroescence. """
    # if cutoff is not given, use parameters

    initial_dG = initialPoints.loc[:, 'dG']

    firstBindingPoint = getMedianFirstBindingPoint(fluorescenceMat)
    return firstBindingPoint.loc[initial_dG.index].loc[initial_dG > cutoff].median()

def initiateFitting(variant_table, fluorescenceMat, fluorescenceMatSplit, concentrations, fmaxDistObject):
    
    # get parameters
    parameters = fitFun.fittingParameters(concentrations)

    # make sure initial points have all of keys that table does
    initialPoints = findInitialPoints(variant_table)
    missing_variants = (np.array(fluorescenceMatSplit.keys())
                        [np.logical_not(np.in1d(fluorescenceMatSplit.keys(), initialPoints.index))])
    initialPoints = pd.concat([initialPoints, pd.DataFrame(index=missing_variants,
                                                           columns=initialPoints.columns)])
    
    # find fmin
    fmin_fixed = returnFminFromFluorescence(initialPoints, fluorescenceMat, parameters.mindG)
    
    # find constraints on fmin and delta G
    fitParameters = getInitialFitParameters(concentrations)
    fitParameters.loc['vary', 'fmin'] = False
    fitParameters.loc['initial', 'fmin'] = fmin_fixed
    fitParameters.loc['initial', 'fmax'] = fmaxDistObject.getDist(1).stats(moments='m')
        

    return initialPoints, fitParameters

def fitBindingCurves(variant_table, fluorescenceMat,
                     fluorescenceMatSplit, concentrations, fmaxDistObject,
                     numCores=20, n_samples=100, variants=None,
                     enforce_fmax=None):
    
    # process before fitting
    initialPoints, fitParameters = initiateFitting(variant_table, fluorescenceMat,
                                                   fluorescenceMatSplit, concentrations,
                                                   fmaxDistObject)
    
    if variants is None:
        variants = fluorescenceMatSplit.keys()
    
    print '\tMultiprocessing bootstrapping...'
    # parallelize fitting
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(perVariant)(concentrations,
                                        fluorescenceMatSplit[variant],
                                        fitParameters,
                                        fmaxDistObject,
                                        initial_points=initialPoints.loc[variant],
                                        n_samples=n_samples,
                                        enforce_fmax=enforce_fmax)
                 for variant in variants if variant in fluorescenceMatSplit.keys()))      
     
    results = pd.concat(results, axis=1).transpose()
    results.index = [variant for variant in variants if variant in fluorescenceMatSplit.keys()]

    # save final results as one dataframe
    variant_final = pd.DataFrame(
        index  =np.unique(variant_table.index.tolist() + results.index.tolist()),
        columns=variant_table.columns)
    variant_final.loc[variant_table.index, variant_table.columns] = variant_table
    columns = results.columns[np.in1d(results.columns, variant_table.columns)]
    variant_final.loc[results.index, columns] = results.loc[:, columns]

    return variant_final.astype(float)



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
    
    # find out file
    if outFile is None:
        outFile = fileFun.stripExtension(bindingCurveFilename)

    # make fig directory    
    figDirectory = os.path.join(os.path.dirname(annotatedClusterFile),
                                'figs_%s'%str(datetime.date.today()))
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    
    # load data
    fmaxDistObject = fileFun.loadFile(fmaxDistFile)
    variant_table = fileFun.loadFile(variantFile)
    fluorescenceMat, fluorescenceMatSplit = loadGroupDict(bindingCurveFilename, annotatedClusterFile)
        
    # subset
    variants = variant_table.index.tolist()
    if args.subset:
        variants = variants[-100:]
        outFile = outFile + '_subset'
        
    # fit
    variant_table = fitBindingCurves(variant_table, fluorescenceMat,
                 fluorescenceMatSplit, concentrations, fmaxDistObject,
                 numCores=numCores, n_samples=n_samples, variants=variants,
                 enforce_fmax=enforce_fmax)

    # save
    variant_table.to_csv(outFile + '.CPvariant', sep='\t', index=True)
        
    # make plots    
    plotFun.plotFmaxInit(variant_table)
    plt.savefig(os.path.join(figDirectory, 'initial_Kd_vs_final.colored_by_fmax.pdf'))
    
    plotFun.plotErrorInBins(variant_table, xdelta=10)
    plt.savefig(os.path.join(figDirectory, 'error_in_bins.dG.pdf'))
    
    plotFun.plotNumberInBins(variant_table, xdelta=10)
    plt.savefig(os.path.join(figDirectory, 'number_in_bins.Kd.pdf'))
    
