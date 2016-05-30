#!/usr/bin/env python
""" Fit on or off rates.

Using the binned times & binding series from binTimes.py,
bootstraps the medians to fit the on or off rates.

Sarah Denny """

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seqfun
import IMlibs
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import lmfit
import fitFun
import itertools    
import fileFun
from fitFun import objectiveFunctionOffRates, objectiveFunctionOnRates
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap on off rate fits')


parser.add_argument('-cs', '--cpseries', metavar="CPseries.pkl", required=True,
                   help='CPseries file containining the time series information')
parser.add_argument('-td', '--time_dict', metavar="timeDict.p", required=True,
                   help='file containining the timing information per tile')
parser.add_argument('-an', '--annotated_clusters', metavar="CPannot.pkl", required=True,
                   help='annotated cluster file. Supply if you wish to take medians per variant.'
                   'If not provided, script will not take medians, otherwise it will.')

group = parser.add_argument_group('optional arguments')
parser.add_argument('--tile', metavar="NNN", default='001',
                   help='tile you wish to subset. Default="001"')
parser.add_argument('-ft', '--fittype', default='off', metavar="[off | on]",
                   help='fittype ["off" | "on"]. Default is "off" for off rates')


group = parser.add_argument_group('additional option arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of annotated_clusters filename')
group.add_argument('--n_samples', default=100, type=int, metavar="N",
                   help='number of times to bootstrap samples')
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores. default = 20')
group.add_argument('--subset',action="store_true", default=False,
                    help='if flagged, will only do a subset of the data for test purposes')


def getInitialParameters(times, fittype=None):
    """ Get standard set of fit parameters across all variants depending on fittype."""
    
    # default fittype is 'off'
    if fittype is None: fittype = 'off'
    
    # different lifetime name for association vs dissociation
    if fittype == 'off':
        param_name = 'koff'
    elif fittype == 'on':
        param_name = 'kobs'
    else:
        print 'Error: "fittype" not recognized. Valid options are "on" or "off".'
        sys.exit()
        
    # fit parameters structure has three parameters (fmax, lifeftime, fmin),
    # each has a lowerbound, upperbound, and initialization
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', param_name, 'fmin'])
    
    # fmin cannot be less than zero. Initial condition will be set based on
    # min median fluorescence of variant
    param = 'fmin'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    
    # fmax cannot be less than zero. Initial condition will be set based on
    # max median fluorescence of variant
    param = 'fmax'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    
    # rate parameter constraints.
    # (1) min lifetime (slowest) is that given by only 1% dropoff after the full time span.
    # (2) initial lifteime is that given by 50% dropoff within half the time span.
    # (3) max lifetime (fastest) is that given by 99% dropoff after 1/10 of the first
    # time interval.
    # note that 'min fraction bound' is 1 - the dropoff amount quoted above.
    fitParameters.loc[:, param_name] = [-np.log(min_fraction_decreased)/t_delta
     for t_delta, min_fraction_decreased in itertools.izip(
        [times.max()-times.min(), (times.max()-times.min())/2, (times[1]-times[0])/10.],
        [0.99,                    0.5,                         0.01])]

    return fitParameters


def perVariant(times, subSeries, fitParameters, func=None, plot=None,
               fittype=None, default_errors=None, n_samples=None):
    """ Fit a variant to objective function by bootstrapping median fluorescence. """
    
    # by default, don't plot results
    if plot is None:
        plot = False
    
    # estimate initial conditions based on min and max of median fluorescence of
    # all clusters
    a, b = np.percentile(subSeries.median().dropna(), [1, 99])
    fitParametersPer = fitParameters.copy()
    fitParametersPer.loc['initial', ['fmin', 'fmax']] = [a, b-a]
    
    # fit by least squares fitting
    results, singles = fitFun.bootstrapCurves(times, subSeries, fitParametersPer,
                                              func=func, enforce_fmax=False,
                                              use_default=True,
                                              n_samples=n_samples )
    # plot results
    if plot:
        plotFun.plotFitCurve(times, subSeries, results, fitParameters,
                                     log_axis=False, func=func, fittype=fittype)
    return results


def fitRates(table, times,
             func, fittype=None, n_samples=None, variants=None, renorm=False):
    """ Initiate fits and parallelize fitting of each variant. """
    
    # default fittype is 'off' for plotting
    if fittype is None: fittype = 'off'
    
    # find groupdict
    print 'Splitting data...'
    grouped = table.astype(float).groupby('variant_number')
    groupDict = {}
    for name, group in grouped:
        subSeries = group.iloc[:, 1:]
        groupDict[name] = group.iloc[:, 1:]
    
    if renorm:
        print 'Renormalizing the data by first point in time series...'
        # divide by first time point and don't use first time point in fit
        for name, subSeries in groupDict.items():
            groupDict[name] = np.divide(subSeries, np.vstack(subSeries.iloc[:, 0].values)).iloc[:, 1:]
        times = times[1:]
    
    # if 'variants' is provided, only fit a subset of the data 
    if variants is None:
        variants = groupDict.keys()
        
    # call each variant-level fit with function perVariant, in a parallelized framework.
    print '\tMultiprocessing bootstrapping...'
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(perVariant)(times, groupDict[variant], fitParameters,
                                     func=func, n_samples=n_samples,
                                     fittype=fittype)
                 for variant in variants))
    
    # concatenate results of parallelized fits.
    results = pd.concat(results, keys=variants, axis=1).transpose()
    return results

def checkFits(results, fittype=None):
    if fittype is None:
        fittype = 'off'
    
    if fittype == 'off':
        goodFit = ((results.fmax > results.fmin)&
                   (results.fmax_lb > 3*results.fmin_ub)&
                   #(results.fmin > 1E-2)&
                   (results.fmax > 2E-1)&
                   ((results.koff_ub - results.koff_lb)/2/results.koff < 1))
    elif fittype == 'on':
        goodFit = ((results.fmax > 0)&
                   (results.fmin > 7.5E-2)&
                   (results.fmax > 5E-2)&
                   (results.kobs > 0.000002)&
                   ((results.kobs_ub - results.kobs_lb)/results.kobs < 2))
    return goodFit




##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()
    annotatedClusterFile   = args.annotated_clusters
    outFile                = args.out_file
    bindingSeriesFile = args.cpseries
    timeDeltaFile        = args.time_dict
    fittype              = args.fittype
    numCores             = args.numCores
    n_samples            = args.n_samples
    tile_to_subset = args.tile
    
    # find out file
    if outFile is None:
        outFile = fileFun.stripExtension(bindingSeriesFile)
    
        
    # depending on fittype, use one of two provided objective functions
    if fittype == 'off':
        func = objectiveFunctionOffRates
    elif fittype == 'on':
        func = objectiveFunctionOnRates
    else:
        print ('Error: fittype "%s" not recognized. Valid options are '
               '"on" or "off".')%fittype
        sys.exit()
        
    # laod data
    print 'Loading data..'
    # load time series data
    table = (pd.concat([fileFun.loadFile(annotatedClusterFile),
                        fileFun.loadFile(bindingSeriesFile)], axis=1).
             sort('variant_number'))
    times = np.array(fileFun.loadFile(timeDeltaFile)[tile_to_subset])
    fitParameters = getInitialParameters(times,
                                         fittype=fittype)
    
    # fit only clusters that are not all NaN
    table.dropna(axis=0, subset=table.columns[1:], how='all',inplace=True)
    
    variants = np.unique(table.variant_number.astype(float).dropna())
    if args.subset:
        variants = variants[-100:]
        outFile = outFile + '_subset'

    # fit curves
    results = fitRates(table, times, func, variants=variants, fittype=fittype, n_samples=n_samples)

    results.to_csv(outFile+'.CPvariant', sep='\t')

