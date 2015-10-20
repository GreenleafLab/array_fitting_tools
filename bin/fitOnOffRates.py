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


### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap on off rate fits')

group = parser.add_argument_group('inputs starting from time-binned binding series file')
group.add_argument('-b', '--binding_curves', required=True, metavar=".bindingSeries.pkl",
                   help='file containining the binding curve information'
                   ' binned over time.')
group.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')
group.add_argument('-t', '--times', metavar="times.txt",
                   help='file containining the binned times')
group.add_argument('-ft', '--fittype', default='off', metavar="[off | on]",
                   help='fittype ["off" | "on"]. Default is "off" for off rates')

group = parser.add_argument_group('additional option arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of annotated_clusters filename')
group.add_argument('--n_samples', default=100, type=int, metavar="N",
                   help='number of times to bootstrap samples')
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores. default = 20')
group.add_argument('--init', action="store_true", default=False, 
                   help='flag if you wish to simply initiate fitting, not actually fit')

def objectiveFunctionOffRates(params, times, data=None, weights=None):
    """ Return fit value, residuals, or weighted residuals of off rate objective function. """
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    koff = parvals['koff']
    fmin = parvals['fmin']
    fracbound = fmin + (fmax - fmin)*np.exp(-koff*times)

    if data is None:
        return fracbound
    elif weights is None:
        return fracbound - data
    else:
        return (fracbound - data)*weights
    
def objectiveFunctionOnRates(params, times, data=None, weights=None):
    """ Return fit value, residuals, or weighted residuals of on rate objective function. """
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    koff = parvals['kobs']
    fmin = parvals['fmin']
    fracbound = fmin + fmax*(1 - np.exp(-koff*times));

    if data is None:
        return fracbound
    elif weights is None:
        return fracbound - data
    else:
        return (fracbound - data)*weights   

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
                                              default_errors=default_errors,
                                              n_samples=n_samples,
                                              use_default=True)
    # plot results
    if plot:
        fitFun.plotFitCurve(times,
                                     subSeries,
                                     results,
                                     fitParameters,
                                     log_axis=False, func=func, fittype=fittype,
                                     default_errors=default_errors,
                                     use_default=True)
    return results

def initiateFits(bindingCurveFilename, timesFilename, annotatedClusterFile):
    """ Loads all relevant data for fitting. """
    print "Loading time series and splitting by variants..."
    
    # load time series data
    table = (pd.concat([pd.read_pickle(annotatedClusterFile),
                        pd.read_pickle(bindingCurveFilename)], axis=1).
             sort('variant_number'))
    
    # fit all labeled variants
    table.dropna(axis=0, subset=['variant_number'], inplace=True)

    # fit only clusters that are not all NaN
    table.dropna(axis=0, subset=table.columns[1:], how='all',inplace=True)
    
    # get deafult errors
    default_errors = table.astype(float).groupby('variant_number').std().mean()
    
    # load times
    times = np.loadtxt(timesFilename)
    fitParameters = getInitialParameters(times,
                                         fittype=fittype)
    # group by variant number for bootstrapping fits.
    grouped = table.groupby('variant_number')
    groupDict = {}
    for name, group in grouped:
        groupDict[name] = group.iloc[:, 1:].astype(float)
    return times, groupDict, fitParameters, table, default_errors

def fitRates(bindingCurveFilename, timesFilename, annotatedClusterFile,
             func, fittype=None, n_samples=None, variants=None):
    """ Initiate fits and parallelize fitting of each variant. """
    
    # default fittype is 'off' for plotting
    if fittype is None: fittype = 'off'
        
    # load relevant data for fitting
    times, groupDict, fitParameters, table, default_errors = (
        initiateFits(bindingCurveFilename, timesFilename, annotatedClusterFile))
    
    # if 'variants' is provided, only fit a subset of the data 
    if variants is None:
        variants = groupDict.keys()
        
    # call each variant-level fit with function perVariant, in a parallelized framework.
    print '\tMultiprocessing bootstrapping...'
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(perVariant)(times, groupDict[name], fitParameters,
                                     func=func, n_samples=n_samples,
                                     default_errors=default_errors, fittype=fittype)
                 for name in variants))
    
    # concatenate results of parallelized fits.
    results = pd.concat(results, keys=variants, axis=1).transpose()
    return results

def checkFits(results, fittype=None):
    if fittype is None:
        fittype = 'off'
    
    if fittype == 'off':
        goodFit = ((results.fmax > results.fmin)&
                   (results.fmax_lb > 3*results.fmin_ub)&
                   (results.fmin > 1E-2)&
                   (results.fmax > 2E-1)&
                   ((results.koff_ub - results.koff_lb)/results.koff < 2))
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
    bindingCurveFilename = args.binding_curves
    timesFilename        = args.times
    fittype              = args.fittype
    numCores             = args.numCores
    n_samples            = args.n_samples
    # find out file
    if outFile is None:
        outFile = os.path.splitext(
            bindingCurveFilename[:bindingCurveFilename.find('.pkl')])[0]
    
    # process inputs
    if timesFilename is None:
        timesFilename = outFile + '.times.pkl'
    
    if bindingCurveFilename is None:
        bindingCurveFilename = outFile + '.bindingCurve.pkl'
        
    # depending on fittype, use one of two provided objective functions
    if fittype == 'off':
        func = objectiveFunctionOffRates
    elif fittype == 'on':
        func = objectiveFunctionOnRates
    else:
        print ('Error: fittype "%s" not recognized. Valid options are '
               '"on" or "off".')%fittype
        sys.exit()
        
    # fit curves
    if not args.init:
        results = fitRates(bindingCurveFilename, timesFilename, annotatedClusterFile,
                           func, fittype=fittype, n_samples=n_samples)
    
        results.to_csv(outFile+'.CPresults', sep='\t')
        sys.exit()
    else:
        # just initiate fitting
        times, groupDict, fitParameters, table, default_errors = (
            initiateFits(bindingCurveFilename,
                         timesFilename,
                         annotatedClusterFile))
        results = pd.read_table(outFile+'.CPresults', index_col=0)
    sys.exit()
    
    # plot all variants
    figDirectory = './'
    for variant in results.index:
        try:
            fitFun.plotFitCurve(times, groupDict[variant],
                                         results.loc[variant],
                                 fitParameters, log_axis=False,
                                 func=objectiveFunctionOffRates, fittype='off',
                                 default_errors=default_errors,use_default=True)
            plt.ylim([0, 1.5])
            plt.savefig(os.path.join(figDirectory, 'off_rate_curve.variant_%d.pdf'%variant))
        except:
            print 'issue with variant %d'%variant
        plt.close()
