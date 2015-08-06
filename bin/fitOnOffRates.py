"""
Sarah Denny
Stanford University

Using the compressed barcode file (from Lauren) and the list of designed variants,
figure out how many barcodes per designed library variant (d.l.v.), and plot histograms.
"""

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
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of annotated_clusters filename')
parser.add_argument('-ft', '--fittype', default='off', metavar="[off | on]",
                   help='fittype ["off" | "on"]. Default is "off" for off rates')

group = parser.add_argument_group('inputs starting from time-binned binding series file') 
group.add_argument('-b', '--binding_curves', metavar="bindingCurve.pkl",
                   help='file containining the binding curve information'
                   ' binned over time.')
group.add_argument('-t', '--times', metavar="times.txt",
                   help='file containining the binned times')

group = parser.add_argument_group('additional option arguments')
group.add_argument('--n_samples', default=100, type=int, metavar="N",
                   help='number of times to bootstrap samples')
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores')

def objectiveFunctionOffRates(params, times, data=None, weights=None):
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
    if fittype is None: fittype = 'off'
    if fittype == 'off':
        param_name = 'koff'
    elif fittype == 'on':
        param_name = 'kobs'
    else:
        print 'Error: "fittype" not recognized. Valid options are "on" or "off".'
        sys.exit()
    
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', param_name, 'fmin'])
    param = 'fmin'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    param = 'fmax'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    
    # rate parameters
    fitParameters.loc[:, param_name] = [-np.log(min_fraction_decreased)/t_delta
     for t_delta, min_fraction_decreased in itertools.izip(
        [times.max()-times.min(), (times.max()-times.min())/2, (times[1]-times[0])/10.],
        [0.99,                    0.5,                         0.01])]


    return fitParameters


def perVariant(times, subSeries, fitParameters, func=None, plot=None, fittype=None):
    if plot is None:
        plot = False
    a, b = np.percentile(subSeries.median().dropna(), [1, 99])
    fitParameters = fitParameters.copy()
    fitParameters.loc['initial', ['fmin', 'fmax']] = [a, b-a]

    results, singles = fitFun.bootstrapCurves(times, subSeries, fitParameters,
                                              func=func, enforce_fmax=False)
    if plot:
        fitFun.plotFitCurve(times,
                                     subSeries,
                                     results,
                                     fitParameters,
                                     log_axis=False, func=func, fittype=fittype)
    return results



def fitRates(bindingCurveFilename, timesFilename, annotatedClusterFile,
                fittype=None):
    if fittype is None: fittype = 'off'
    if fittype == 'off':
        func = objectiveFunctionOffRates
    elif fittype == 'on':
        func = objectiveFunctionOnRates
    else:
        print ('Error: fittype "%s" not recognized. Valid options are '
               '"on" or "off".')%fittype

    table = (pd.concat([pd.read_pickle(annotatedClusterFile),
                        pd.read_pickle(bindingCurveFilename)], axis=1).
             sort('variant_number'))
    
    # fit all labeled variants
    table.dropna(axis=0, subset=['variant_number'], inplace=True)

    # fit only clusters that are not all NaN
    table.dropna(axis=0, subset=table.columns[1:], how='all',inplace=True)
    
    # load times
    times = np.loadtxt(timesFilename)
    fitParameters = getInitialParameters(times,
                                         fittype=fittype)
    # group by variant number
    grouped = table.groupby('variant_number')
    groupDict = {}
    for name, group in grouped:
        groupDict[name] = group.iloc[:, 1:].astype(float)


    print '\tMultiprocessing bootstrapping...'
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(perVariant)(times, groupDict[name], fitParameters,
                                     func=func)
                 for name in groupDict.keys()))
    results = pd.concat(results, keys=[name for name in groupDict.iterkeys()], axis=1).transpose()
    return results

    


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
    
    # find out file
    if outFile is None:
        outFile = os.path.splitext(
            annotatedClusterFile[:annotatedClusterFile.find('.pkl')])[0]
    
    # process inputs
    if timesFilename is None:
        timesFilename = outFile + '.times.pkl'
    
    if bindingCurveFilename is None:
        bindingCurveFilename = outFile + '.bindingCurve.pkl'

    # fit curves
    results = fitRates(bindingCurveFilename, timesFilename, annotatedClusterFile,
                fittype=fittype)

    results.to_csv(outFile+'.CPvariant', sep='\t')
    
    sys.exit()
    
    
    # plot all variants
    figDirectory = 'PUF4/onRates/11.1nM/CPfitted/figs_2015-07-31/on_rate_curves'
    for variant in results.index:
        try:
            fitFun.plotSingleVariantFits(times.values, groupDict[variant],
                                         results.loc[variant],
                                 fitParameters, log_axis=False,
                                 func=objectiveFunctionOnRates, fittype='on')
            plt.ylim([0, 1.5])
            plt.savefig(os.path.join(figDirectory, 'on_rate_curve.variant_%d.pdf'%variant))
        except:
            print 'issue with variant %d'%variant
        plt.close()
