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

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-a', '--annotated_clusters', required=True,
                   help='file with clusters annotated by variant number')
parser.add_argument('-b', '--binding_curves', required=True,
                   help='file containining the binding curve information')
parser.add_argument('-t', '--times', required=True,
                   help='file containining the time information')
parser.add_argument('-out', '--out_file', required=True,
                   help='output filename')
parser.add_argument('-map', '--mapCPfluors', required=True,
                   help='map_file containing concentrations')


group = parser.add_argument_group('additional option arguments')
group.add_argument('--n_samples', default=100, type=int,
                   help='number of times to bootstrap samples')
group.add_argument('--not_pickled', default=False, action="store_true",
                   help='program assumes inputs are pickled. Flag if text files')
group.add_argument('-n', '--numCores', default=20, type=int,
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

def getInitialParameters(times, fluorescence):
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'koff', 'fmin'])
    param = 'fmin'
    fitParameters.loc[:, param] = [0, fluorescence.min(), 2*fluorescence.min()]
    param = 'fmax'
    fitParameters.loc[:, param] = [fluorescence.min(), fluorescence.max(),
                                   2*fluorescence.max()]
    param = 'koff'
    t_range = times.max() - times.min()
    min_fraction_decreased = 0.9
    # assuming fmin is zero
    fitParameters.loc['lowerbound', param] =  -np.log(min_fraction_decreased)/t_range
    
    t_delta = (times[1] - times[0])/10.
    min_fraction_decreased = 0.01
    fitParameters.loc['upperbound', param] =  -np.log(min_fraction_decreased)/t_delta
    
    t_delta = (times.max() - times.min())/2.
    min_fraction_decreased = 0.5   
    fitParameters.loc['initial', param] =  -np.log(min_fraction_decreased)/t_delta

    return fitParameters

def fitVariants(subSeries, times, plot=None, plot_dists=None):
    if plot is None:
        plot=False
    if plot_dists is None:
        plot_dists=plot
    fitParameters = getInitialParameters(times, subSeries.median())
    
    # find errors
    try:
        eminus, eplus = fitFun.findErrorBarsBindingCurve(subSeries)
    except:
        eminus, eplus = [np.ones(len(times))*np.nan]*2
    # bootStrap variants
    n_tests = len(subSeries)
    n_samples = 100
    indices = np.random.choice(subSeries.index,
                               size=(n_samples, n_tests), replace=True)
    singles = {}
    for i, clusters in enumerate(indices):
        if plot_dists:
            if i%(n_tests/10.)==0:
                print 'working on %d out of %d, %d%%'%(i, n_tests, i/float(n_tests)*100)
        fluorescence = subSeries.loc[clusters].median()
        index = np.isfinite(fluorescence)
        singles[i] = fitFun.fitSingleBindingCurve(times[index.values], fluorescence.loc[index],
                                                  fitParameters,
                                                  func=objectiveFunctionOffRates,
                                                  errors=None,
                                                  plot=False, log_axis=False)
    singles = pd.concat(singles, axis=1).transpose()
    
    param_names = fitParameters.columns.tolist()
    not_outliers = ~seqfun.is_outlier(singles.koff)
    data = np.ravel([[np.percentile(singles.loc[not_outliers, param], idx) for param in param_names] for idx in [50, 2.5, 97.5]])
    index = param_names + ['%s_lb'%param for param in param_names] + ['%s_ub'%param for param in param_names]
    results = pd.Series(index = index, data=data)




def fitOffRates():

    table = pd.concat([pd.read_pickle(annotatedClusterFile),
                       pd.read_pickle(bindingCurveFilename)], axis=1).sort('variant_number')
    
    times = np.loadtxt(timesFilename)
    
    grouped = table.groupby('variant_number')
    groupDict = {}
    for name, group in grouped:
        groupDict[name] = group.drop('variant_number', axis=1).dropna(axis=0,
                                                                      thresh=4)


    



##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()
    
    fittedBindingFilename = args.single_cluster_fits
    annotatedClusterFile  = args.annotated_clusters
    bindingCurveFilename  = args.binding_curves
    pickled = not args.not_pickled
    n_samples = args.n_samples
    numCores = args.numCores
