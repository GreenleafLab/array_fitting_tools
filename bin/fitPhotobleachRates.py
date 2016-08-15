#!/usr/bin/env python
""" Fit on or off rates.

using

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
from scikits.bootstrap import bootstrap
import fitFun
from fitFun import objectiveFunctionOffRates, objectiveFunctionOnRates
import itertools
import warnings
import fileFun
import singleClusterFits
import findFmaxDist

### MAIN ###
#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap on off rate fits')
parser.add_argument('-cs', '--cpseries', metavar="CPseries.pkl",
                   help='CPseries file containining the fluorescence information')
parser.add_argument('-t', '--tiles', metavar="CPtiles.pkl",
                   help='CPtiles file giving the tile per cluster')
parser.add_argument('-i', '--image_n_dict', metavar="image_ns.p",
                   help='file giving the image number for all tiles')

group = parser.add_argument_group('additional option arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of -cs (CPseries) filename')
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores. default = 20')


def photobleachObjectiveFunction(params, n,  data=None, weights=None, index=None):
    if index is None:
        index = np.ones(len(n)).astype(bool)

    parvals = params.valuesdict()
    alpha = parvals['alpha']
    fmax = parvals['fmax']

    y = fmax*np.power(alpha, n)
    # return fit value of data is not given
    if data is None:
        return y[index]
    
    # return residuals if data is given
    elif weights is None:
        return (y - data)[index]
    
    # return weighted residuals if data is given
    else:
        return ((y - data)*weights)[index]
    
    
def photobleachWithTimeObjectiveFunction(params, n,  data=None, weights=None, index=None, times=None):
    if index is None:
        index = np.ones(len(n)).astype(bool)

    parvals = params.valuesdict()
    alpha = parvals['alpha']
    fmax = parvals['fmax']
    koff = parvals['koff']
    fmin = parvals['fmin']

    y = fmin + (fmax-fmin)*np.power(alpha, n)*np.exp(-koff*times)
    # return fit value of data is not given
    if data is None:
        return y[index]
    
    # return residuals if data is given
    elif weights is None:
        return (y - data)[index]
    
    # return weighted residuals if data is given
    else:
        return ((y - data)*weights)[index]

def plotResults(results, n,  fitParameters, data=None, ax=None, x=None, func=photobleachObjectiveFunction, kwargs={}):
    
    params = fitFun.returnParamsFromResults(results, param_names=fitParameters.columns)
    if x is None:
        x=n
    
    if ax is None:  
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
    ax.plot(x, func(params, n, **kwargs), 'k-')
    
    if data is not None:
        ax.plot(x, data, 'ro')
    

##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()
    outFile  = args.out_file
    
    a = fileFun.loadFile(args.cpseries)
    tileSeries =fileFun.loadFile(args.tiles)
    imageNDict = fileFun.loadFile(args.image_n_dict)
    numCores = args.numCores
    
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['alpha'])
    fitParameters.loc[:, 'alpha'] = [0, 0.985, 1]
    fitParameters.loc[:, 'fmax'] = [0, a.mean().loc[0], np.inf]

    first_point = 0
    results = {}
    for tile, image_ns in imageNDict.items():
        if len(image_ns) > 3:
            vec = a.loc[tileSeries==tile].mean()[:len(image_ns)]
            
            results[tile] = fitFun.fitSingleCurve(image_ns[image_ns>=first_point], vec.iloc[image_ns>=first_point],
                                            fitParameters, func=photobleachObjectiveFunction)
            #plotResults(results[tile], image_ns, fitParameters, data=vec)
            #plt.title(tile)
    results_all = pd.concat(results, axis=1).mean(axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plotResults(results_all, np.arange(40), fitParameters, ax=ax)
    for tile, image_ns in imageNDict.items():
        if len(image_ns) > 3:
            vec = a.loc[tileSeries==tile].mean()[:len(image_ns)]
        
            ax.plot(image_ns, vec, '.')
    sys.exit()
    
    # b is the stuff that is actually bright
    b = fileFun.loadFile('photobleachRate/AFFYB_ALL_Bottom_filtered_reduced_normalized_KL120.CPseries.pkl')
    
    # make a bg only clusters
    a = a.loc[pd.Series(np.logical_not(np.in1d(np.array(a.index.tolist()),
                                               np.array(b.index.tolist()))), index=a.index)]
    
    for tile in imageNDict.keys():
        image_ns = imageNDict[tile]
        if len(image_ns) > 3:
            vec = b.loc[tileSeries==tile].mean()[:len(image_ns)]
            vec_bg = a.loc[tileSeries==tile].mean()[:len(image_ns)]
            
            vec = vec - vec_bg
            results[tile] = fitFun.fitSingleCurve(image_ns[image_ns>=first_point], vec.iloc[image_ns>=first_point],
                                            fitParameters, func=photobleachObjectiveFunction)
            plotResults(results[tile], image_ns, fitParameters, data=vec)
            plt.title(tile)   
    
    
    
    vec_tile1 = a.loc[tileSeries=='001'].dropna(how='all', axis=1).mean()

    #
    #fitParameters_time = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
    #                             columns=['fmax', 'koff', 'fmin'])
    fitParameters_time = fitParameters.copy()
    fitParameters_time.loc[:, 'koff'] = [np.power(10., -8), 0.001998, np.power(10., -1)]
    fitParameters_time.loc[:, 'fmin'] = [0, 0.1, np.inf]
    fitParameters_time.loc[:, 'fmax'] = [0, vec.iloc[0], np.inf]
    fitParameters_time.loc['vary'] = True
    fitParameters_time.loc['vary', 'koff'] = False
    fitParameters_time.loc['initial', 'koff'] =5.7E-4
    kwargs = {'times':np.array(times[tile])}
    results_time = fitFun.fitSingleCurve( imageNDict[tile], vec, fitParameters_time, func=photobleachWithTimeObjectiveFunction, kwargs=kwargs)
   
    sys.exit()
    
    # make sure image_ns are right
    cpfluor_files = (
        pd.concat([pd.DataFrame(int(filename[filename.find('tile')+4:filename.find('tile')+7]),
                                columns=['tile'],
                                index=[filename[filename.find('2015'):-8]])
               for filename in os.listdir('AFFYB_14_dilution_2uM_offrates_green/CPfluor/')
               if os.path.splitext(filename)[1] == '.CPfluor']))
    
    tif_files = (
        pd.concat([pd.DataFrame(int(filename[filename.find('tile')+4:filename.find('_green')]),
                                columns=['tile'],
                                index=[filename[filename.find('2015'):-4]])
               for filename in os.listdir('AFFYB_14_dilution_2uM_offrates_green/')
               if os.path.splitext(filename)[1] == '.tif']))
    
    final = pd.concat([tif_files, cpfluor_files], axis=1, ignore_index=True)
    final.columns = ['original', 'fit']
    final.sort_index(inplace=True)
    
    imageNDict = {}
    for tile in np.unique(final.original):
        if tile < 10:
            format_tile = '00%d'%tile
        else:
            format_tile = '0%d'%tile
        
        subtile = final.loc[final.original==tile].copy()
        subtile.loc[:, 'imagen'] = np.arange(len(subtile))
        imageNDict[format_tile] = subtile.dropna(subset=['fit']).imagen.values
    
    ###### old stuff
    indices = np.random.choice(a.index, size=[100, len(a)])
    
    n_samples = 100
    results = []
    for index in indices[:n_samples]:
        results.append(fitFun.fitSingleCurve(image_ns, a.loc[index].mean(), fitParameters, func=photobleachObjectiveFunction))
    results = pd.concat(results, axis=1)
    
    sys.exit()
    n = 3
    results_first = []
    for index in indices[:n_samples]:
        results_first.append(fitFun.fitSingleCurve(image_ns[:n], a.loc[index].iloc[:, :n].mean(), fitParameters, func=photobleachObjectiveFunction))
    results_first = pd.concat(results_first, axis=1)

    results_last = []
    for index in indices[:n_samples]:
        results_last.append(fitFun.fitSingleCurve(image_ns[-n:], a.loc[index].iloc[:, -n:].mean(), fitParameters, func=photobleachObjectiveFunction))
    results_last = pd.concat(results_last, axis=1)


    params_last = fitFun.returnParamsFromResults(results_last.mean(axis=1), param_names=fitParameters.columns)
    params_first = fitFun.returnParamsFromResults(results_first.mean(axis=1), param_names=fitParameters.columns)
    params = fitFun.returnParamsFromResults(results.mean(axis=1), param_names=fitParameters.columns)

    more_ns = np.arange(image_ns.min(), image_ns.max()+1)

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.plot(more_ns, photobleachObjectiveFunction(params, more_ns), 'k-')
    
    ax.plot(image_ns, a.mean(), 'ro')
    fix_axes(ax)
    plt.xlabel('# images')
    plt.ylabel('average fluorescence')
    plt.tight_layout()

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.plot(image_ns, photobleachObjectiveFunction(params_last, image_ns), 'k-')
    ax.plot(image_ns, photobleachObjectiveFunction(params_first, image_ns), '--', color='b')
    
    ax.plot(image_ns, a.mean(), 'ro')
    fix_axes(ax)
    plt.xlabel('# images')
    plt.ylabel('average fluorescence')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.plot(times['001'], photobleachObjectiveFunction(params, image_ns), 'k-')
    ax.plot(times['001'], photobleachObjectiveFunction(params_first, image_ns), '--', color='b')
    ax.plot(times['001'], photobleachObjectiveFunction(params_last, image_ns), '-', color='0.5')

    ax.plot(times['001'], a.mean(), 'ro')
    fix_axes(ax)
    plt.xlabel('time (s)')
    plt.ylabel('average fluorescence')
    plt.tight_layout()
    
    
    params = fitFun.returnParamsFromResults(results, param_names=fitParameters.columns)
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.plot(times['001'], fitFun.objectiveFunctionOffRates(params, np.array(times['001'])), 'k-')

    ax.plot(times['001'], a.mean(), 'ro')
    fix_axes(ax)
    plt.xlabel('time (s)')
    plt.ylabel('average fluorescence')
    plt.tight_layout()


    sys.exit()
    #initial_fmax = findFmaxDist.fitGammaDistribution( a.iloc[:, 0].dropna(), plot=True, set_offset=0).mean()
    fitParameters = pd.concat([fitParameters, pd.DataFrame([[0, val, np.inf] for val in a.iloc[:,0]],
        index=['fmax_%d'%i for i in np.arange(len(a))],
        columns=['lowerbound', 'initial', 'upperbound']).transpose()], axis=1)

    image_n_mat = np.array([image_ns]*len(a))
    results = fitFun.fitSingleCurve(image_n_mat, a.values, fitParameters, func=photobleachObjectiveFunction, image_ns=image_n_mat)

    