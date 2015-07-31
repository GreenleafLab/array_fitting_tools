#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import sys
import seqfun
import scipy.stats as st
import IMlibs
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF               
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import lmfit
import fitFun
from fitFun import fittingParameters
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='fit single clusters to binding curve')

group = parser.add_argument_group('required arguments for fitting single clusters')
group.add_argument('-cs', '--cpsignal', required=True,
                    help='reduced CPsignal file')
group.add_argument('-c', '--concentrations', required=True,
                    help='text file giving the associated concentrations')

group = parser.add_argument_group('optional arguments for fitting single clusters')
group.add_argument('-od','--output_dir', default="binding_curves",
                    help='save output files to here. default = ./binding_curves')
group.add_argument('-fp','--filterPos', nargs='+', help='set of filters '
                    'that designate clusters to fit. If not set, use all')                        
group.add_argument('-fn','--filterNeg',  nargs='+', help='set of filters '
                     'that designate "background" clusters. If not set, assume '
                     'complement to filterPos')
group.add_argument('-bp', '--binding_point', type=int, default=-1,
                    help='point in binding series to use for null scores (Default is '
                    'last concentration. -2 for second to last concentration)', )


group = parser.add_argument_group('other settings')
group.add_argument('-n','--num_cores', type=int, default=1,
                    help='maximum number of cores to use. default=1')



def getQvalue(bindingSeries, null_column, null_scores=None):
    if null_scores is not None:
        ecdf = ECDF(pd.Series(null_scores).dropna())
        qvalues = pd.Series(1-ecdf(bindingSeries.iloc[:, null_column].dropna()),
                            index=bindingSeries.iloc[:, null_column].dropna().index) 
        return qvalues
    else:
        return pd.Series(index=bindingSeries.index)


def getEstimatedBinders(bindingSeries, null_column, null_scores=None,
                        nonBinders=None, num_clusters=None, qvalue_cutoff=None):
    parameters = fittingParameters()
    if num_clusters is None:
        num_clusters = parameters.num_clusters
    if nonBinders is None:
        nonBinders = False # by default, get binders
    
    if qvalue_cutoff is None:
        if nonBinders:
            qvalue_cutoff = parameters.qvalue_cutoff_nonbinders
        else:
            qvalue_cutoff = parameters.qvalue_cutoff_binders

    
    if null_scores is not None:

        # get binding estimation 
        qvalues = getQvalue(bindingSeries, null_column, null_scores)
        
        # first filter: qvalue cutoff
        if nonBinders:
            # for nonbinders, sort in descending order and take those above qvalue cutoff
            subset = qvalues > qvalue_cutoff
        else:
            # for binders, sort in ascending order and take those below qvalue cutoff
            subset = qvalues <= qvalue_cutoff
    
        index = qvalues.loc[subset].index
    else:
        # if null scores can't be found based on filterPos or Neg,
        # just take top num_cluster binders
        fluorescence = bindingSeries.iloc[:, null_column].copy()
        if nonBinders:
            fluorescence.sort(ascending=True)
        else:
            fluorescence.sort(ascending=False)
        index = fluorescence.iloc[:num_clusters].index
    return index

def plotInitialEstimates(bindingSeriesNorm, null_column, indexBinders=None, indexNonBinders=None):
    # get bins
    yvalues = bindingSeriesNorm.iloc[:, null_column]
    yvalues.loc[np.logical_not(np.isfinite(yvalues))] = np.nan
    yvalues.dropna(inplace=True)

    lowerbound = 0
    upperbound = yvalues.median() + 5*yvalues.std()
    binedges = np.linspace(lowerbound, upperbound, 100)
    
    fig = plt.figure(figsize=(4,4))
    ax1 = fig.add_subplot(111)
    ax1.hist(yvalues.values, bins=binedges, histtype='stepfilled', color='grey',
             alpha=0.5, label='all',)
    
    if indexBinders is not None:
        ax2 = ax1.twinx()
        ax2.hist(yvalues.loc[indexBinders].values, bins=binedges, histtype='stepfilled',
                 color='red', alpha=0.5, label='binder',)
    if indexNonBinders is not None:
        ax1.hist(yvalues.loc[indexNonBinders].values, bins=binedges, histtype='stepfilled',
                 color='blue', alpha=0.5, label='nonbinder',)
    plt.legend(loc='upper right')
    plt.xlabel('fluorescence')
    plt.ylabel('count')
    plt.tight_layout()
    ax1.tick_params(right='off', top='off')
    return

def getInitialFitParameters(bindingSeriesNorm, concentrations, indexBinders,
                            indexNonBinders, assumeSaturation=None, saturationLevel=None):
    parameters = fittingParameters()
    if assumeSaturation is None:
        assumeSaturation = False

    if saturationLevel is None:    
        if assumeSaturation:
            saturationLevel = 1
        else:
            saturationLevel =  parameters.saturation_level # assume these non binders are at least 50% bound
        
    parameters = fittingParameters(concentrations=concentrations)
    
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    
    # find fmin
    param = 'fmin'
    fitParameters.loc[:, param] = fitFun.getBoundsGivenDistribution(
        bindingSeriesNorm.loc[indexNonBinders].iloc[:,0], label=param)

    # find fmax
    param = 'fmax'
    fitParameters.loc[:, param] = fitFun.getBoundsGivenDistribution(
        bindingSeriesNorm.loc[indexBinders].iloc[:,-1], label=param,
        saturation_level=saturationLevel)
    fitParameters.loc['upperbound', param] = fitParameters.loc['upperbound', param]
    
    # find dG
    fitParameters.loc[:, 'dG'] = parameters.dGparam
 
    return fitParameters

# define functions
def bindingSeriesByCluster(reducedCPsignalFile, concentrations, binding_point,
                           numCores=None, backgroundTileFile=None,
                           filterPos=None, filterNeg=None,
                           num_clusters=None, subset=None,
                           ):
    if subset is None:
        subset = False

    # get binding series
    print '\tLoading binding series and all RNA signal:'
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
        reducedCPsignalFile, concentrations=concentrations)
    
    # make normalized binding series
    allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # get null scores
    if (filterPos is not None or filterNeg is not None) and backgroundTileFile is not None:
        null_scores = IMlibs.loadNullScores(backgroundTileFile,
                                            filterPos=filterPos,
                                            filterNeg=filterNeg,
                                            binding_point=binding_point,)
    else:
        null_scores=None

    # estimate binders and nonbinders
    indexBinders = getEstimatedBinders(bindingSeries, binding_point,
                                       null_scores=null_scores, num_clusters=num_clusters)
    indexNonBinders = getEstimatedBinders(bindingSeries, binding_point,
                                       null_scores=null_scores, num_clusters=num_clusters,
                                       nonBinders=True)
    # plot
    plotInitialEstimates(bindingSeriesNorm, binding_point, indexBinders, indexNonBinders)
    
    # find initial parameters
    fitParameters = getInitialFitParameters(bindingSeriesNorm, concentrations,
                                            indexBinders, indexNonBinders,
                                            assumeSaturation=False)

    maxInitialBinders = 1E4
    if maxInitialBinders < len(indexBinders):
        index = indexBinders[np.linspace(0, len(indexBinders)-1,
                                         maxInitialBinders).astype(int)]
    else:
        index = indexBinders
    fitUnconstrained = IMlibs.splitAndFit(bindingSeriesNorm, 
                                          concentrations, fitParameters, numCores,
                                          index=index, mod_fmin=False)
    fitParameters.loc[:, 'fmax'] = fitFun.getBoundsGivenDistribution(fitUnconstrained.fmax)

    # now refit all remaining clusters
    print ('Fitting all with constraints on fmax (%4.2f, %4.2f, %4.2f)'
           %(fitParameters.loc['lowerbound', 'fmax'],
             fitParameters.loc['initial', 'fmax'],
             fitParameters.loc['upperbound', 'fmax']))
    print ('Fitting all with constraints on fmin (%4.4f, %4.4f, %4.4f)'
           %(fitParameters.loc['lowerbound', 'fmin'],
             fitParameters.loc['initial', 'fmin'],
             fitParameters.loc['upperbound', 'fmin']))

    # sort by fluorescence in null_column to try to get groups of equal
    # distributions of binders/nonbinders
    fluorescence = bindingSeriesNorm.iloc[:, binding_point].copy()
    fluorescence.sort()
    index_all = bindingSeriesNorm.loc[fluorescence.index].dropna(axis=0, thresh=4).index

    if subset:
        num = 5E3
        index_all = index_all[np.linspace(0, len(index_all)-1, num).astype(int)]
        
    fitConstrained = IMlibs.splitAndFit(bindingSeriesNorm, concentrations,
                                        fitParameters, numCores, index=index_all)
    fitConstrained = pd.concat([fitConstrained,
                                pd.DataFrame(getQvalue(bindingSeries, binding_point, null_scores),
                                             columns=['qvalue'])], axis=1)

    return fitConstrained, fitParameters, bindingSeriesNorm
    



if __name__=="__main__":    
    args = parser.parse_args()

    concentrations = np.loadtxt(args.concentrations)
    fitConstrained, fitParameters, bindingSeriesNorm = (
        fitFun.bindingSeriesByCluster(
            reducedCPsignalFile, concentrations, args.binding_point, numCores=numCores,
            backgroundTileFile=backgroundTileFile,
            filterPos=args.filterPos, filterNeg=args.filterNeg, num_clusters=None))