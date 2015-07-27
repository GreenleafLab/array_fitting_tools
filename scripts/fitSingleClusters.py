#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import sys
import seqfun
import IMlibs
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF               
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='fit single clusters to binding curve')
parser.add_argument('reducedCPseq', 
                    help='CPsignal file to fit')
parser.add_argument('-m', '--mapCPfluors',
                    help='map file giving the dir names to look for CPfluor files')
parser.add_argument('-n','--num_cores', type=int, default=1,
                    help='maximum number of cores to use. default=1')
parser.add_argument('-nc', '--null_column', type=int, default=-1,
                    help='point in binding series to use for null scores (Default is '
                    'last concentration. -2 for second to last concentration)', )


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
    if num_clusters is None:
        num_clusters = 1E5
    if nonBinders is None:
        nonBinders = False # by default, get binders
    
    if qvalue_cutoff is None:
        if nonBinders:
            qvalue_cutoff = 0.8
        else:
            qvalue_cutoff = 0.005

    
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
    yvalues = bindingSeriesNorm.iloc[:, null_column].dropna()
    index = np.logical_not(seqfun.is_outlier(yvalues))
    binedges = np.linspace(yvalues.loc[index].min(), yvalues.loc[index].max(), 100)
    
    plt.figure(figsize=(4,4))
    plt.hist(yvalues.values, bins=binedges, histtype='stepfilled', color='grey',
             alpha=0.5, label='all',)
    if indexBinders is not None:
        plt.hist(yvalues.loc[indexBinders].values, bins=binedges, histtype='stepfilled',
                 color='red', alpha=0.5, label='binder',)
    if indexNonBinders is not None:
        plt.hist(yvalues.loc[indexNonBinders].values, bins=binedges, histtype='stepfilled',
                 color='blue', alpha=0.5, label='nonbinder',)
    plt.legend(loc='upper right')
    plt.xlabel('fluorescence')
    plt.ylabel('count')
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params(right='off', top='off')
    return

def plotFmaxMinDist(fDist, params, ):
    fDist.dropna(inplace=True)
    fmax_lb, fmax_initial, fmax_upperbound = params
    fig = plt.figure(figsize=(4,3));
    ax = fig.add_subplot(111)
    sns.distplot(fDist, color='r', hist_kws={'histtype':'stepfilled'}, ax=ax)
    ylim = [0, ax.get_ylim()[1]*1.1]
    ax.plot([fmax_lb]*2, ylim, 'k--')
    ax.plot([fmax_initial]*2, ylim, 'k:')
    ax.plot([fmax_upperbound]*2, ylim, 'k--')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.xlim(0, np.percentile(fDist, 100)*1.05)
    plt.ylim(ylim)
    plt.tight_layout()
    ax.tick_params(right='off', top='off')
    return ax

def findMaxProbability(x):
    counts, binedges = np.histogram(x, bins=np.linspace(x.min(), x.max(), 50))
    counts = counts[1:]; binedges=binedges[1:] # ignore first bin
    idx_max = np.argmax(counts)
    if idx_max != 0 and idx_max != len(counts)-1:
        return binedges[idx_max+1]
    else:
        return None

def getFitParameters(bindingSeriesNorm, concentrations, indexBinders, indexNonBinders):
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    
    # find fmin
    fminDist = seqfun.remove_outlier(bindingSeriesNorm.loc[indexNonBinders].iloc[:,0])
    fitParameters.loc['lowerbound', 'fmin'] = fminDist.min()
    fitParameters.loc['upperbound', 'fmin'] = fminDist.max()
    maxProb = findMaxProbability(fminDist)
    if maxProb is not None:
        fitParameters.loc['initial', 'fmin'] = maxProb
    else:
        fitParameters.loc['initial', 'fmin'] = fminDist.median()
    ax = plotFmaxMinDist(fminDist, fitParameters.fmin); ax.set_xlabel('fmin')
    
    # find fmax
    fmaxDist = seqfun.remove_outlier(bindingSeriesNorm.loc[indexBinders].iloc[:,-1])
    fitParameters.loc['lowerbound', 'fmax'] = fmaxDist.min()
    fitParameters.loc['upperbound', 'fmax'] = fmaxDist.max()
    maxProb = findMaxProbability(fmaxDist)
    if maxProb is not None:
        fitParameters.loc['initial', 'fmax'] = maxProb
    else:
        fitParameters.loc['initial', 'fmax'] = fmaxDist.median()    
    ax = plotFmaxMinDist(fmaxDist, fitParameters.fmax); ax.set_xlabel('fmax')
    
    # find dG
    fitParameters.loc['lowerbound', 'dG'] = (IMlibs.find_dG_from_Kd(
        IMlibs.find_Kd_from_frac_bound_concentration(0.99, concentrations[0])))
    fitParameters.loc['upperbound', 'dG'] = (IMlibs.find_dG_from_Kd(
        IMlibs.find_Kd_from_frac_bound_concentration(0.01, concentrations[-1])))    
    fitParameters.loc['initial', 'dG'] = (IMlibs.find_dG_from_Kd(
        IMlibs.find_Kd_from_frac_bound_concentration(0.5, concentrations[-1])))       
    return fitParameters

# define functions
def bindingSeriesByCluster(reducedCPsignalFile, concentrations, null_column,
                           numCores=None, signalNamesByTileDict=None,
                           filterPos=None, filterNeg=None,
                           num_clusters=None, subset=None):
    if subset is None:
        subset = False
    # get binding series
    print '\tLoading binding series and all RNA signal:'
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
        reducedCPsignalFile, concentrations, index_col=0)
    
    # make normalized binding series
    allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # get null scores
    if (filterPos is not None or filterNeg is not None) and signalNamesByTileDict is not None:
        null_scores = IMlibs.loadNullScores(signalNamesByTileDict,
                                            filterPos=filterPos,
                                            filterNeg=filterNeg,
                                            binding_point=null_column,)
    else:
        null_scores=None

    # estimate binders and nonbinders
    indexBinders = getEstimatedBinders(bindingSeries, null_column,
                                       null_scores=null_scores, num_clusters=num_clusters)
    indexNonBinders = getEstimatedBinders(bindingSeries, null_column,
                                       null_scores=null_scores, num_clusters=num_clusters,
                                       nonBinders=True)
    # plot
    plotInitialEstimates(bindingSeriesNorm, null_column, indexBinders, indexNonBinders)
    
    # find initial parameters
    fitParameters = getFitParameters(bindingSeriesNorm, concentrations, indexBinders, indexNonBinders)
    
    ## fit first round
    #print '\tFitting best binders with no constraints...'
    ## subsample to only do max
    #numToFit = 5E3
    #if len(indexBinders) > numToFit:
    #    indexBinders = indexBinders[np.linspace(0, len(indexBinders)-1, numToFit).astype(int)]
    #fitUnconstrained = IMlibs.splitAndFit(bindingSeriesNorm, 
    #                                      concentrations, fitParameters, numCores, index=indexBinders, mod_fmin=False)
    
        
    # now refit all remaining clusters
    print ('Fitting all with constraints on fmax (%4.2f, %4.2f, %4.2f)'
           %(fitParameters.loc['lowerbound', 'fmax'],
             fitParameters.loc['initial', 'fmax'],
             fitParameters.loc['upperbound', 'fmax']))
    print ('Fitting all with constraints on fmin (%4.4f, %4.4f, %4.4f)'
           %(fitParameters.loc['lowerbound', 'fmin'],
             fitParameters.loc['initial', 'fmin'],
             fitParameters.loc['upperbound', 'fmin']))
    
    ## save fit parameters
    #fitParametersFilename = os.path.join(os.path.dirname(fittedBindingFilename),
    #                                     'bindingParameters.%s.fp'%datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    #fitParameters.to_csv(fitParametersFilename, sep='\t')
    
    
    # sort by fluorescence in null_column to try to get groups of equal distributions of binders/nonbinders
    fluorescence = bindingSeriesNorm.iloc[:, null_column].copy()
    fluorescence.sort()
    index_all = bindingSeriesNorm.loc[fluorescence.index].dropna(axis=0, thresh=4).index

    if subset:
        num = 5E3
        index_all = index_all[np.linspace(0, len(index_all)-1, num).astype(int)]
        
    fitConstrained = IMlibs.splitAndFit(bindingSeriesNorm, concentrations,
                                                       fitParameters, numCores, index=index_all)
    fitConstrained = pd.concat([fitConstrained,
                                pd.DataFrame(getQvalue(bindingSeries, null_column, null_scores), columns=['qvalue']),
                                bindingSeriesNorm], axis=1)

    return fitConstrained, fitParameters

if __name__=="__main__":    
    args = parser.parse_args()

    tmp, tmp, concentrations = IMlibs.loadMapFile(args.mapCPfluors)
    fitConstrained, fitParameters = bindingSeriesByCluster(args.reducedCPseq, concentrations, args.null_columns)

