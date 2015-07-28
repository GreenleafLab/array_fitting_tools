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

class fittingParameters():
    
    """
    stores some parameters and functions
    """
    def __init__(self, concentrations=None, params=None, fitParameters=None,
                 default_errors=None):

        
        # save the units of concentration given in the binding series
        self.concentration_units = 1E-9 # i.e. nM
        self.RT = 0.582
        
        # if null scores are provided, use them to estimate binders and non-
        # binders. 'qvalue' is estimated based on the empircal null distribution
        # of these null scores. Binders are everything with qvalue less than
        # 'qvalue_cutoff_binders', Nonbinders are clusters with qvalue greater
        # than 'qvalue_cutoff_nonbinders'. 
        self.qvalue_cutoff_binders = 0.005
        self.qvalue_cutoff_nonbinders = 0.8
        
        # if null scores is not provided, rank clusters by fluorescence in the
        # last point (or alternately 'binding point') of the binding series.
        # take the top and bottom 'num_clusters' as accurately representing
        # binders and non binders. I've found 100K to be a good number, but
        # change this value to be smaller if poor separation is seen.
        self.num_clusters = 1E5
        
        # When constraining the upper and lower bounds of dG, say you only think
        # can fit binding curves if at most it is 99% bound in the first
        # point of the binding series. This defines 'frac_bound_lowerbound'.
        # 'frac_bound_upperbound' is the minimum binding at the last point of the
        # binding series that you think you can still fit.
        self.frac_bound_upperbound = 0.01
        self.frac_bound_lowerbound = 0.99
        self.frac_bound_initial = 0.5

        # also add other things
        self.params = params
        self.fitParameters = fitParameters
        self.default_errors = default_errors

        # if concentrations are defined, do some more things
        if concentrations is not None:
            self.concentrations = concentrations
            self.maxdG = self.find_dG_from_Kd(
                self.find_Kd_from_frac_bound_concentration(0.9,
                                                           concentrations[-1]))
            self.mindG = self.find_dG_from_Kd(
                self.find_Kd_from_frac_bound_concentration(0.5,
                                                            concentrations[-1]))
            
            # get dG upper and lowerbounds
            self.dGparam = pd.Series(index=['lowerbound', 'initial', 'upperbound'])
            self.dGparam.loc['lowerbound'] = self.find_dG_from_Kd(
                self.find_Kd_from_frac_bound_concentration(self.frac_bound_lowerbound,
                                                           concentrations[0]))
            self.dGparam.loc['upperbound'] = self.find_dG_from_Kd(
                self.find_Kd_from_frac_bound_concentration(self.frac_bound_upperbound,
                                                           concentrations[-1]))
            self.dGparam.loc['initial'] = self.find_dG_from_Kd(
                self.find_Kd_from_frac_bound_concentration(self.frac_bound_initial,
                                                           concentrations[-1]))
        

                
            

    def find_dG_from_Kd(self, Kd):
        return self.RT*np.log(Kd*self.concentration_units)

    def find_Kd_from_dG(self, dG):
        return np.exp(dG/self.RT)/self.concentration_units
    
    def find_Kd_from_frac_bound_concentration(self, frac_bound, concentration):
        return concentration/float(frac_bound) - concentration
    
    def find_fmax_bounds_given_n(self, n, alpha=None, return_dist=None):
        if alpha is None: alpha = 0.99
        if return_dist is None: return_dist = False
        
        if self.params is None:
            print 'Error: define popts'
            return
        params = self.params
        sigma = objectiveFunction(params, n)
        mean = params.valuesdict()['median']
        
        if return_dist:
            return st.norm(loc=mean, scale=sigma)
        else:
            interval = st.norm.interval(alpha, loc=mean, scale=sigma)
            return interval[0], mean, interval[1]

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

def findMaxProbability(x, numBins=None):
    if numBins is None:
        numBins = 200
    counts, binedges = np.histogram(x, bins=np.linspace(x.min(), x.max(), numBins))
    counts = counts[1:]; binedges=binedges[1:] # ignore first bin
    idx_max = np.argmax(counts)
    if idx_max != 0 and idx_max != len(counts)-1:
        return binedges[idx_max+1]
    else:
        return None
    
def getBoundsGivenDistribution(values, label=None):
    fitParameters = pd.Series(index=['lowerbound', 'initial', 'upperbound'])
    fDist = seqfun.remove_outlier(values)
    fitParameters.loc['lowerbound'] = fDist.min()
    fitParameters.loc['upperbound'] = fDist.max()
    maxProb = findMaxProbability(fDist)
    if maxProb is not None:
        fitParameters.loc['initial'] = maxProb
    else:
        fitParameters.loc['initial'] = fDist.median()
    ax = plotFmaxMinDist(fDist, fitParameters);
    if label is not None:
        ax.set_xlabel(label)
    return fitParameters


def getInitialFitParameters(bindingSeriesNorm, concentrations, indexBinders, indexNonBinders):
    parameters = fittingParameters(concentrations=concentrations)
    
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    
    # find fmin
    param = 'fmin'
    fitParameters.loc[:, param] = getBoundsGivenDistribution(
        bindingSeriesNorm.loc[indexNonBinders].iloc[:,0], label=param)

    # find fmax
    param = 'fmax'
    fitParameters.loc[:, param] = getBoundsGivenDistribution(
        bindingSeriesNorm.loc[indexBinders].iloc[:,-1], label=param)
    
    # find dG
    fitParameters.loc[:, 'dG'] = parameters.dGparam
 
    return fitParameters

# define functions
def bindingSeriesByCluster(reducedCPsignalFile, concentrations, binding_point,
                           numCores=None, bindingSeriesByCluster=None,
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
    if (filterPos is not None or filterNeg is not None) and bindingSeriesByCluster is not None:
        null_scores = IMlibs.loadNullScores(bindingSeriesByCluster,
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
                                            indexBinders, indexNonBinders)
            
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
                                             columns=['qvalue']),
                                bindingSeriesNorm], axis=1)

    return fitConstrained, fitParameters

def findFinalBoundsParameters(table, concentrations, use_actual=None):
    parameters = fittingParameters(concentrations=concentrations)

    # actual data
    param_names = ['dG', 'fmax', 'fmin']
    grouped = table.groupby('variant_number')
    grouped_binders = pd.concat([grouped.count().loc[:, 'fmax'],
                                 grouped.median().loc[:, param_names]], axis=1);
    grouped_binders.columns = ['number'] + param_names
    tight_binders = grouped_binders.loc[grouped_binders.dG <= parameters.maxdG]
    
    # if at least 20 data points have at least 10 counts in that bin, use actual
    # data. This statistics seem reasonable for fitting
    if use_actual is None:
        counts, binedges = np.histogram(tight_binders.number,
                                        np.arange(1, tight_binders.number.max()))
        if (counts > 10).sum() >= 20:
            use_actual = True
        else:
            use_actual = False
    else:
        # whether you use actual or 'simulated' data depends on input
        pass
        
    stds_actual = pd.Series(index=np.unique(tight_binders.number))
    weights = pd.Series(index=np.unique(tight_binders.number))
    for n in stds_actual.index:
        stds_actual.loc[n] = tight_binders.loc[tight_binders.number==n, 'fmax'].std()
        weights.loc[n] = np.sqrt((tight_binders.number==n).sum())
    stds_actual.dropna(inplace=True)
    weights = weights.loc[stds_actual.index]
    
    # also do 'simulated' variants
    other_binders = table.loc[np.in1d(table.variant_number, tight_binders.index),
                              ['variant_number', 'dG', 'fmax', 'fmin']]
    
    # for each n, choose n variants
    stds = pd.Series(index=np.arange(1, 101, 4))
    for n in stds.index:
        print n
        n_reps = np.ceil(float(len(other_binders))/n)
        index = np.random.permutation(np.tile(np.arange(n_reps), n))[:len(other_binders)]
        other_binders.loc[:, 'faux_variant'] = index
        stds.loc[n] = other_binders.groupby('faux_variant').median().fmax.std()
    stds.dropna(inplace=True)
        
    # fit to curve
    params = lmfit.Parameters()

    if use_actual:
        params.add('sigma', value=stds_actual.iloc[0], min=0)
        params.add('c',     value=stds_actual.min(),   min=0)
        lmfit.minimize(objectiveFunction, params,
                                       args=(stds_actual.index,),
                                       kws={'y':stds_actual.values,
                                            'weights':weights},
                                       xtol=1E-6, ftol=1E-6, maxfev=10000)
    else:
        params.add('sigma', value=stds.iloc[0], min=0)
        params.add('c',     value=stds.min(),   min=0)
        lmfit.minimize(objectiveFunction, params,
                                       args=(stds.index,),
                                       kws={'y':stds.values,
                                            'weights':None},
                                       xtol=1E-6, ftol=1E-6, maxfev=10000)
    
    # plot data
    plt.figure(figsize=(4,3));
    
    if use_actual:
        plt.scatter(stds_actual.index, stds_actual, color='k', marker='o',
                    label='actual');
        plt.plot(stds.index, stds, 'r:', label='simulated')
        plt.plot(stds_actual.index, objectiveFunction(params, stds_actual.index), 'c',
                 label='fit')
    else:
        plt.scatter(stds.index, stds, color='k', marker='o', label='simulated');
        plt.scatter(stds_actual.index, stds_actual,  color='r', marker='x', label='actual');
        plt.plot(stds.index, objectiveFunction(params, stds.index), 'c', label='fit')
   
    # plot fit
    ax = plt.gca()
    ax.tick_params(right='off', top='off')
    ax.set_position([0.2, 0.2, 0.5, 0.75])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    plt.xlim(0, np.max(stds.index.tolist()))
    plt.ylim(0, ylim[-1])
    
    
    # fit 1/sqrt(n)
    plt.xlabel('number of tests')
    plt.ylabel('std of fit fmaxes in bin')
    
    # save fitting parameters
    params.add('median', value=np.average(tight_binders.fmax,
                                          weights=np.sqrt(tight_binders.number)))
    
    # also include fitParameters
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    
    # find fmin
    loose_binders = grouped_binders.loc[grouped_binders.dG > parameters.mindG]
    fitParameters.loc[:, 'fmin'] = getBoundsGivenDistribution(
            loose_binders.fmin, label='fmin'); plt.close()
    fitParameters.loc[:, 'fmax'] = getBoundsGivenDistribution(
            tight_binders.fmax, label='fmax'); plt.close()
    # find dG
    fitParameters.loc[:, 'dG'] = parameters.dGparam
    
    fitParameters.loc['vary'] = True
    fitParameters.loc['vary', 'fmin'] = False
    
    # also find default errors
    default_std_dev = grouped.std().loc[:, IMlibs.formatConcentrations(concentrations)].mean()
    parameters = fittingParameters(concentrations=concentrations,
                                   fitParameters=fitParameters,
                                   params=params,
                                   default_errors=default_std_dev
                                   )
    return parameters

def objectiveFunction(params, x, y=None, weights=None):
    parvals = params.valuesdict()
    sigma = parvals['sigma']
    c   = parvals['c']
    fit = sigma/np.sqrt(x) + c
    if y is None:
        return fit
    elif weights is None:
        return y-fit
    else:
        return (y - fit)*weights


if __name__=="__main__":    
    args = parser.parse_args()

    tmp, tmp, concentrations = IMlibs.loadMapFile(args.mapCPfluors)
    fitConstrained, fitParameters = fitFun.bindingSeriesByCluster(
            reducedCPsignalFile, concentrations, args.binding_point, numCores=numCores,
            backgroundTileFile=backgroundTileFile,
            filterPos=args.filterPos, filterNeg=args.filterNeg, num_clusters=Non)