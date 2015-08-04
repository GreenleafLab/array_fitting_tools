#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import argparse
import sys
import seqfun
import itertools
import scipy.stats as st
import IMlibs
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
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
group.add_argument('-cs', '--cpsignal', metavar="CPsignal.pkl", required=True,
                    help='reduced CPsignal file.')
group.add_argument('-c', '--concentrations', required=True, metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')

group = parser.add_argument_group('initial constraints')
group.add_argument('-ft', '--fit_parameters',
                    help='fitParameters file. If file is given, use these '
                    'upperbound/lowerbounds')

group = parser.add_argument_group('other ways to find initial constraints')
group.add_argument('-bg','--background_fluor_file', metavar="fluor.npy",
                   help='file containing '
                   'fluorescent values of "background" clusters.') 
group.add_argument('-bp', '--binding_point', type=int, default=-1,
                    help='point in binding series to compare to null scores (Default is '
                    'last concentration. -2 for second to last concentration)', )
group.add_argument('--n_clusters', type=int,
                    help='if background file not provided, number of clusters to'
                    'assume are binders' )


group = parser.add_argument_group('other settings')
group.add_argument('-n','--numCores', type=int, default=20,
                    help='maximum number of cores to use. default=20')



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
    ax1.hist(yvalues.values, bins=binedges, histtype='stepfilled',
              normed=True, color='grey',
             alpha=0.5, label='all',)
    
    if indexBinders is not None:
        #ax1 = ax1.twinx()
        ax1.hist(yvalues.loc[indexBinders].values, bins=binedges,
                 histtype='stepfilled', normed=True,
                 color='red', alpha=0.5, label='binder',)
    if indexNonBinders is not None:
        ax1.hist(yvalues.loc[indexNonBinders].values, bins=binedges,
                 histtype='stepfilled', normed=True,
                 color='blue', alpha=0.5, label='nonbinder',)
    plt.legend(loc='upper right')
    plt.xlabel('fluorescence')
    plt.ylabel('probabililty')
    plt.tight_layout()
    ax1.tick_params(right='off', top='off')
    return

def getInitialFitParameters(concentrations):

    parameters = fittingParameters(concentrations=concentrations)
    
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    
    # find fmin
    param = 'fmin'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]

    # find fmax
    param = 'fmax'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    
    # find dG
    fitParameters.loc[:, 'dG'] = [parameters.find_dG_from_Kd(
        parameters.find_Kd_from_frac_bound_concentration(frac_bound, concentration))
             for frac_bound, concentration in itertools.izip(
                                    [0.99, 0.5, 0.01],
                                    [concentrations[0], concentrations[-1], concentrations[-1]])]
 
    return fitParameters


def splitAndFit(bindingSeries, concentrations, fitParameters, numCores,
                index=None, change_params=None):
    
    if index is None:
        index = bindingSeries.index
    
    # split into parts
    print 'Splitting clusters into %d groups:'%numCores
    # assume that list is sorted somehow
    indicesSplit = [index[i::numCores] for i in range(numCores)]
    bindingSeriesSplit = [bindingSeries.loc[indices] for indices in indicesSplit]
    printBools = [True] + [False]*(numCores-1)
    print 'Fitting binding curves:'
    fits = (Parallel(n_jobs=numCores, verbose=10)
            (delayed(fitSetClusters)(concentrations, subBindingSeries,
                                     fitParameters, print_bool, change_params)
             for subBindingSeries, print_bool in itertools.izip(bindingSeriesSplit, printBools)))

    return pd.concat(fits)

def fitSetClusters(concentrations, subBindingSeries, fitParameters, print_bool=None,
                   change_params=None):
    if print_bool is None: print_bool = True

    #print print_bool
    singles = []
    for i, idx in enumerate(subBindingSeries.index):
        if print_bool:
            num_steps = max(min(100, (int(len(subBindingSeries)/100.))), 1)
            if (i+1)%num_steps == 0:
                print ('working on %d out of %d iterations (%d%%)'
                       %(i+1, len(subBindingSeries.index), 100*(i+1)/
                         float(len(subBindingSeries.index))))
                sys.stdout.flush()
        fluorescence = subBindingSeries.loc[idx]
        singles.append(perCluster(concentrations, fluorescence, fitParameters,
                                  change_params=change_params))

    return pd.concat(singles)

def perCluster(concentrations, fluorescence, fitParameters, plot=None, change_params=None):
    if plot is None:
        plot = False
    if change_params is None:
        change_params = False
    try:
        if change_params:
            a, b = np.percentile(fluorescence.dropna(), [0, 100])
            fitParameters = fitParameters.copy()
            fitParameters.loc['initial', ['fmin', 'fmax']] = [a, b-a]
        
        single = fitFun.fitSingleCurve(concentrations,
                                                       fluorescence,
                                                       fitParameters)
    except:
        print 'Error with %s'%fluorescence.name
        single = fitFun.fitSingleCurve(concentrations,
                                                       fluorescence,
                                                       fitParameters,
                                                       do_not_fit=True)
    if plot:
        fitFun.plotFitCurve(concentrations, fluorescence, single, fitParameters) 
              
    return pd.DataFrame(columns=[fluorescence.name],
                        data=single).transpose()


# define functions
def bindingSeriesByCluster(concentrations, bindingSeries, bindingSeriesNorm, 
                           numCores=None,  subset=None):
    if subset is None:
        subset = False

    # find initial parameters
    fitParameters = getInitialFitParameters(concentrations)

    # sort by fluorescence in null_column to try to get groups of equal
    # distributions of binders/nonbinders
    fluorescence = bindingSeries.iloc[:, -1].copy()
    fluorescence.sort()
    index_all = bindingSeries.loc[fluorescence.index].dropna(axis=0, thresh=4).index

    if subset:
        num = 5E3
        index_all = index_all[np.linspace(0, len(index_all)-1, num).astype(int)]
    
    fitResults = pd.DataFrame(index=bindingSeries.index,
                              columns=fitFun.fitSingleCurve(concentrations,
                                                            None, fitParameters,
                                                            do_not_fit=True).index)
    fitResults.loc[index_all] = splitAndFit(bindingSeries, concentrations,
                                        fitParameters, numCores, index=index_all,
                                        change_params=True)


    return fitResults 
    



if __name__=="__main__":    
    args = parser.parse_args()
    
    pickleCPsignalFilename = args.cpsignal
    outFile               = args.out_file
    numCores = args.numCores
    concentrations = np.loadtxt(args.concentrations)
    fitParametersFilename = args.fit_parameters
    backgroundFilename = args.background_fluor_file
    binding_point = args.binding_point
    #  check proper inputs
    if outFile is None:
        
        # make bindingCurve file 
        outFile = os.path.splitext(
                pickleCPsignalFilename[:pickleCPsignalFilename.find('.pkl')])[0]
        
    bindingCurveFilename = outFile + '.bindingCurve.pkl'
    # get binding series
    print '\tLoading binding series and all RNA signal:'
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
        pickleCPsignalFilename, concentrations=concentrations)
    
    # make normalized binding series
    allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # save
    bindingSeriesNorm.to_pickle(bindingCurveFilename)
    
    # load or find fitParameters
    if fitParametersFilename is not None:
        print 'Using given fit parameters.. %s'%fitParametersFilename
        fitParameters = pd.read_table(fitParametersFilename, index_col=0)
    
    else:
        if backgroundFilename is not None:
            # load null columns
            null_scores = np.load(backgroundFilename)
    
            # estimate binders and nonbinders
            qvalue_cutoff = 0.005
            print ('Finding constraints on fmax using null distribution %s,\n'
                   'binding point %d,\n'
                   'and qvalue cutoff < %g')%(backgroundFilename, binding_point, qvalue_cutoff)
    
            indexBinders = getEstimatedBinders(bindingSeries, binding_point,
                                                  null_scores=null_scores,
                                                  qvalue_cutoff=qvalue_cutoff)
        elif args.n_clusters is not None:
            print ('Using top %d clusters to estimate fmax cutoff')%args.n_clusters
            indexBinders = (bindingSeriesNorm.sort(
                bindingSeriesNorm.columns[args.binding_point]).
                            iloc[:args.n_clusters].index)
        else:
            print ('Error: must define either fitParameters file, background file, '
                   'or number of clusters')
            sys.exit()
        
        # plot and initiate fit
        plotInitialEstimates(bindingSeriesNorm, binding_point, indexBinders)
        fitParameters = getInitialFitParameters(concentrations)

        maxInitialBinders = 1E4
        if maxInitialBinders < len(indexBinders):
            index = indexBinders[np.linspace(0, len(indexBinders)-1,
                                             maxInitialBinders).astype(int)]
        else:
            index = indexBinders
        fitUnconstrained =  splitAndFit(bindingSeriesNorm, concentrations,
                                        fitParameters, numCores, index=index,
                                        change_params=True)
        fitParameters.loc[:, 'fmax'] = fitFun.getBoundsGivenDistribution(
            fitUnconstrained.fmax)
        fitParameters.loc[:, 'fmin'] = fitFun.getBoundsGivenDistribution(
            fitUnconstrained.fmin)        
        
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
    fluorescence = bindingSeries.iloc[:, -1].copy()
    fluorescence.sort()
    index_all = bindingSeries.loc[fluorescence.index].dropna(axis=0, thresh=4).index
    
    fitResults = pd.DataFrame(index=bindingSeriesNorm.index,
                              columns=fitFun.fitSingleCurve(concentrations,
                                                            None, fitParameters,
                                                            do_not_fit=True).index)
    sys.exit()
    fitResults.loc[index_all] = splitAndFit(bindingSeries, concentrations,
                                        fitParameters, numCores, index=index_all)
    
    fitParametersFilename = args.fit_parameters