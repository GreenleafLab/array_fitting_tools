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
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import lmfit
import itertools
import fitFun
import plotFun
import findFmaxDist
import singleClusterFits
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-t', '--single_cluster_fits', required=True, metavar=".CPfitted.pkl",
                   help='file with single cluster fits')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')
parser.add_argument('-b', '--binding_curves', required=True, metavar=".bindingSeries.pkl",
                   help='file containining the binding curve information')
parser.add_argument('-c', '--concentrations', required=True, metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')


group = parser.add_argument_group('additional option arguments')
group.add_argument('--n_samples', default=100, type=int, metavar="N",
                   help='number of times to bootstrap samples')
group.add_argument('--kd_cutoff', type=float,  
                   help='highest kd for tight binders (nM). default is 0.99 bound at '
                   'highest concentration')
group.add_argument('--use_simulated', type=int,
                   help='set to 0 or 1 if you want to use simulated distribution (1) or'
                   'not (0). Otherwise program will decide.')
group.add_argument('--enforce_fmax', type=int,  
                   help='set to 0 or 1 if you want to always enforce fmax (1) or'
                   'never enforce it (0). Otherwise the program will decide. ')
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores')
group.add_argument('--init', action="store_true", default=False,
                   help="flag if you just want to initiate fitting, not actually fit")
group.add_argument('--dont_load', action="store_true", default=False,
                   help="flag if you don't want to load data")

##### functions #####
def findVariantTable(table, parameter=None, name=None, concentrations=None):
    """ Find per-variant information from single cluster fits. """
    # define defaults
    if parameter is None: parameter = 'dG'
    if name is None:
        name = 'variant_number'   # default for lib 2
    
    # define columns as all the ones between variant number and fraction consensus
    test_stats = ['fmax', parameter, 'fmin']
    test_stats_init = ['%s_init'%param for param in ['fmax', parameter, 'fmin']]
    other_cols = ['numTests', 'fitFraction', 'pvalue', 'numClusters',
                  'fmax_lb','fmax', 'fmax_ub',
                  '%s_lb'%parameter, parameter, '%s_ub'%parameter,
                  'fmin', 'rsq', 'numIter', 'flag']
    
    table.dropna(axis=0, inplace=True)
    grouped = table.groupby('variant_number')
    variant_table = pd.DataFrame(index=grouped.first().index,
                                 columns=test_stats_init+other_cols)
    
    # filter for nan, barcode, and fit
    variant_table.loc[:, 'numTests'] = grouped.count().loc[:, parameter]
    
    fitFilteredTable = IMlibs.filterFitParameters(table)
    fitFilterGrouped = fitFilteredTable.groupby('variant_number')
    index = variant_table.loc[:, 'numTests'] > 0
    variant_table.loc[index, 'fitFraction'] = (fitFilterGrouped.count().loc[index, parameter]/
                                           variant_table.loc[index, 'numTests'])
    
    # then save parameters
    old_test_stats = grouped.median().loc[:, test_stats]
    old_test_stats.columns = test_stats_init
    variant_table.loc[:, test_stats_init] = old_test_stats
    
    # null model is that all the fits are bad. bad fits happen ~15% of the time
    #p = 1-variant_table.loc[(variant_table.numTests>=5)&(variant_table.dG < -10), 'fitFraction'].mean()
    p = 0.4
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*
             variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n].dropna().astype(float)
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, p)
    
    return variant_table


def findMaxProbability(x, numBins=None):
    """ Return the value of a distribution that maximizes the probability distribution.
    
    Also assume that the first bin isn't interesting. """
    
    # number of bins can be tuned. should think about having a kde estimation instead.
    if numBins is None:
        numBins = 200
    counts, binedges = np.histogram(x, bins=np.linspace(x.min(), x.max(), numBins))
    counts = counts[1:]; binedges=binedges[1:] # ignore first bin
    idx_max = np.argmax(counts)
    if idx_max != 0 and idx_max != len(counts)-1:
        return binedges[idx_max+1]
    else:
        return None


def getBoundsGivenDistribution(values, label=None, saturation_level=None,
                               use_max_prob=None):
    """ Get lowerbound, median, and upperbound of a distribution. """
    if saturation_level is None:
        saturation_level = 1 # i.e. assume max is max. 
    if use_max_prob is None:
        use_max_prob = False
    
    # return as a Series object with keys
    fitParameters = pd.Series(index=['lowerbound', 'initial', 'upperbound'])
    
    # remove the outliers from the distribution by MAD
    fDist = seqfun.remove_outlier(values)
    
    # lowerbound is the min
    fitParameters.loc['lowerbound'] = fDist.min()
    
    # upperbound is the max over some tunable scale factor 'saturation_level'
    fitParameters.loc['upperbound'] = fDist.max()/saturation_level
    
    # find the initial either by median or by maximum of prob distribution
    maxProb = findMaxProbability(fDist)
    if (maxProb is not None) and use_max_prob:
        fitParameters.loc['initial'] = maxProb
    else:
        fitParameters.loc['initial'] = fDist.median()
    
    # plot the initial point
    ax = plotFmaxMinDist(fDist, fitParameters);
    if label is not None:
        ax.set_xlabel(label)
    return fitParameters

def getInitialFitParameters(concentrations):
    """ Return initial fit parameters from single cluster fits.
    
    Add a row 'vary' that indicates whether parameter should vary or not. """
    fitParameters = singleClusterFits.getInitialFitParameters(concentrations)
    return pd.concat([fitParameters.astype(object),
                      pd.DataFrame(True, columns=fitParameters.columns,
                                         index=['vary'], dtype=bool)])


def perVariant(concentrations, subSeries, fitParameters, fmaxDistObject, initial_points=None,
               plot=None, n_samples=None, enforce_fmax=None):
    """ Fit a variant to objective function by bootstrapping median fluorescence. """
    
    # default id to not plot results
    if plot is None:
        plot = False

    # change initial guess on fit parameters if given previous fit
    fitParametersPer = fitParameters.copy()
    params_to_change = fitParameters.loc[:, fitParametersPer.loc['vary'].astype(bool)].columns
    if initial_points is not None:
        fitParametersPer.loc['initial', params_to_change] = (initial_points.loc[params_to_change])
    
    # find actual distribution of fmax given number of measurements
    fmaxDist = fmaxDistObject.getDist(len(subSeries))
    
    # fit variant
    results, singles = fitFun.bootstrapCurves(concentrations, subSeries, fitParameters,
                                              func=None,
                                              enforce_fmax=enforce_fmax,
                                              fmaxDist=fmaxDist,
                                              n_samples=n_samples,
                                              verbose=plot)
    
    # plot
    if plot:
        fitFun.plotFitCurve(concentrations,
                                     subSeries,
                                     results,
                                     fitParameters,
                                     log_axis=True)
        # also plot initial
        more_concentrations = np.logspace(0, 4)
        fit = fitFun.bindingCurveObjectiveFunction(fitFun.returnParamsFromResults(initial_points),
                                                   more_concentrations)
        plt.plot(more_concentrations, fit, 'k--')

    return results

def getMedianFirstBindingPoint(table):
    """ Return the median fluoresence in first binding point of each variant. """
    return table.groupby('variant_number').median().iloc[:, 0]

def loadGroupDict(fittedBindingFilename, annotatedClusterFile):
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

def returnCorrectedInitialPoints(initialPointsAll, fmin_fixed):
    """ Correct initial points by fmin. """
    initialPointsCorr = initialPointsAll.copy()
    initialPointsCorr.loc[:, 'fmax'] += (initialPointsAll.loc[:, 'fmin'] -
                                         fmin_fixed)
    return initialPointsCorr

def findInitialPoints(variant_table):
    """ Return initial points with different column names. """
    initialPoints = variant_table.loc[:, ['fmax_init', 'dG_init', 'fmin_init', 'numTests']]
    initialPoints.columns = ['fmax', 'dG', 'fmin', 'numTests']  
    return initialPoints

def returnFminFromFluorescence(initialPointsAll, fluorescenceMat, cutoff):
    """ Return the estimated fixed fmin based on affinity and fluroescence. """
    # if cutoff is not given, use parameters

    initial_dG = initialPointsAll.groupby('variant_number')['dG'].median()

    firstBindingPoint = getMedianFirstBindingPoint(fluorescenceMat)
    return firstBindingPoint.loc[initial_dG.index].loc[initial_dG > cutoff].median()

def findFmaxDistObject(variant_table, initialPointsCorr, affinity_cutoff, use_simulated=None):
    """ Find the fmax distribution. """
    parameters = fitFun.fittingParameters()
    kd_cutoff = parameters.find_Kd_from_dG(affinity_cutoff)
    index = variant_table.pvalue < 0.01
    plotFun.plotFmaxVsKd(variant_table, kd_cutoff, index)
    
    # if use_simulated is not given, decide
    if use_simulated is None:
        use_simulated = not findFmaxDist.useSimulatedOrActual(variant_table.loc[index], affinity_cutoff)
    if use_simulated:
        print 'Using fmaxes drawn randomly from clusters'
    else:
        print 'Using median fmaxes of variants'
    tight_binders = variant_table.loc[index&
                                      (variant_table.dG_init<affinity_cutoff)]
    fmaxDistObject = findFmaxDist.findParams(tight_binders,
                                       use_simulated=use_simulated,
                                       table=initialPointsCorr)
    return fmaxDistObject

def loadData(fittedBindingFilename, annotatedClusterFile,bindingCurveFilename):
    # load initial fitting parameters
    initialPointsAll = pd.concat([pd.read_pickle(annotatedClusterFile),
                                pd.read_pickle(fittedBindingFilename)], axis=1).astype(float)
    
    # load fluoresceence values
    fluorescenceMat, fluorescenceMatSplit = loadGroupDict(fittedBindingFilename, annotatedClusterFile)
    return initialPointsAll, fluorescenceMat, fluorescenceMatSplit

def initiateFitting(initialPointsAll, fluorescenceMat, fluorescenceMatSplit, concentrations,
                    kd_cutoff=None, use_simulated=None):
    
    # get parameters
    parameters = fitFun.fittingParameters(concentrations)
    if kd_cutoff is not None:
        # adjust cutoff to reflect this fraction bound at last concentration
        maxdG = parameters.find_dG_from_Kd(kd_cutoff)
        print ('Tight binding for kd less than %4.2f nM, or %4.1f%% bound at %4.2f nM'
               %(kd_cutoff, 100./(1+kd_cutoff/concentrations[-1]), concentrations[-1]))
    else:
        maxdG = parameters.maxdG
        print 'Tight binding for kd less than %4.2f nM'%parameters.find_Kd_from_dG(maxdG)

    # find fmin
    fmin_fixed = returnFminFromFluorescence(initialPointsAll, fluorescenceMat, parameters.mindG)
    
    # find variant_table
    initialPointsCorr = returnCorrectedInitialPoints(initialPointsAll, fmin_fixed)
    variant_table = findVariantTable(initialPointsCorr).astype(float)
    
    # find fmax Dist
    fmaxDistObject = findFmaxDistObject(variant_table, initialPointsCorr, maxdG,
                                        use_simulated=use_simulated)
    
    # find constraints on fmin and delta G
    fitParameters = getInitialFitParameters(concentrations)
    fitParameters.loc['vary', 'fmin'] = False
    fitParameters.loc['initial', 'fmin'] = fmin_fixed
    fitParameters.loc['initial', 'fmax'] = fmaxDistObject.getDist(1).stats(moments='m')
        
    # make sure initial points have all of keys that table does
    initialPoints = findInitialPoints(variant_table)
    missing_variants = (np.array(fluorescenceMatSplit.keys())
                        [np.logical_not(np.in1d(fluorescenceMatSplit.keys(), initialPoints.index))])
    initialPoints = pd.concat([initialPoints, pd.DataFrame(index=missing_variants,
                                                           columns=initialPoints.columns)])
    return variant_table, initialPoints, fmaxDistObject, fitParameters

def fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                     bindingCurveFilename, concentrations,
                     numCores=None, n_samples=None, variants=None,
                     use_initial=None, enforce_fmax=None, kd_cutoff=None, use_simulated=None,
                     figDirectory=None):
    if numCores is None:
        numCores = 20
    if use_initial is None:
        use_initial = False
    
    # load data
    initialPointsAll, fluorescenceMat, fluorescenceMatSplit = loadData(
        fittedBindingFilename, annotatedClusterFile,bindingCurveFilename)
    
    # process before fitting
    variant_table, initialPoints, fmaxDistObject, fitParameters = initiateFitting(
        initialPointsAll, fluorescenceMat, fluorescenceMatSplit, concentrations,
        kd_cutoff=kd_cutoff, use_simulated=use_simulated)
    
    # save figs
    if figDirectory is not None:
        plt.savefig(os.path.join(figDirectory, 'fmax_stde_vs_n.pdf')); plt.close()
        plt.savefig(os.path.join(figDirectory, 'fmax_passing_pvalue_and_fit.pdf')); plt.close()
        plt.savefig(os.path.join(figDirectory, 'fmax_vs_Kd_init.pdf')); plt.close()
    
    if variants is None:
        variants = fluorescenceMatSplit.keys()
    
    print '\tMultiprocessing bootstrapping...'
    if use_initial:
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
    else:
        # parallelize fitting
        results = (Parallel(n_jobs=numCores, verbose=10)
                    (delayed(perVariant)(concentrations,
                                            fluorescenceMatSplit[variant],
                                            fitParameters,
                                            fmaxDistObject,
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
    
    fittedBindingFilename = args.single_cluster_fits
    annotatedClusterFile  = args.annotated_clusters
    bindingCurveFilename  = args.binding_curves
    n_samples = args.n_samples
    numCores = args.numCores
    outFile  = args.out_file
    kd_cutoff = args.kd_cutoff
    enforceFmax = args.enforce_fmax
    use_simulated = args.use_simulated
    concentrations = np.loadtxt(args.concentrations)
    
    # find out file
    if outFile is None:
        outFile = os.path.splitext(
            fittedBindingFilename[:fittedBindingFilename.find('.pkl')])[0]

    # make fig directory    
    figDirectory = os.path.join(os.path.dirname(annotatedClusterFile),
                                'figs_%s'%str(datetime.date.today()))
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    
    # fit
    if not args.init:
        variant_table = fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                         bindingCurveFilename, concentrations,
                         numCores=numCores, n_samples=n_samples,
                         use_initial=True, enforce_fmax=enforceFmax,
                         kd_cutoff=kd_cutoff, use_simulated=use_simulated,
                         figDirectory=figDirectory)
    
    
        variant_table.to_csv(outFile + '.CPvariant', sep='\t', index=True)
            
        # make plots    
        plotFun.plotFmaxInit(variant_table)
        plt.savefig(os.path.join(figDirectory, 'initial_Kd_vs_final.colored_by_fmax.pdf'))
        
        plotFun.plotErrorInBins(variant_table, xdelta=10)
        plt.savefig(os.path.join(figDirectory, 'error_in_bins.dG.pdf'))
        
        plotFun.plotNumberInBins(variant_table, xdelta=10)
        plt.savefig(os.path.join(figDirectory, 'number_in_bins.Kd.pdf'))
        sys.exit()
    
    else:
        # load data
        if not args.dont_load:
            initialPointsAll, fluorescenceMat, fluorescenceMatSplit = loadData(
                fittedBindingFilename, annotatedClusterFile,bindingCurveFilename)
        
            # process before fitting
            variant_table, initialPoints, fmaxDistObject, fitParameters = initiateFitting(
                initialPointsAll, fluorescenceMat, fluorescenceMatSplit, concentrations,
                kd_cutoff=kd_cutoff, use_simulated=use_simulated)
        sys.exit()
    
    # other stuff you can do
    parameters = fitFun.fittingParameters()
    numPointsLost = 3
    maxdG = parameters.find_dG_from_Kd(concentrations[-3])
    variants = variant_table.loc[(variant_table.numTests >= 5)&
                                 (variant_table.dG <= maxdG)].sort('dG').index[::30]
    
    results = pd.concat((Parallel(n_jobs=numCores, verbose=10)
                        (delayed(perVariant)(concentrations[:-numPointsLost],
                                                groupDict[variant].iloc[:,:-numPointsLost],
                                                fitParameters,
                                                fmaxDist,
                                                initialPoints.loc[variant],
                                                n_samples=n_samples)
                         for variant in variants if variant in groupDict.keys())), axis=1).transpose();
    results.index = [variant for variant in variants if variant in groupDict.keys()]
    plotFun.plotScatterPlotColoredByFlag(variant_table.loc[results.index], results,
                                         concentrations, numPointsLost)
    plt.savefig(os.path.join(figDirectory, ('correlation_Kd_%dconcentrations.pdf')
                             %(len(concentrations)-numPointsLost)))
    
    # plot single variants
    index = ((results2.loc[variants].flag == 1)&(variant_table.loc[variants].flag==0))
    i=0
    variant = variants[index.values][i]
    fitFun.plotFitCurve(concentrations,
                        groupDict[variant],
                        variant_table.loc[variant],
                        fitParameters)
    fitFun.plotFitCurve(concentrations[:-numPointsLost],
                        groupDict[variant].iloc[:, :-numPointsLost],
                        results2.loc[variant],
                        fitParameters)
    
    ids = libChar.loc[variants].length.astype(int).astype(str) + '_' +libChar.loc[variants].helix_one_length.astype(int).astype(str)
    seq = libChar.loc[variants].junction_seq.astype(str)
    for variant in variants:
        plotFun.plotNormalizedFitCurve(concentrations, groupDict[variant],
                                       variant_table.loc[variant], fitParameters)
        plt.savefig(os.path.join(figDirectory, 'binding_curve.variant_%d.seq_%s.%s.pdf'%(variant, seq.loc[variant], ids.loc[variant])))
