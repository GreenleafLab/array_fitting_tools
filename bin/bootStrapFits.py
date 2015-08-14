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
group.add_argument('-n', '--numCores', default=20, type=int, metavar="N",
                   help='number of cores')
group.add_argument('--init', action="store_true", default=False,
                   help="flag if you just want to initiate fitting, not actually fit")


##### functions #####
def findVariantTable(table, parameter=None, name=None, concentrations=None):
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
    p = 0.25
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*
             variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n].dropna().astype(float)
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, p)
    
    return variant_table

def plotSingleVariantFits(table, results, variant, concentrations, plot_init=None,
                          annotate=None):
    if annotate:
        subresults = results.loc[variant]
        if int(subresults.flag) == 0:
            fitting_method = 'A'
        else:
            fitting_method = 'B'
            
        annotateText = ('variant %d\n' +
                        'fitting method %s\n' +
                        '$\Delta$G = %4.2f (%4.2f, %4.2f)\n' +
                        '%d measurements')%(
            variant, fitting_method,
            subresults.dG, subresults.dG_lb, subresults.dG_ub, subresults.numClusters)
        plt.annotate(annotateText, xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top',
                    fontsize=12)
                
        
def getInitialFitParameters(concentrations, initial_points=None):
    parameters = fitFun.fittingParameters(concentrations=concentrations)
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    # find dG
    fitParameters.loc[:, 'dG'] = [parameters.find_dG_from_Kd(
                parameters.find_Kd_from_frac_bound_concentration(frac_bound, concentration))
                                  for frac_bound, concentration in itertools.izip(
                                    [0.99, 0.5, 0.01],
                                    [concentrations[0], concentrations[-1], concentrations[-1]])]
    if initial_points is not None:
        # find fmin
        loose_binders = initial_points.loc[initial_points.dG > parameters.mindG]
        
        fitParameters.loc[:, 'fmin'] = fitFun.getBoundsGivenDistribution(
                loose_binders.fmin, label='fmin'); plt.close()

        fitParameters.loc['vary'] = True
        fitParameters.loc['vary', 'fmin'] = False
    return fitParameters

def perVariant(concentrations, subSeries, fitParameters, fmaxDist, initial_points=None,
               plot=None, n_samples=None):
    if plot is None:
        plot = False

    fitParameters = fitParameters.copy()
    rows_to_change = ['lowerbound', 'initial', 'upperbound']
    cols_to_change = fitParameters.loc['vary'].astype(bool)
    
    # change bound of fmax given fmaxDist and number of measurements
    fitParameters.loc[rows_to_change, 'fmax'] = fmaxDist.find_fmax_bounds_given_n(len(subSeries))
    if initial_points is not None:
        fitParameters.loc['initial', cols_to_change] = (initial_points.loc[
            cols_to_change.loc[cols_to_change].index])
    results, singles = fitFun.bootstrapCurves(concentrations, subSeries, fitParameters,
                                              func=None, enforce_fmax=True,
                                              fmaxDist=
                                              fmaxDist.find_fmax_bounds_given_n(len(subSeries),
                                                       return_dist=True),
                                              n_samples=n_samples,
                                              verbose=plot)
    if plot:
        fitFun.plotFitCurve(concentrations,
                                     subSeries,
                                     results,
                                     fitParameters,
                                     log_axis=True)
    return results

def initiateFitting(fittedBindingFilename, annotatedClusterFile,
                     bindingCurveFilename, concentrations ):

    # load initial points and find fitParameters
    initialPointsAll = pd.concat([pd.read_pickle(annotatedClusterFile),
                                pd.read_pickle(fittedBindingFilename)], axis=1).astype(float)
    
    # find constraints on fmin and delta G
    fitParameters = getInitialFitParameters(concentrations, initialPointsAll)
        
    # find constraints on fmax 
    variant_table = findVariantTable(initialPointsAll).astype(float)
    initialPoints = variant_table.loc[:, ['fmax_init', 'dG_init', 'fmin_init', 'numTests']]
    initialPoints.columns = ['fmax', 'dG', 'fmin', 'numTests']
    
    # only use those clusters corresponding to variants that pass fit fraction cutff
    index = variant_table.pvalue < 0.01
    plotFun.plotFmaxVsKd(variant_table, concentrations, index)
    
    if fitFun.useSimulatedOrActual(variant_table.loc[index], concentrations):
        print 'Using median fmaxes of variants to measure stderr'
        fmaxDist = fitFun.findFinalBoundsParameters(
            variant_table.loc[index], concentrations)
        x, y = fitFun.findFinalBoundsParametersSimulated(
            variant_table.loc[index],
                                    initialPointsAll,
                                    concentrations,
                                    return_vals=True)
        plt.plot(x, y, 'r:', label='simulated')
        plt.legend()
    else:
        print ('Using median fmaxes of subsampled clusters to measure stderr.\n'
               'Assuming stde is maximum of [fit std, std of all variants]')
        # simulate relationship as well
        fmaxDist = fitFun.findFinalBoundsParametersSimulated(
            variant_table.loc[index],
                                    initialPointsAll,
                                    concentrations)
    
    # load binding series information with variant numbers
    table = (pd.concat([pd.read_pickle(annotatedClusterFile),
                       pd.read_pickle(bindingCurveFilename).astype(float)], axis=1).
                sort('variant_number'))

    # fit all labeled variants
    table.dropna(axis=0, subset=['variant_number'], inplace=True)

    # fit only clusters that are not all NaN
    table.dropna(axis=0, subset=table.columns[1:], how='all',inplace=True)
    
    print '\tDividing table into groups...'
    groupDict = {}
    for name, group in table.groupby('variant_number'):
        groupDict[name] = group.iloc[:, 1:]
        
    # make sure initial points have all of keys that table does
    missing_variants = (np.array(groupDict.keys())
                        [np.logical_not(np.in1d(groupDict.keys(), initialPoints.index))])
    initialPoints = pd.concat([initialPoints, pd.DataFrame(index=missing_variants,
                                                           columns=initialPoints.columns)])
    return groupDict, initialPoints, variant_table, fmaxDist, fitParameters

def fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                     bindingCurveFilename, concentrations,
                     numCores=None, n_samples=None, variants=None,
                     use_initial=None):
    if numCores is None:
        numCores = 20
    if use_initial is None:
        use_initial = False
    
    groupDict, initialPoints, variant_table, fmaxDist, fitParameters = initiateFitting(
        fittedBindingFilename, annotatedClusterFile, bindingCurveFilename, concentrations)
    
    if variants is None:
        variants = groupDict.keys()
    
    print '\tMultiprocessing bootstrapping...'
    if use_initial:
        results = (Parallel(n_jobs=numCores, verbose=10)
                    (delayed(perVariant)(concentrations,
                                            groupDict[variant],
                                            fitParameters,
                                            fmaxDist,
                                            initialPoints.loc[variant],
                                            n_samples=n_samples)
                     for variant in variants if variant in groupDict.keys()))
    else:
        results = (Parallel(n_jobs=numCores, verbose=10)
                    (delayed(perVariant)(concentrations,
                                            groupDict[variant],
                                            fitParameters,
                                            fmaxDist,
                                            n_samples=n_samples)
                     for variant in variants if variant in groupDict.keys()))        
    results = pd.concat(results, axis=1).transpose()
    results.index = [variant for variant in variants if variant in groupDict.keys()]

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
    concentrations = np.loadtxt(args.concentrations)

    # find out file
    if outFile is None:
        outFile = os.path.splitext(
            annotatedClusterFile[:annotatedClusterFile.find('.pkl')])[0]
    
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
                         use_initial=True)
    
    
        variant_table.to_csv(outFile + '.CPvariant', sep='\t', index=True)
            
        # make plots
        plt.savefig(os.path.join(figDirectory, 'fmax_vs_Kd_init.pdf')); plt.close()
        plt.savefig(os.path.join(figDirectory, 'fmax_stde_vs_n.pdf'))
        
        plotFun.plotFmaxInit(variant_table)
        plt.savefig(os.path.join(figDirectory, 'initial_Kd_vs_final.colored_by_fmax.pdf'))
        
        plotFun.plotErrorInBins(variant_table, xdelta=10)
        plt.savefig(os.path.join(figDirectory, 'error_in_bins.dG.pdf'))
        
        plotFun.plotPercentErrorInBins(variant_table, xdelta=10)
        plt.savefig(os.path.join(figDirectory, 'error_in_bins.Kd.pdf'))
        
        plotFun.plotNumberInBins(variant_table, xdelta=10)
        plt.savefig(os.path.join(figDirectory, 'number_in_bins.Kd.pdf'))
        sys.exit()
    
    else:
        # subtract binding points
        groupDict, initialPoints, variant_table, fmaxDist, fitParameters = initiateFitting(
            fittedBindingFilename, annotatedClusterFile, bindingCurveFilename, concentrations)
        
        variant_table = pd.read_table(outFile + '.CPvariant', index_col=0)
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
    
    
