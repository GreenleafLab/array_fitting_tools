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
import itertools
import fitFun

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-t', '--single_cluster_fits', required=True,
                   help='file with single cluster fits')
parser.add_argument('-a', '--annotated_clusters', required=True,
                   help='file with clusters annotated by variant number')
parser.add_argument('-b', '--binding_curves', required=True,
                   help='file containining the binding curve information')
parser.add_argument('-c', '--concentrations', required=True,
                    help='text file giving the associated concentrations')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')


group = parser.add_argument_group('additional option arguments')
group.add_argument('--n_samples', default=100, type=int,
                   help='number of times to bootstrap samples')
group.add_argument('--not_pickled', default=False, action="store_true",
                   help='program assumes inputs are pickled. Flag if text files')
group.add_argument('-n', '--numCores', default=20, type=int,
                   help='number of cores')


##### functions #####
def findVariantTable(table, parameter=None, name=None, concentrations=None):
    # define defaults
    if parameter is None: parameter = 'dG'
    if name is None:
        name = 'variant_number'   # default for lib 2
    
    # define columns as all the ones between variant number and fraction consensus
    test_stats = ['fmax', parameter, 'fmin']
    test_stats_init = ['%s_init'%param for param in ['fmax', parameter, 'fmin']]
    other_cols = ['numTests', 'numRejects', 'fitFraction', 'numClusters',
                  'pvalue', 'fmax_lb','fmax', 'fmax_ub',
                  '%s_lb'%parameter, parameter, '%s_ub'%parameter,
                  'fmin', 'rsq', 'numIter', 'fractionOutlier', 'flag']
    
    table.dropna(axis=0, inplace=True)
    grouped = table.groupby('variant_number')
    variant_table = pd.DataFrame(index=grouped.first().index,
                                 columns=test_stats_init+other_cols)
    
    # filter for nan, barcode, and fit
    variant_table.loc[:, 'numTests'] = grouped.count().loc[:, parameter]
    variant_table.loc[:, 'numRejects'] = 0
    
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



def matchTogetherResults(variant_table, results):
    columns = results.columns[np.in1d(results.columns, variant_table.columns)]
    x = variant_table.copy()
    x.loc[results.index, columns] = results.loc[:, columns]
    return x




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
                
        
def getInitialParameters(initial_points, concentrations):
    parameters = fitFun.fittingParameters(concentrations=concentrations)
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    # find fmin
    loose_binders = initial_points.loc[initial_points.dG > parameters.mindG]
    
    fitParameters.loc[:, 'fmin'] = fitFun.getBoundsGivenDistribution(
            loose_binders.fmin, label='fmin'); plt.close()

    # find dG
    fitParameters.loc[:, 'dG'] = [parameters.find_dG_from_Kd(
                parameters.find_Kd_from_frac_bound_concentration(frac_bound, concentration))
                                  for frac_bound, concentration in itertools.izip(
                                    [0.99, 0.5, 0.01],
                                    [concentrations[0], concentrations[-1], concentrations[-1]])]
                                  

    
    fitParameters.loc['vary'] = True
    fitParameters.loc['vary', 'fmin'] = False
    return fitParameters

def perVariant(concentrations, subSeries, fitParameters, initial_points, fmaxDist,
               plot=None, n_samples=None):
    if plot is None:
        plot = False

    fitParameters = fitParameters.copy()
    rows_to_change = ['lowerbound', 'initial', 'upperbound']
    cols_to_change = fitParameters.loc['vary'].astype(bool)
    fitParameters.loc[rows_to_change, 'fmax'] = fmaxDist.find_fmax_bounds_given_n(len(subSeries))
    fitParameters.loc['initial', cols_to_change] = (
        initial_points.loc[fitParameters.loc[:, cols_to_change].columns])
    results, singles = fitFun.bootstrapCurves(concentrations, subSeries, fitParameters,
                                              func=None, enforce_fmax=True,
                                              fmaxDist=
                                              fmaxDist.find_fmax_bounds_given_n(len(subSeries),
                                                       return_dist=True),
                                              n_samples=n_samples)
    if plot:
        fitFun.plotFitCurve(concentrations,
                                     subSeries,
                                     results,
                                     fitParameters,
                                     log_axis=True)
    return results

def fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                     bindingCurveFilename, concentrations,
                     numCores=None, n_samples=None):
    if numCores is None:
        numCores = 20
    
    # load initial points and find fitParameters
    initialPointsAll = pd.concat([pd.read_pickle(annotatedClusterFile),
                                pd.read_pickle(fittedBindingFilename)], axis=1).astype(float)
    
    # find constraints on fmin and delta G
    fitParameters = getInitialParameters(initialPointsAll, concentrations)
    
    # correct fmax measuremenets by fmin because likely variants that went to
    # saturation had artifactually high fit fmins.
    initialPointsAll.loc[:, 'fmax'] = (initialPointsAll.fmax + initialPointsAll.fmin -
                                        fitParameters.loc['initial', 'fmin'])
    
    variant_table = findVariantTable(initialPointsAll).astype(float)
    initialPoints = variant_table.loc[:, ['fmax_init', 'dG_init', 'fmin_init', 'numTests']]
    initialPoints.columns = ['fmax', 'dG', 'fmin', 'numTests']
    
    # only use those clusters corresponding to variants that pass fit fraction cutff
    if fitFun.useSimulatedOrActual(variant_table):
        print 'Using median fmaxes of variants to measure stderr'
        fmaxDist = fitFun.findFinalBoundsParameters(variant_table.loc[
            variant_table.pvalue < 0.01], concentrations)
        x, y = fitFun.findFinalBoundsParametersSimulated(
            variant_table.loc[variant_table.pvalue < 0.01],
                                                             initialPointsAll,
                                                             concentrations,
                                                             return_vals=True)
        plt.plot(x, y, 'r:', label='simulated')
        plt.legend()
    else:
        print ('Using median fmaxes of subsampled clusters to measure stderr'
               'Assuming stde is maximum of fit and std of all variants')
        # simulate relationship as well
        fmaxDist = fitFun.findFinalBoundsParametersSimulated(
            variant_table.loc[variant_table.pvalue < 0.01],
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
    
    print '\tMultiprocessing bootstrapping...'
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(perVariant)(concentrations,
                                        groupDict[variant],
                                        fitParameters,
                                        initialPoints.loc[variant],
                                        fmaxDist)
                 for variant in groupDict.keys()))
    results = pd.concat(results)

    variant_table_final = matchTogetherResults(variant_table, results)
    
    return variant_table_final




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
    
    concentrations = np.loadtxt(args.concentrations)
    sys.exit()
    variant_table = fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                     bindingCurveFilename, concentrations,
                     numCores=numCores, n_samples=n_samples)


    IMlibs.saveDataFrame(variant_table, args.out_file,
                         float_format='%4.3f', index=True)
    
    sys.exit()
    
    # plot
    figDirectory = ps.path.join(os.path.dirname(args.out_file), 'figs_%s'%str(datetime.date.today()))
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    
    subFigDirectory = os.path.join(figDirectory, 'binding_curves')
    if not os.path.exists(subFigDirectory):
        os.mkdir(subFigDirectory)
        
    for variant in variant_table.index:
        bootStrapFits.plotSingleVariantFits(table, variant_table, variant, concentrations)
        plt.savefig(os.path.join(subFigDirectory, 'binding_curve.variant_%d.pdf'%variant))
        plt.close()
        
    # plot error
    plt.figure(figsize=(4,3));
    sns.distplot(error.loc[(variant_table.dG < -7)&(variant_table.flag=='0')],
                 bins=np.arange(0, 2, 0.1), kde=False, label='method A');
    sns.distplot(error.loc[(variant_table.dG < -7)&(variant_table.flag=='1')],
                 bins=np.arange(0, 2, 0.1), label='method B', kde=False);
    plt.xlabel('width of confidence interval (kcal/mol)');
    plt.ylabel('count');
    plt.tight_layout();
    ax=plt.gca(); ax.tick_params(right='off', top='off');
    plt.legend()
