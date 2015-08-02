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


##### functions #####

        
def findInitialBounds():
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
        
def findVariantTable(table, parameter=None, name=None, concentrations=None):
    # define defaults
    if parameter is None: parameter = 'dG'
    if name is None:
        name = 'variant_number'   # default for lib 2
    
    # define columns as all the ones between variant number and fraction consensus
    test_stats = ['fmax', parameter, 'fmin',  'qvalue']
    test_stats_init = ['%s_init'%param for param in ['fmax', parameter, 'fmin']] + ['qvalue']
    other_cols = ['numTests', 'numRejects', 'fitFraction', 'numClusters',
                  'pvalue', 'fmax_lb','fmax', 'fmax_ub',
                  '%s_lb'%parameter, parameter, '%s_ub'%parameter,
                  'fmin', 'rsq', 'numIter', 'fractionOutlier', 'flag']
    
    grouped = table.groupby('variant_number')
    variant_table = pd.DataFrame(index=grouped.first().index,
                                 columns=test_stats_init+other_cols)
    
    # filter for nan, barcode, and fit
    filteredTable = IMlibs.filterStandardParameters(table)
    firstFilterGrouped = filteredTable.groupby('variant_number')
    variant_table.loc[:, 'numTests'] = firstFilterGrouped.count().loc[:, parameter]
    variant_table.loc[:, 'numRejects'] = (grouped.count().loc[:, parameter] -
                                          variant_table.loc[:, 'numTests'])
    
    fitFilteredTable = IMlibs.filterFitParameters(filteredTable)
    fitFilterGrouped = fitFilteredTable.groupby('variant_number')
    index = variant_table.loc[:, 'numTests'] > 0
    variant_table.loc[index, 'fitFraction'] = (fitFilterGrouped.count().loc[index, parameter]/
                                           variant_table.loc[index, 'numTests'])
    
    # then save parameters
    old_test_stats = fitFilterGrouped[test_stats].median()
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


def getBootstrappedErrors(table,  parameters, numCores, parameter=None,
                          variants=None, n_samples=None):
    if parameter is None: parameter = 'dG'
    if variants is None:
        variants = np.unique(table.variant_number.dropna())
    
    concentrations = parameters.concentrations
    concentrationCols = IMlibs.formatConcentrations(concentrations)
    param_names = parameters.fitParameters.dropna(axis=1).columns.tolist()
    subtable = (IMlibs.filterStandardParameters(table).
                loc[:, ['variant_number'] + concentrationCols + param_names])
    grouped = subtable.groupby('variant_number')
    
    print '\tDividing table into groups...'
    actual_variants = []
    groups = []
    for name, group in grouped:
        if name in variants:
            actual_variants.append(name)
            groups.append(group)
        
    print '\tMultiprocessing bootstrapping...'
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(perVariantFit)(group.loc[:, concentrationCols],
                                        concentrations,
                                        parameters,
                                        group.loc[:, param_names].median(),
                                        n_samples=n_samples)
                 for group in groups))
    results = pd.concat(results)
    results.index = actual_variants
    return results

def fitSingleVariant(table, variant, parameters):
    # format inputs for perVariantFit
    concentrationCols = IMlibs.formatConcentrations(parameters.concentrations)
    param_names = parameters.fitParameters.dropna(axis=1).columns.tolist()
    subtable = (IMlibs.filterStandardParameters(table, parameters.concentrations).
                loc[:, ['variant_number'] + concentrationCols + param_names])
    grouped = subtable.groupby('variant_number')
    
    initial_points = grouped.median().loc[variant, param_names]
    results = perVariantFit(subtable.loc[subtable.variant_number==variant, concentrationCols],
                  parameters.concentrations, parameters, initial_points, plot=True)
    return results
    
def perVariantFitOld(subSeries, concentrations, parameters, initial_points, eps=None,
                  plot=None, n_samples=None):
    # do a single variant bootstrapping
    if eps is None:
        eps = 1E-3
    if plot is None:
        plot = False
    fitParameters = parameters.fitParameters.copy()
    
    # set fmax upper and lower bound based on number of measurements
    fitParameters.loc[fitParameters.index[:3], 'fmax'] = (
        parameters.find_fmax_bounds_given_n(len(subSeries)))
    
    # set intial guesses for fmax and dG based on initial points
    for param_name in ['fmax', 'dG']:
        if (initial_points.loc[param_name] < fitParameters.loc['upperbound', param_name]-eps and
            initial_points.loc[param_name] > fitParameters.loc['lowerbound', param_name]+eps):
            #print 'changing %s'%param_name
            if param_name == 'dG':
                fitParameters.loc['initial', param_name] = initial_points.loc[param_name]
            if param_name == 'fmax':
                fitParameters.loc['initial', param_name] = initial_points.loc[param_name] 
                                                            
    fmaxDist = parameters.find_fmax_bounds_given_n(len(subSeries), return_dist=True)

    results, singles = fitFun.bootstrapCurves(concentrations, subSeries, fitParameters,
                                     fmaxDist=fmaxDist, default_errors=parameters.default_errors,
                                     verbose=plot, n_samples=n_samples)
    variant = initial_points.name
    return pd.DataFrame(results, columns=[variant]).transpose()


def matchTogetherResults(variant_table, results):
    columns = results.columns[np.in1d(results.columns, variant_table.columns)]
    x = variant_table.copy()
    x.loc[results.index, columns] = results.loc[:, columns]
    return x




def plotSingleVariantFits(table, results, variant, concentrations, plot_init=None,
                          annotate=None):
    if plot_init is None:
        plot_init = False
    if annotate is None:
        annotate = True
    
    # load variant binding series
    concentrationCols = IMlibs.formatConcentrations(concentrations)
    filteredTable = IMlibs.filterStandardParameters(table, concentrations)
    bindingSeries = filteredTable.loc[table.variant_number==variant,
                                      concentrationCols]
    # get error
    try:
        eminus, eplus = fitFun.findErrorBarsBindingCurve(bindingSeries)
    except NameError:
        eminus, eplus = [np.ones(len(concentrations))*np.nan]*2
    
    # plot binding points
    plt.figure(figsize=(4,4));
    plt.errorbar(concentrations, bindingSeries.median(),
                 yerr=[eminus, eplus], fmt='.', elinewidth=1,
                 capsize=2, capthick=1, color='k', linewidth=1)
    
    # plot fit
    more_concentrations =  np.logspace(-2, 4, 50)
    params = lmfit.Parameters()
    for param in ['dG', 'fmax', 'fmin']:
        params.add(param, value=results.loc[variant, param])
    fit = fitFun.bindingCurveObjectiveFunction(params, more_concentrations)
    plt.plot(more_concentrations, fit, 'r')

    try:
        # find upper bound
        params_ub = lmfit.Parameters()
        for param in ['dG_lb', 'fmax_ub', 'fmin']:
            name = param.split('_')[0]
            params_ub.add(name, value=results.loc[variant, param])
        ub = fitFun.bindingCurveObjectiveFunction(params_ub, more_concentrations)
    
        # find lower bound
        params_lb = lmfit.Parameters()
        for param in ['dG_ub', 'fmax_lb', 'fmin']:
            name = param.split('_')[0]
            params_lb.add(name, value=results.loc[variant, param])
        lb = fitFun.bindingCurveObjectiveFunction(params_lb, more_concentrations)
        
        # plot upper and lower bounds
        plt.fill_between(more_concentrations, lb, ub, color='0.5',
                         label='95% conf int', alpha=0.5)
    except:
        pass
    if plot_init:
        try:
            params_init = lmfit.Parameters()
            for param in ['dG_init', 'fmax_init', 'fmin_init']:
                name = param.split('_')[0]
                params_init.add(name, value=results.loc[variant, param])
            init = fitFun.bindingCurveObjectiveFunction(params_init, more_concentrations)
            plt.plot(more_concentrations, init, sns.xkcd_rgb['purplish'], linestyle=':')
        except:
            pass

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
                
        
    ax = plt.gca()
    ax.set_xscale('log')
    ax.tick_params(right='off', top='off')
    plt.xlabel('concentration (nM)')
    plt.ylabel('normalized fluorescence')
    plt.tight_layout()

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
               plot=None):
    if plot is None:
        plot = False

    fitParameters = fitParameters.copy()
    fitParameters.loc[['lowerbound', 'initial', 'upperbound'],
        'fmax'] = fmaxDist.find_fmax_bounds_given_n(len(subSeries))
    fitParameters.loc['initial'] = initial_points.loc[fitParameters.columns]
    results, singles = fitFun.bootstrapCurves(concentrations, subSeries, fitParameters,
                                              func=None, enforce_fmax=True,
                                              fmaxDist=fmaxDist.find_fmax_bounds_given_n(len(subSeries),
                                                                                         return_dist=True))
    if plot:
        fitFun.plotSingleVariantFits(concentrations,
                                     subSeries,
                                     results,
                                     fitParameters,
                                     log_axis=True)
    return results

def fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                     bindingCurveFilename, concentrations, pickled=None,
                     numCores=None, n_samples=None):
    if numCores is None:
        numCores = 20
        
    initialPoints = (pd.concat([pd.read_pickle(annotatedClusterFile),
                                pd.read_pickle(fittedBindingFilename)], axis=1).
                     groupby('variant_number').median())
    table = pd.concat([pd.read_pickle(annotatedClusterFile),
                       pd.read_pickle(bindingCurveFilename)], axis=1).sort('variant_number')

    
    fmaxDist = fitFun.findFinalBoundsParameters(initialPoints, concentrations)
    fitParameters = getInitialParameters(initialPoints, concentrations)
    
    print '\tDividing table into groups...'
    groupDict = {}
    for name, group in table.groupby('variant_number'):
        groupDict[name] = group.drop('variant_number', axis=1)

    print '\tMultiprocessing bootstrapping...'
    results = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(perVariantFit)(concentrations,
                                        groupDict[variant],
                                        fitParameters,
                                        initialPoints.loc[variant],
                                        fmaxDist)
                 for variant in variants))
    results = pd.concat(results)
    results.index = actual_variants
    
    results = getBootstrappedErrors(table, parameters, numCores, n_samples=n_samples)
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
    
    tmp, tmp, concentrations = IMlibs.loadMapFile(args.mapCPfluors)
    variant_table = fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                     bindingCurveFilename, concentrations, pickled=pickled,
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
