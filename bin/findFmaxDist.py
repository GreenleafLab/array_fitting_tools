#!/usr/bin/env python
""" Find the fmax distribution.

Returns a class describing the fmax.

Parameters:
-----------
variant_table : per-variant DataFrame including columns fmax, dG, and fmin, pvalue
initial_points : per-cluster DataFrame including variant_number, fmax, dG, fmin
affinity_cutoff : maximum dG (kcal/mol) to be included in fit of fmax
use_simulated : (bool) whether to use distribution of median fmax or
    subsampled fmaxes

Returns:
--------
fmaxDistObject :
"""

import os
import numpy as np
import scipy.stats as st
import seqfun
from lmfit import Parameters, minimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import argparse
import sys
import pickle
import plotFun
from plotFun import fix_axes
import fitFun
import IMlibs
import fileFun

parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-cf', '--single_cluster_fits', required=True, metavar=".CPfitted.pkl",
                   help='file with single cluster fits')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')

group = parser.add_argument_group('additional option arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')
group.add_argument('-c', '--concentrations', metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
group.add_argument('-k', '--kd_cutoff', type=float,  
                   help='highest kd for tight binders (nM). default is 0.99 bound at '
                   'highest concentration')
group.add_argument('-p', '--pvalue_cutoff', type=float, default=0.01,
                   help='maximum pvalue for good binders. default is 0.01.')
group.add_argument('--use_simulated', type=int,
                   help='set to 0 or 1 if you want to use simulated distribution (1) or'
                   'not (0). Otherwise program will decide.')

class fmaxDistAny():
    __module__ = os.path.splitext(os.path.basename(__file__))[0]
    # for fitting stde of fmaxes
    def __init__(self, params=None):
        self.params = params
        
    def sigma_by_n_fit(self, params, x, y=None, weights=None):
        # objective for 1/sqrt(n) fit to stand error
        parvals = params.valuesdict()
        sigma = parvals['sigma']
        c   = parvals['c']
        fit = sigma/np.sqrt(x) + c
        if y is None:
            return fit
        elif weights is None:
            return y-fit
        else:
            return (y-fit)*weights
    
    def getDist(self, n, do_gamma=None):

        if self.params is None:
            print 'Error: define popts'
            return
        params = self.params
        
        sigma = self.sigma_by_n_fit(params, n)
        mean = params.valuesdict()['median']
        
        return self.find_fmax_bounds(mean, sigma,
                                     alpha=None,
                                     return_dist=True,
                                     do_gamma=do_gamma)
    
    

    def find_fmax_bounds(self, mean, sigma, alpha=None, return_dist=None, do_gamma=None):
        if alpha is None: alpha = 0.99
        if return_dist is None: return_dist = False
        if do_gamma is None:
            do_gamma = True
            
        self.do_gamma = do_gamma
        if do_gamma:    
            return self._find_fmax_bounds_given_gamma(mean, sigma, alpha, return_dist)
        else:
            return self._find_fmax_bounds_given_norm(mean, sigma, alpha, return_dist)
   
    def _get_percentiles_given_alpha(self, alpha):
        # first, middle, and last percentiles that cover fraction (alpha) of data 
        return (1-alpha)/2., 0.5, (1+alpha)/2
    
    def _find_fmax_bounds_given_gamma(self, mean, sigma, alpha, return_dist):
        # distribution is a gamma distribution
        k, theta = returnGammaParams(mean, sigma)
        dist = st.gamma(k, scale=theta)
        return self._return_bounds(dist, alpha, return_dist)

    def _find_fmax_bounds_given_norm(self, mean, sigma, alpha, return_dist):
        # distribution is a normal distribution
        dist = st.norm(loc=mean, scale=sigma)
        return self._return_bounds(dist, alpha, return_dist)
        
    def _return_bounds(self, dist, alpha, return_dist):
        # return upper and lower bounds or distribution
        if return_dist:
            return dist
        else:
            percentiles = self._get_percentiles_given_alpha(alpha)
            return dist.ppf(percentiles)

def findVariantTable(table, parameter='dG', min_fraction_fit=0.25, filterFunction=IMlibs.filterFitParameters):
    """ Find per-variant information from single cluster fits. """
    
    # define columns as all the ones between variant number and fraction consensus
    test_stats = ['fmax', parameter, 'fmin']
    test_stats_init = ['%s_init'%param for param in ['fmax', parameter, 'fmin']]
    other_cols = ['numTests', 'fitFraction', 'pvalue', 'numClusters',
                  'fmax_lb','fmax', 'fmax_ub',
                  '%s_lb'%parameter, parameter, '%s_ub'%parameter,
                  'fmin_lb', 'fmin', 'fmin_ub', 'rsq', 'numIter', 'flag']
    
    table.dropna(subset=['variant_number'], axis=0, inplace=True)
    grouped = table.groupby('variant_number')
    variant_table = pd.DataFrame(index=grouped.first().index,
                                 columns=test_stats_init+other_cols)
    
    # filter for nan, barcode, and fit
    variant_table.loc[:, 'numTests'] = grouped.count().loc[:, parameter]
    
    fitFilteredTable = filterFunction(table)
    fitFilterGrouped = fitFilteredTable.groupby('variant_number')
    index = variant_table.loc[:, 'numTests'] > 0
    
    variant_table.loc[index, 'fitFraction'] = (fitFilterGrouped.count().loc[index, parameter]/
                                           variant_table.loc[index, 'numTests'])
    variant_table.loc[index, 'fitFraction'].fillna(0)
    # then save parameters
    old_test_stats = grouped.median().loc[:, test_stats]
    old_test_stats.columns = test_stats_init
    variant_table.loc[:, test_stats_init] = old_test_stats
    
    # null model is that all the fits are bad. 
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*
             variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n].dropna().astype(float)
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, min_fraction_fit)
    
    return variant_table

def useSimulatedOrActual(variant_table, cutoff):
    # if at least 20 data points have at least 10 counts in that bin, use actual
    # data. This statistics seem reasonable for fitting
    index = variant_table.dG_init < cutoff
    counts, binedges = np.histogram(variant_table.loc[index].numTests,
                                    np.arange(1, variant_table.numTests.max()))
    if (counts > 10).sum() >= 20:
        use_actual = True
    else:
        use_actual = False
    return use_actual

def fitSigmaDist(x, y, weights=None, set_c=None, at_n=None):
    # fit how sigmas scale with number of measurements
    fmaxDist = fmaxDistAny()
    params = Parameters()
    params.add('sigma', value=y.max(), min=0)
    if set_c is None:
        params.add('c',     value=y.min(),   min=0)
    else:
        params.add('c', expr="%4.5f-sigma/sqrt(%4.5f)"%(set_c, at_n))
    minimize(fmaxDist.sigma_by_n_fit, params,
                                   args=(x,),
                                   kws={'y':y,
                                        'weights':weights},
                                   xtol=1E-6, ftol=1E-6, maxfev=10000)
    return params

def gammaObjective(params, x, data=None, weights=None, return_pdf=None):
    if return_pdf is None:
        return_pdf is False
    parvals = params.valuesdict()
    mean = parvals['mean']
    std  = parvals['std']
    mu = parvals['offset']
    
    k, theta = returnGammaParams(mean, std)

    if return_pdf:
        cdf = st.gamma.pdf(x, k, scale=theta, loc=mu)
    else:
        cdf = st.gamma.cdf(x, k, scale=theta, loc=mu)
    
    if data is None:
        return cdf
    elif weights is None:
        return cdf - data
    else:
        return (cdf - data)*weights   
   

def fitGammaDistribution(vec, plot=None, set_mean=None, set_offset=None, initial_mean=None):
    """ Fit the CDF of a vector to the gamma distribution. """
    
    if plot is None:
        plot = False
    
    if initial_mean is None:
        # set initial mean for fit
        initial_mean = vec.median() 
    
    # fit the cdf of vec to find parameters that best represent the distribution
    x, y = seqfun.getCDF(vec)
    param_names = ['mean', 'std', 'offset']
    params = Parameters()
    
    # distribution has mean (first moment)
    if set_mean is None:
        params.add('mean', value=initial_mean, vary=True, min=0, max=np.inf)
    else:
        params.add('mean', value=set_mean, vary=False)

    # distribution has standard deviation (sqrt of second moment)
    params.add('std',  value=vec.std(), vary=True, min=0, max=np.inf)

    # gamma distribution can have offset (probably centered at 0)
    if set_offset is None:
        params.add('offset', value=0, vary=True)
    else:
        params.add('offset', value=set_offset, vary=False)
        
    # go through with least squares minimization
    results = minimize(gammaObjective, params,
                       args=(x,),
                       kws={'data':y})

    # save params and rsq
    index = (param_names + ['%s_stde'%param for param in param_names] +
             ['rsq', 'exit_flag', 'rmse'])
    final_params = pd.Series(index=index)
    
    # save params in structure
    for param in param_names:
        final_params.loc[param] = params[param].value
        final_params.loc['%s_stde'%param] = params[param].stderr

    # find rqs
    ss_total = np.sum((y - y.mean())**2)
    ss_error = np.sum((results.residual)**2)
    rsq = 1-ss_error/ss_total
    rmse = np.sqrt(ss_error)
    final_params.loc['rsq'] = rsq
    final_params.loc['exit_flag'] = results.ier
    final_params.loc['rmse'] = rmse
    
    if plot:
        # plot pdf 
        plotGammaFunction(vec, params=params)
    return final_params

def returnGammaParams(mean, std):
    # moments of distrubtion mean, std are related to shape and scale parameters
    k = (mean/std)**2
    theta = (std)**2/mean
    return k, theta

def plotDist(vec, bounds):
    plt.figure(figsize=(3.5,3))
    sns.distplot(vec, hist_kws={'histtype':'stepfilled'}, color='0.5', kde_kws={'clip':bounds})
    ax = fix_axes(plt.gca())
    xlim = ax.get_xlim()
    plt.xlim(0, xlim[1])
    plt.xlabel('initial $f_{max}$')
    plt.ylabel('probability density')
    plt.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)
    return


def plotGammaFunction(vec, results=None, params=None, bounds=[0,5]):
    """ Take vector and fit and plot distribution. """
    # plot pdf 
    plotDist(vec, bounds)
    more_x = np.linspace(*bounds, num=100)

    if results is None and params is None:
        print "need to define either results or params to plot fit"
    else:
        if params is None:
            params = fitFun.returnParamsFromResults(results, param_names=['mean', 'std', 'offset'])
        plt.plot(more_x, gammaObjective(params, more_x, return_pdf=True), 'r')
    

def getFmaxMeanAndBounds(tight_binders, cutoff=1E-12):
    """ Return median fmaxes of variants. """
    # find defined mean shared by all variants by fitting all
    fmaxes = tight_binders.fmax_init
    fmaxAllFit = fitGammaDistribution(fmaxes, set_offset=0, initial_mean=fmaxes.max())
    
    # use fit to also define upper and lower bound of expected values
    mean_fmax = fmaxAllFit.loc['mean']
    std_fmax =  fmaxAllFit.loc['std']
    lowerbound, median, upperbound = (fmaxDistAny().
                                      find_fmax_bounds(mean_fmax, std_fmax, alpha=1-cutoff))
    loose_bounds = [0, 2*upperbound]
    plotGammaFunction(fmaxes, fmaxAllFit, bounds=loose_bounds)
    plt.axvline(lowerbound, color='k', linestyle=':')
    plt.axvline(upperbound, color='k', linestyle=':')
    return mean_fmax, [lowerbound, upperbound], loose_bounds
    

def getFmaxesToFit(tight_binders, bounds=[0, np.inf]):
    """ Return fmax initial fits that fall within bounds. """
    fmaxes = tight_binders.fmax_init
    
    # find those within bounds
    index = (fmaxes>=bounds[0])&(fmaxes<=bounds[1])
    n_tests = tight_binders.loc[index].numTests
    return fmaxes.loc[index], n_tests

def getFmaxesToFitSimulated(all_clusters, good_variants, bounds=[0, np.inf], n_subset=np.arange(1,15)):
    """ Return simulated median fmaxes from randomly sampled clusters.
    
    Returns a distribution of fmaxes for increasing number of samples.

    Parameters:
    -----------
    all_clusters : per-cluster DataFrame giving initial fits. Columns variant
        number and fmax.
    good_variants : list of variant numbers that represent variants passing cutoffs.
    bounds : only include median values of fmax within bounds.
    n_subset : list of number of tests to simulate.
    
    Returns:
    --------
    DataFrame of median fmaxes
    DataFrame of number of tests
    
    """
        
    # find those clusters associated with good variants and use those it fmaxes
    good_clusters = pd.Series(np.in1d(all_clusters.variant_number,
                                      good_variants),
                              index=all_clusters.index)
    fmax_clusters = all_clusters.loc[good_clusters, 'fmax']
    
    # first assign each cluster to a variant. different assignments for each n
    fmax_df = pd.concat([fmax_clusters,
                         pd.DataFrame(index=fmax_clusters.index, columns=n_subset)], axis=1)
    for n in n_subset:
        # make at most 1000 or max_num_variants independent subsamples of n samples each 
        num_variants = min(1000, np.floor(len(fmax_clusters)/float(n)))
        n_samples = n*num_variants
        clusters = np.random.choice(fmax_clusters.index, n_samples, replace=False)
        fmax_df.loc[clusters, n] = np.tile(np.arange(num_variants), n)
    
    # now find median of fmaxes assigned to each variant
    fmaxes = []
    n_tests = []
    for n in n_subset:
        
        # only take values within bounds
        vec = fmax_df.groupby(n)['fmax'].median()
        index = (vec>=bounds[0])&(vec<=bounds[1])
        
        # make vector of tests
        n_vec = vec.loc[index].copy()
        n_vec.loc[:] = n
        
        # save
        fmaxes.append(vec.loc[index])
        n_tests.append(n_vec)
    
    return pd.concat(fmaxes, ignore_index=True), pd.concat(n_tests, ignore_index=True)

def getFractionOfData(n_test_counts, fraction_of_data):
    for n in n_test_counts.index:
        if n_test_counts.loc[:n].sum()/float(n_test_counts.sum()) < fraction_of_data:
            pass
        else:
            min_n = n
            break
    return min_n

def findMinStd(fmaxes, n_tests, mean_fmax, fraction_of_data=0.5):
    """ Find a minimum standard deviation. """
    n_test_counts = n_tests.value_counts().sort_index()
    # find number of tests that contains 50% of the variants
    min_n = getFractionOfData(n_test_counts, fraction_of_data)
    
    # fit to find the "floor" of std below which represents experimental noise
    fmaxHighNFit = fitGammaDistribution(fmaxes.loc[n_tests>=min_n],
                                        set_mean=mean_fmax, set_offset=0)
    min_std = fmaxHighNFit.loc['std']
    at_n = np.average(n_test_counts.loc[min_n:].index.tolist(),
                      weights=n_test_counts.loc[min_n:].values)
    return min_std, at_n

def findStdParams(fmaxes, n_tests, mean_fmax, min_std, at_n, min_num_to_fit=4):
    """ Find the relationship between number of tests and std. """
    n_test_counts = n_tests.value_counts().sort_index()
    all_ns = n_test_counts.loc[n_test_counts>=min_num_to_fit].index.tolist()
    
    stds_actual = {}
    for n in all_ns:
        stds_actual[n] = fitGammaDistribution(fmaxes.loc[n_tests==n],
                                              set_mean=mean_fmax,
                                              )
    stds_actual = pd.concat(stds_actual, axis=1).transpose()
    stds_actual.dropna(inplace=True)

    x = stds_actual.index.tolist()
    y = stds_actual.loc[:, 'std']
    weights_fit = np.sqrt(n_test_counts.loc[stds_actual.index])

    params = fitSigmaDist(x, y,
                          weights=weights_fit,
                          set_c=min_std, at_n=at_n)
    params.add('median', value=mean_fmax)
    return params, stds_actual
    
def findParams(tight_binders, use_simulated=None, table=None):
    """ Initialize, find the fmax distribution object and plot. """
    if use_simulated is None:
        use_simulated = False
    if use_simulated and (table is None):
        print "error: need to give table of all cluster fits to do simulated data"
        return
    
    mean_fmax, bounds, loose_bounds = getFmaxMeanAndBounds(tight_binders)
    fmaxes_data, n_tests_data = getFmaxesToFit(tight_binders, bounds=bounds)
    
    if use_simulated:
        # find min std of distribution anyways
        min_std, at_n = findMinStd(fmaxes_data, n_tests_data, mean_fmax)
        fmaxes, n_tests = getFmaxesToFitSimulated(table, tight_binders.index, bounds=bounds)
    else:
        min_std = None; at_n = None
        fmaxes, n_tests = fmaxes_data, n_tests_data

    # fit relationship of std with number of measurements
    params, stds_actual = findStdParams(fmaxes, n_tests, mean_fmax, min_std, at_n)
    
    fmaxDist = fmaxDistAny(params=params)
    
    # plot
    maxn = getFractionOfData(n_tests_data.value_counts().sort_index(), 0.995)
    ax = plotFun.plotFmaxStdeVersusN(fmaxDist, stds_actual, maxn)
    # if use_simulated, plot actual as well
    if use_simulated:
        try:
            params1, stds_not_sim = findStdParams(fmaxes_data, n_tests_data, mean_fmax, None, None)
            fmaxDist1 = fmaxDistAny(params=params1)
            plotFun.plotFmaxStdeVersusN(fmaxDist1, stds_not_sim, maxn, ax=ax)
        except:
            pass
        
    # plot offsets
    plotFun.plotFmaxOffsetVersusN(fmaxDist, stds_actual, maxn)
    # if use_simulated, plot actual as well
    if use_simulated:
        try:
            params1, stds_not_sim = findStdParams(fmaxes_data, n_tests_data, mean_fmax, None, None)
            fmaxDist1 = fmaxDistAny(params=params1)
            plotFun.plotFmaxOffsetVersusN(fmaxDist1, stds_not_sim, maxn, ax=ax)
        except:
            pass  

    # plot numbers
    plotFun.plotNumberVersusN(n_tests_data, maxn)
    
    return fmaxDist

def resultsFromFmaxDist(fmaxDist, n):
    """ Return results from fmaxDist. """
    mean, var = fmaxDist.getDist(n).stats(moments='mv')
    return pd.Series({'std':np.sqrt(var), 'mean':mean, 'offset':0})

def findUpperboundFromFmaxDistObject(fmaxDist):
    """ Return loose bounds on fmax from fmaxDist. """
    return fmaxDist.params['median'] + 3*fmaxDist.params['sigma']

def plotAnyN(tight_binders, fmaxDistObject, n):
    bounds = [0, findUpperboundFromFmaxDistObject(fmaxDistObject)]
    x = np.linspace(*bounds, num=100)
    plotDist(tight_binders.loc[tight_binders.numTests==n].fmax_init, bounds)
    plt.plot(x, fmaxDistObject.getDist(n).pdf(x), 'r')
    plt.tight_layout()
    
if __name__=="__main__":
    args = parser.parse_args()
    fittedBindingFilename = args.single_cluster_fits
    annotatedClusterFile  = args.annotated_clusters

    outFile  = args.out_file
    kd_cutoff = args.kd_cutoff
    use_simulated = args.use_simulated
    
    # find out file
    if outFile is None:
        outFile = os.path.splitext(
            fittedBindingFilename[:fittedBindingFilename.find('.pkl')])[0]
        
    # make fig directory
    figDirectory = os.path.join(os.path.dirname(outFile), fileFun.returnFigDirectory())
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)

    # need to define concentrations or kd_cutoff in order to find affinity cutoff
    if args.concentrations is not None:
        concentrations = np.loadtxt(args.concentrations)
    elif kd_cutoff is None:
        print 'Error: need to either give concentrations or kd_cutoff.'
        sys.exit()
        
    # define cutoffs
    parameters = fitFun.fittingParameters()
    if kd_cutoff is not None:
        # adjust cutoff to reflect this fraction bound at last concentration
        affinity_cutoff = parameters.find_dG_from_Kd(kd_cutoff)
    else:
        parameters = fitFun.fittingParameters(concentrations)
        affinity_cutoff = parameters.maxdG
    pvalue_cutoff = args.pvalue_cutoff
    print 'Using variants with kd less than %4.2f nM'%parameters.find_Kd_from_dG(affinity_cutoff)
    print 'Using variants with pvalue less than %.1e'%pvalue_cutoff

    # load initial fits per cluster
    print 'Loading data...'
    initial_points = pd.concat([pd.read_pickle(annotatedClusterFile),
                                pd.read_pickle(fittedBindingFilename)], axis=1).astype(float)

    # find variant_table
    variant_table = findVariantTable(initial_points).astype(float)
    good_fits = variant_table.pvalue < pvalue_cutoff
    tight_binders = variant_table.loc[good_fits&(variant_table.dG_init<affinity_cutoff)]
    print ('%d out of %d variants pass cutoff'
           %(len(tight_binders), len(variant_table)))
    if len(tight_binders) < 10:
        print 'Error: need more variants passing cutoffs to fit'
        print sys.exit()

    # find good variants
    plotFun.plotFmaxVsKd(variant_table.loc[good_fits], parameters.find_Kd_from_dG(affinity_cutoff))
    plt.savefig(os.path.join(figDirectory, 'fmax_vs_kd_init.pdf'))
    
    plotFun.plotFmaxVsKd(variant_table.loc[good_fits], parameters.find_Kd_from_dG(affinity_cutoff),
                         plot_fmin=True)
    plt.savefig(os.path.join(figDirectory, 'fmin_vs_kd_init.pdf'))  

    # if use_simulated is not given, decide
    if use_simulated is None:
        use_simulated = not useSimulatedOrActual(variant_table.loc[good_fits], affinity_cutoff)
    if use_simulated:
        print 'Using fmaxes drawn randomly from clusters'
    else:
        print 'Using median fmaxes of variants'
    
    # find fmax dist object
    fmaxDist = findParams(tight_binders,
                                use_simulated=use_simulated,
                                table=initial_points)
    plt.savefig(os.path.join(figDirectory, 'counts_vs_n_tight_binders.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'offset_fmax_vs_n.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'stde_fmax_vs_n.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'fmax_dist.all.pdf'));
    
    # save
    pickle.dump(fmaxDist, open( outFile+'.fmaxdist.p', "wb" ))
    
    # save variant table
    variant_table.to_pickle(outFile + '.init.CPvariant.pkl')
    
    # plot some examples
    for n in np.percentile(variant_table.numTests, [10, 50, 90]):
        plotAnyN(tight_binders, fmaxDist, n)
        plt.savefig(os.path.join(figDirectory, 'fmax_dist.n_%d.pdf'%n))
    
    # plot fraction fit
    plotFun.plotFractionFit(variant_table, pvalue_threshold=pvalue_cutoff)
    plt.savefig(os.path.join(figDirectory, 'fraction_passing_cutoff_in_affinity_bins.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'histogram_fraction_fit.pdf'))
    

                
    