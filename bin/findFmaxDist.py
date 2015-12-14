#!/usr/bin/env python
#
# will find fmax distribution given the single cluster fits on 
# good binders and good fitters.
#
# fmax distribtuion is then enforced for weak binders, allowing
# good measurements even when no reaching saturation.
# 
# Sarah Denny
# July 2015

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
import plotFun
import fitFun
import IMlibs

parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-t', '--single_cluster_fits', required=True, metavar=".CPfitted.pkl",
                   help='file with single cluster fits')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')

group = parser.add_argument_group('additional option arguments')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')
parser.add_argument('-c', '--concentrations', metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
group.add_argument('--kd_cutoff', type=float,  
                   help='highest kd for tight binders (nM). default is 0.99 bound at '
                   'highest concentration')
group.add_argument('--use_simulated', type=int,
                   help='set to 0 or 1 if you want to use simulated distribution (1) or'
                   'not (0). Otherwise program will decide.')

class fmaxDistAny():
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

def findVariantTable(table, parameter='dG', min_fraction_fit=0.25):
    """ Find per-variant information from single cluster fits. """
    
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
    
    # null model is that all the fits are bad. 
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*
             variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n].dropna().astype(float)
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, min_fraction_fit)
    
    return variant_table

def findFmaxDistObject(variant_table, initial_points, affinity_cutoff, use_simulated=None):
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
    parameters = fitFun.fittingParameters()
    kd_cutoff = parameters.find_Kd_from_dG(affinity_cutoff)
    index = variant_table.pvalue < 0.01
    plotFun.plotFmaxVsKd(variant_table, kd_cutoff, index)
    
    # if use_simulated is not given, decide
    if use_simulated is None:
        use_simulated = not useSimulatedOrActual(variant_table.loc[index], affinity_cutoff)
    if use_simulated:
        print 'Using fmaxes drawn randomly from clusters'
    else:
        print 'Using median fmaxes of variants'
    tight_binders = variant_table.loc[index&
                                      (variant_table.dG_init<affinity_cutoff)]
    fmaxDistObject = findParams(tight_binders,
                                       use_simulated=use_simulated,
                                       table=initial_points)
    return fmaxDistObject

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

def plotGammaFunction(vec, results=None, params=None):
    """ Take vector and fit and plot distribution. """
    # plot pdf 
    more_x = np.linspace(0, 5, 100)
    plt.figure(figsize=(4,4))
    sns.distplot(vec, hist_kws={'histtype':'stepfilled'}, color='0.5')
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.xlim(0, xlim[1])
    plt.tight_layout()
    
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
    plotGammaFunction(fmaxes, fmaxAllFit)
    ax = plt.gca()
    ylim = ax.get_ylim()
    ax.plot([lowerbound]*2, ylim, 'k:')
    ax.plot([upperbound]*2, ylim, 'k:')
    return mean_fmax, [lowerbound, upperbound]
    

def getFmaxesToFit(tight_binders, bounds=None):
    if bounds is None:
        bounds = [0, np.inf]
        
    # find fmaxes and n_tests
    fmaxes = tight_binders.fmax_init
    index = (fmaxes>=bounds[0])&(fmaxes<=bounds[1])
    
    n_tests = tight_binders.loc[index].numTests
    return fmaxes.loc[index], n_tests

def getFmaxesToFitSimulated(all_clusters, good_variants, bounds=None, n_subset=None):
    """ Return simulated median fmaxes from randomly sampled clusters. """
    if bounds is None:
        bounds = [0, np.inf]
    if n_subset is None:
        n_subset = np.arange(1, 6)
        
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
        

def findMinStd(fmaxes, n_tests, mean_fmax, fraction_of_data=0.5):
    """ Find a minimum standard deviation. """

    # find number of tests that contains 50% of the variants
    n_test_counts = n_tests.value_counts().sort_index()
    for n in n_test_counts.index:
        if n_test_counts.loc[:n].sum()/float(n_test_counts.sum()) < fraction_of_data:
            pass
        else:
            min_n = n
            break
    
    # fit to find the "floor" of std below which represents experimental noise
    fmaxHighNFit = fitGammaDistribution(fmaxes.loc[n_tests>=min_n],
                                        set_mean=mean_fmax, set_offset=0)
    min_std = fmaxHighNFit.loc['std']
    at_n = np.average(n_test_counts.loc[n:].index.tolist(),
                      weights=n_test_counts.loc[n:].values)
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
    return params, x, y, stds_actual
    
def findParams(tight_binders, use_simulated=None, table=None):
    """ Initialize, find the fmax distribution object and plot. """
    if use_simulated is None:
        use_simulated = False
    if use_simulated and (table is None):
        print "error: need to give table of all cluster fits to do simulated data"
        return
    
    mean_fmax, bounds = getFmaxMeanAndBounds(tight_binders)
    fmaxes_data, n_tests_data = getFmaxesToFit(tight_binders, bounds=bounds)
    
    if use_simulated:
        # find min std of distribution anyways
        min_std, at_n = findMinStd(fmaxes_data, n_tests_data, mean_fmax)
        fmaxes, n_tests = getFmaxesToFitSimulated(table, tight_binders.index, bounds=bounds)
    else:
        min_std = None; at_n = None
        fmaxes, n_tests = fmaxes_data, n_tests_data

    # fit relationship of std with number of measurements
    params, x, y, stds_actual = findStdParams(fmaxes, n_tests, mean_fmax, min_std, at_n)
    
    fmaxDist = fmaxDistAny(params=params)
    
    # plot
    x_fit = np.arange(1, n_tests_data.max())
    y_fit = fmaxDist.sigma_by_n_fit(params, x_fit)
    plt.figure(figsize=(4,3))
    if use_simulated:
        try:
            params1, x1, y1, stds_not_sim = findStdParams(fmaxes_data, n_tests_data, mean_fmax, None, None)
            plt.scatter(x1, y1, s=5, marker='.', color='0.5')
            y1_fit = fmaxDist.sigma_by_n_fit(params1, x_fit)
            plt.plot(x_fit, y1_fit, ':', color='0.5', label='using variant fmaxes')
        except: pass
        
    plt.scatter(x, y, s=10, marker='o', color='k')
    plt.plot(x_fit, y_fit, 'c')
    plt.xlabel('number of measurements')
    plt.ylabel('standard deviation of median fmax')
    plt.xlim(0, x_fit.max())
    plt.tight_layout()
    
    plot=False
    if plot:
        # also plot offsets
        plt.figure(figsize=(4,3))
        sns.distplot(stds_actual.offset)
        
        y_smoothed = gaussian_filter(stds_actual.offset, 2)
        plt.figure(figsize=(4,3));
        plt.errorbar(x, stds_actual.offset, yerr=stds_actual.offset_stde)
        #plt.plot(x, y_smoothed, 'c')
        plt.xlabel('number of measurements')
        plt.ylabel('offset')
        plt.tight_layout()
    return fmaxDist

def resultsFromFmaxDist(fmaxDist, n):
    mean, var = fmaxDist.getDist(n).stats(moments='mv')
    return pd.Series({'std':np.sqrt(var), 'mean':mean, 'offset':0})

def other():
    x1 = stds_actual.index.tolist()
    y1 = stds_actual.loc[:, 'std']
    weights_fit = np.sqrt(n_test_counts.loc[stds_actual.index])
    
    params2 = fitSigmaDist(x1, y1,
                          weights=weights_fit,
                          set_c=min_std, at_n=at_n)
    y_fit2 = fmaxDist.sigma_by_n_fit(params2, x)
    
    plt.figure()
    plt.plot(x, y, 'k.', )
    plt.scatter(x1, y1, facecolors='none', edgecolors='k')
    plt.plot(x, y_fit, 'c')
    plt.plot(x, y_fit2, 'r')
    
    # save fitting parameters
    params.add('median', value=mean_fmax)
    fmaxDist = fmaxDistAny(params=params)
    y_fit = fmaxDist.sigma_by_n_fit(params, x)
    y_fit2 = fmaxDist.sigma_by_n_fit(params2, x)
    plt.figure()
    plt.plot(stds_actual.index, stds_actual.loc[:, 'std'], 'k.', )
    plt.scatter(stds_sim.index, stds_sim.loc[:, 'std'], facecolors='none', edgecolors='k')
    plt.plot(x, y_fit, 'c')
    plt.plot(x, y_fit2, 'r')
    

    
    x = stds_actual.index
    y = stds_actual.loc[:, 'std']   
    
    
    plt.figure()
    plt.plot(x, stds_actual.loc[:, 'mean'], 'o')

    plt.figure()
    plt.plot(stds_actual.index, stds_actual.loc[:, 'offset'], 'o')
    # plot some examples
    n = 52
    x = np.linspace(0, 2)
    plt.figure(figsize=(4,4));
    sns.distplot(fmax_sub.loc[numTests==n], bins=np.linspace(0, 2, 20),
                 hist_kws={'histtype':'stepfilled'}, color='0.5');
    plt.plot(x, gammaObjective(returnParamsFromResults(stds_actual.loc[n]), x,
                               return_pdf=True), color='r');
    std = fmaxDist.sigma_by_n_fit(params, n)
    
    plt.plot(x, gammaObjective(returnParamsFromResults(pd.Series({'std':std, 'mean':mean_fmax, 'offset':0})), x,
                               return_pdf=True), color='b');
    plt.xlim(0, 2);
    plt.tight_layout();
    
    n = 16
    x = np.linspace(0, 2)
    plt.figure(figsize=(4,4));
    sns.distplot(fmax_sub.loc[numTests==n], bins=np.linspace(0, 2, 20),
                 hist_kws={'histtype':'stepfilled'}, color='0.5');
    plt.plot(x, gammaObjective(returnParamsFromResults(stds_actual.loc[n]), x,
                               return_pdf=True), color='r');
    plt.xlim(0, 2);
    plt.tight_layout();
    
    mean, var = fmaxDist.find_fmax_bounds_given_n(n, return_dist=True).stats(moments='mv')
    final_params = pd.Series({'std':np.sqrt(var), 'mean':mean, 'offset':0})
    
    
if __name__=="__main__":
    args = parser.parse_args()
    fittedBindingFilename = args.single_cluster_fits
    annotatedClusterFile  = args.annotated_clusters

    outFile  = args.out_file
    kd_cutoff = args.kd_cutoff
    use_simulated = args.use_simulated
    

    if args.concentrations is not None:
        concentrations = np.loadtxt(args.concentrations)
    elif kd_cutoff is None:
        print 'Error: need to either give concentrations or kd_cutoff.'
        sys.exit()

    # load arguments
    initial_points = pd.concat([pd.read_pickle(annotatedClusterFile),
                                pd.read_pickle(fittedBindingFilename)], axis=1).astype(float)

    # find variant_table
    variant_table = findVariantTable(initial_points).astype(float)
    
    # find maxdG
    parameters = fitFun.fittingParameters()
    if kd_cutoff is not None:
        # adjust cutoff to reflect this fraction bound at last concentration
        maxdG = parameters.find_dG_from_Kd(kd_cutoff)
    else:
        parameters = fitFun.fittingParameters(concentrations)
        maxdG = parameters.maxdG
    print 'Use variants with kd less than %4.2f nM'%parameters.find_Kd_from_dG(maxdG)

    
    # find fmax Dist
    fmaxDistObject = findFmaxDistObject(variant_table, initial_points, maxdG,
                                        use_simulated=use_simulated)
    
    # save
    
    