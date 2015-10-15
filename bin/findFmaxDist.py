import numpy as np
import scipy.stats as st
import seqfun
from lmfit import Parameters, minimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def find_fmax_bounds_given_n(self, n, alpha=None, return_dist=None, do_gamma=None):
        if alpha is None: alpha = 0.99
        if return_dist is None: return_dist = False
        if do_gamma is None:
            do_gamma = False
        
        if self.params is None:
            print 'Error: define popts'
            return
        params = self.params
        
        sigma = self.sigma_by_n_fit(params, n)
        mean = params.valuesdict()['median']

        if do_gamma:
            return self._find_fmax_bounds_given_n_gamma(mean, sigma, alpha, return_dist)
        else:
            return self._find_fmax_bounds_given_n_norm(mean, sigma, alpha, return_dist)

    def _get_percentiles_given_alpha(self, alpha):
        # first, middle, and last percentiles that cover fraction (alpha) of data 
        return (1-alpha)/2., 0.5, (1+alpha)/2
    
    def _find_fmax_bounds_given_n_gamma(self, mean, sigma, alpha, return_dist):
        # distribution is a gamma distribution
        k, theta = returnGammaParams(mean, sigma)
        dist = st.gamma(k, scale=theta)
        return self._return_bounds(dist, alpha, return_dist)

    def _find_fmax_bounds_given_n_norm(self, mean, sigma, alpha, return_dist):
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
    

def skewPDF(x, mu=None, omega=None, alpha=None):
    if mu is None:
        mu = 0
    if omega is None:
        omega = 1
    if alpha is None:
        alpha = 0
        
    t = (x - mu) / omega
    return 2/omega*st.norm.pdf(t)*st.norm.cdf(alpha*t)

def skewedGaussianObjective(params, x, data=None, weights=None):
    
    parvals = params.valuesdict()
    mean = parvals['mean']
    std  = parvals['std']
    skew = parvals['skew']
    
    alpha = skew
    omega = np.sqrt(std**2/(1-2/np.pi*gamma(alpha)**2))
    mu = mean - omega*np.sqrt(2/np.pi)*gamma(alpha)
    
    
    pdf = skewPDF(x, mu=mu, omega=omega, alpha=alpha)

    
    if data is None:
        return pdf
    elif weights is None:
        return pdf - data
    else:
        return (pdf - data)*weights

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
   

def fitDistribution(vec, plot=None, set_mean=None, set_offset=None):
    if plot is None:
        plot = False

    x, y = seqfun.getCDF(vec)
    param_names = ['mean', 'std', 'offset']
    params = Parameters()
    params.add('std',  value=vec.std(), vary=True, min=0, max=1)
    
    if set_offset is None:
        params.add('offset', value=0, vary=True)
    else:
        params.add('offset', value=set_offset, vary=False)
    # mean
    if set_mean is None:
        params.add('mean', value=vec.mean(), vary=True, min=0, max=1.5)
    else:
        params.add('mean', value=set_mean, vary=False)
    
    
    results = minimize(gammaObjective, params,
                       args=(x,),
                       kws={'data':y})

    index = (param_names + ['%s_stde'%param for param in param_names] +
             ['rsq', 'exit_flag', 'rmse'])
    final_params = pd.Series(index=index)
    
    # find rqs
    ss_total = np.sum((y - y.mean())**2)
    ss_error = np.sum((results.residual)**2)
    rsq = 1-ss_error/ss_total
    rmse = np.sqrt(ss_error)
    
    # save params in structure
    for param in param_names:
        final_params.loc[param] = params[param].value
        final_params.loc['%s_stde'%param] = params[param].stderr
    final_params.loc['rsq'] = rsq
    final_params.loc['exit_flag'] = results.ier
    final_params.loc['rmse'] = rmse
    
    if plot:
        more_x = np.linspace(0, 2)
        plt.figure(figsize=(4,4))
        sns.distplot(vec, hist_kws={'histtype':'stepfilled'}, color='0.5')
        plt.plot(more_x, gammaObjective(params, more_x, return_pdf=True), 'r')
        plt.xlim(0, 2)
        plt.tight_layout()
    return final_params

def returnParamsFromResults(final_params):
    param_names = ['mean', 'std', 'offset']
    params = Parameters()
    for param in param_names:
        params.add(param, value=final_params.loc[param])
    return params

def returnGammaParams(mean, std, skew=None):
    if skew is None:
        k = (mean/std)**2
        theta = (std)**2/mean
    
        return k, theta
    else:
        k = 2/skew**2
        theta = std*skew/2
        mu = mean - k*theta
        return k, theta, mu

def findParams():
    tight_binders = variant_table.loc[(variant_table.dG_init <= parameters.maxdG)&(variant_table.pvalue < 0.01)]

    # has mean (weighted average of fmax)
    fmax_sub = excludeOutliersGaussian(tight_binders.fmax_init,
                                       loc=tight_binders.fmax_init.median())
    mean_fmax = fitDistribution(tight_binders.fmax_init).loc['mean']
    n_tests = tight_binders.loc[fmax_sub.index].numTests
    n_tests_counts = n_tests.value_counts().sort_index()
    lowerbound, upperbound =  np.percentile(fmax_sub, [0, 100])
    
    cdf = pd.Series([n_tests_counts.loc[:i].sum()/float(n_tests_counts.sum()) for i in n_tests_counts.index],index=n_tests_counts.index)
    min_n = np.abs(cdf-0.95).argmin()
    min_std = fitDistribution(fmax_sub.loc[n_tests>=min_n], set_mean=mean_fmax).loc['std']
    # find relationship of std and number of tests
    min_num_to_fit = 10
    all_ns = np.array(n_tests_counts.loc[n_tests_counts >= min_num_to_fit].index.tolist())
    stds_actual = {}
    stds_sim = {}
    for n in all_ns:
        # find std
        stds_actual[n] = fitDistribution(fmax_sub.loc[n_tests==n], set_mean=mean_fmax)
        if n < 80 and n%4==0:
            print n
            vec = returnSimulatedFmaxes(fmax_clusters_sub, n)
            stds_sim[n] = fitDistribution(vec.loc[(vec>=lowerbound)&(vec<=upperbound)], set_mean=mean_fmax, set_offset=0)
    stds_actual = pd.concat(stds_actual, axis=1).transpose()
    stds_sim  = pd.concat(stds_sim, axis=1).transpose()
    stds_actual.dropna(inplace=True)

    x = stds_actual.index
    y = stds_actual.loc[:, 'std']
    weights_fit = np.sqrt(n_tests_counts.loc[stds_actual.index])

    params = fitSigmaDist(stds_actual.index, stds_actual.loc[:, 'std'], weights=weights_fit)
    params2 = fitSigmaDist(stds_sim.index, stds_sim.loc[:, 'std'], set_c=min_std, at_n=all_ns.max())
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