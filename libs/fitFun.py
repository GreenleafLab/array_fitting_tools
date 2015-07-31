from lmfit import minimize, Parameters, report_fit, conf_interval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scikits.bootstrap import bootstrap
from statsmodels.stats.weightstats import DescrStatsW
import warnings
import itertools
import seqfun
import IMlibs
import scipy.stats as st


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
        
        # assume that fluorsecnce in last binding point of tightest binders
        # is on average at least 25% bound. May want to lower if doing
        # different point for binding point. 
        self.saturation_level   = 0.25
        
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
    
    def sigma_by_n_fit(self, params, x, y=None, weights=None):

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
    
    def find_fmax_bounds_given_n(self, n, alpha=None, return_dist=None):
        if alpha is None: alpha = 0.99
        if return_dist is None: return_dist = False
        
        if self.params is None:
            print 'Error: define popts'
            return
        params = self.params
         
        sigma = self.sigma_by_n_fit(params, n)
        mean = params.valuesdict()['median']

        if 'min_sigma' in params.keys():
            if sigma < params['min_sigma'].value:
                sigma = params['min_sigma'].value

        if return_dist:
            return st.norm(loc=mean, scale=sigma)
        else:
            interval = st.norm.interval(alpha, loc=mean, scale=sigma)
            return interval[0], mean, interval[1]
        
def bindingCurveObjectiveFunction(params, concentrations, data=None, weights=None):
    parameters = fittingParameters()
    
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    dG   = parvals['dG']
    fmin = parvals['fmin']
    fracbound = (fmin + (fmax - fmin)*concentrations/
                 (concentrations + np.exp(dG/parameters.RT)/parameters.concentration_units))

    if data is None:
        return fracbound
    elif weights is None:
        return fracbound - data
    else:
        return (fracbound - data)*weights
    
def jacobianBindingCurve(params, concentrations, data=None, weights=None):
    parvals = params.valuesdict()
    jcb = pd.DataFrame(index=parvals.keys(), columns=np.arange(len(concentrations)))
    
    alpha = 0.582
    beta = 1e-9
    jcb.loc['fmax'] = concentrations/(concentrations+np.exp(parvals['dG']/alpha)/beta)
    jcb.loc['dG']   = (-beta*parvals['fmax']*concentrations*np.exp(parvals['dG']/
                                                                   alpha)/
                       (alpha*(beta*concentrations+np.exp(parvals['dG']/alpha))**2))
    jcb.loc['fmin'] = 1
    
    return jcb.values.astype(float)

def fitSingleBindingCurve(concentrations, fluorescence, fitParameters, func=None,
                          errors=None, plot=None, log_axis=None):
    if plot is None:
        plot = False
    if log_axis is None:
        log_axis = True
    if func is None:
        func = bindingCurveObjectiveFunction
    param_names = fitParameters.columns.tolist()
    params = Parameters()
    for param in param_names:
        if 'vary' in fitParameters.loc[:, param].index:
            vary = fitParameters.loc['vary', param]
        else:
            vary = True
        params.add(param, value=fitParameters.loc['initial', param],
                   min = fitParameters.loc['lowerbound', param],
                   max = fitParameters.loc['upperbound', param],
                   vary= vary)
    
    if errors is not None:
        eminus, eplus = errors
        weights = 1/(eminus+eplus)
        if np.isnan(weights).all():
            weights = None
    else:
        eminus, eplus = [[np.nan]*len(concentrations)]*2
        weights = None
        
    results = minimize(func, params,
                       args=(concentrations,),
                       kws={'data':fluorescence, 'weights':weights},
                       xtol=1E-6, ftol=1E-6, maxfev=10000)
    
    # find rqs
    ss_total = np.sum((fluorescence - fluorescence.mean())**2)
    ss_error = np.sum((results.residual)**2)
    rsq = 1-ss_error/ss_total
    rmse = np.sqrt(ss_error)
    
    # plot binding curve
    if plot:
        try:
            print 'weighted fit:'
            print(report_fit(params))
            plt.figure(figsize=(4,4))
            if log_axis:
                ax = plt.gca()
                ax.set_xscale('log')
                more_concentrations = np.logspace(np.log10(concentrations.min()/10),
                                                  np.log10(concentrations.max()*10),
                                                  100)
            else:
                more_concentrations = np.linspace(concentrations.min(),
                                                  concentrations.max(), 100)                

            plt.errorbar(concentrations, fluorescence, yerr=[eminus, eplus], fmt='.',
                         elinewidth=1, capsize=2, capthick=1, color='k', linewidth=1)
            plt.plot(more_concentrations, func(params, more_concentrations), 'b',
                     label='weighted fit')

            plt.legend(loc='upper left')
        except:pass 
    
    # save params in structure
    final_params = pd.Series(index=param_names+['%s_stde'%param for param in param_names])
    for param in param_names:
        final_params.loc[param] = params[param].value
        final_params.loc['%s_stde'%param] = params[param].stderr
    final_params.loc['rsq'] = rsq
    final_params.loc['exit_flag'] = results.ier
    final_params.loc['rmse'] = rmse
    
    return final_params

    

def findErrorBarsBindingCurve(subSeries):
    eminus, eplus = np.asarray([np.abs(subSeries.loc[:, i].median() -
                                       bootstrap.ci(subSeries.loc[:, i], np.median, n_samples=1000))
                                for i in subSeries]).transpose()
    return eminus, eplus 

def bootstrapCurves(subSeries, fitParameters, concentrations, parameters,
                    default_errors=None, plot=None,plot_dists=None, n_samples=None,
                    eps=None, min_fraction_railed=None):
    # set defaults for various parameters
    if n_samples is None:
        n_samples = 100
    if eps is None:
        eps = (fitParameters.loc['upperbound', 'fmax'] -
               fitParameters.loc['lowerbound', 'fmax'])/100.
    if min_fraction_railed is None:
        min_fraction_railed = 0.75
    if plot is None:
        plot = False
    if plot_dists is None:
        plot_dists = plot
    
    # estimate weights to use in weighted least squares fitting
    numTests = len(subSeries)
    if default_errors is None:
        default_errors = np.ones(len(concentrations))*np.nan
    if numTests > 2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eminus, eplus = findErrorBarsBindingCurve(subSeries)
        except:
            eminus = eplus = default_errors/np.sqrt(numTests)
    else:
        eminus = eplus = default_errors/np.sqrt(numTests)
    
    # find number of samples to bootstrap
    if numTests <10 and np.power(numTests, numTests) <= n_samples:
        # then do all possible permutations
        if plot_dists: print 'Doing all possible %d product of indices'%np.power(numTests, numTests)
        indices = [list(i) for i in itertools.product(*[subSeries.index]*numTests)]
    else:
        if plot_dists:
            print 'making %d randomly selected (with replacement) bootstrapped median binding curves'%n_samples
        indices = np.random.choice(subSeries.index, size=(n_samples, len(subSeries)), replace=True)
  
    # if last point in binding series is below fmax constraints, do by method B
    median_fluorescence = subSeries.median()
    flag = 0
    if median_fluorescence[-1] < fitParameters.loc['lowerbound', 'fmax']:
        redoFitFmax = True
        flag = 1
        if plot_dists:
            print ('last concentration is below lb for fmax '+
                   '(%4.2f out of %4.2f (%d%%). Doing bootstrapped fit with fmax'+
                   'samples from dist')%(
                median_fluorescence[-1],
                fitParameters.loc['lowerbound', 'fmax'],
                median_fluorescence[-1]/fitParameters.loc['lowerbound', 'fmax']*100)
    else:
        redoFitFmax = False
        if plot_dists:
            print ('last concentration is above lb for fmax (%4.2f out of %4.2f '+
                   '(%d%%). Proceeding by varying fmax')%(
                median_fluorescence[-1],
                fitParameters.loc['lowerbound', 'fmax'],
                median_fluorescence[-1]/fitParameters.loc['lowerbound', 'fmax']*100)
    

    if not redoFitFmax:
        singles = {}
        for i, clusters in enumerate(indices):
            if plot_dists:
                if i%(n_samples/10.)==0:
                    print 'working on %d out of %d, %d%%'%(i, n_samples, i/float(n_samples)*100)
            fluorescence = subSeries.loc[clusters].median()
            singles[i] = fitSingleBindingCurve(concentrations, fluorescence, fitParameters, errors=[eminus, eplus], plot=False)
            
            # check if after 100 tries, the fmax_ub is railed.
            if i==min(49, n_samples-1):
                current_data = pd.concat(singles, axis=1).transpose()
                fit_fmaxes = current_data.fmax
                fit_dG = pd.concat(singles, axis=1).transpose().dG.median()
                
                num_railed = ((np.abs(fit_fmaxes-fitParameters.loc['upperbound', 'fmax']) < eps).sum() +
                              (np.abs(fit_fmaxes-fitParameters.loc['lowerbound', 'fmax']) < eps).sum())
                num_tests = len(fit_fmaxes)
                
                if (num_railed/float(num_tests) > min_fraction_railed):
                    # problem encountered. do everything again with set fmax.
                    if plot_dists:
                        print 'After %d iterations, cannot find fmax.'%(num_tests)
                        print '\tMedian fit dG = %4.2f'%(fit_dG)
                        print '\t%d (%d%%) of fits are within eps of bounds.'%(num_railed, num_railed/float(num_tests)*100)
                        print 'Starting bootstrapped fit again with fmax sampled from dist'
                    redoFitFmax = True
                    flag = 2
                    break
                else:
                    if plot_dists:
                        print 'After %d iterations: median fmax is %4.2f. median dG is %4.2f'%(num_tests, fit_fmaxes.median(), fit_dG)
                        print '\t%d (%d%%) of fits are within eps of bounds.'%(num_railed, num_railed/float(num_tests)*100)
                    redoFitFmax = False

    if redoFitFmax:        
        singles = {}
        fitParametersNew = fitParameters.copy()
        if 'vary' not in fitParametersNew.index:
            fitParametersNew.loc['vary'] = True
        fitParametersNew.loc['vary', 'fmax'] = False
        fmaxes = parameters.find_fmax_bounds_given_n(numTests, return_dist=True).rvs(n_samples)
        for i, (clusters, fmax) in enumerate(itertools.izip(indices, fmaxes)):
            if plot_dists:
                if i%(n_samples/10.)==0:
                    print 'working on %d out of %d, %d%%'%(i, n_samples, i/float(n_samples)*100)
            fluorescence = subSeries.loc[clusters].median()
            fitParametersNew.loc['initial', 'fmax'] = fmax
            singles[i] = fitSingleBindingCurve(concentrations, fluorescence, fitParametersNew, errors=[eminus, eplus], plot=False)

    singles = pd.concat(singles, axis=1).transpose()
    
    param_names = ['fmax', 'dG', 'fmin']
    not_outliers = ~seqfun.is_outlier(singles.dG)
    data = np.ravel([[np.percentile(singles.loc[not_outliers, param], idx) for param in param_names] for idx in [50, 2.5, 97.5]])
    index = param_names + ['%s_lb'%param for param in param_names] + ['%s_ub'%param for param in param_names]
    results = pd.Series(index = index, data=data)
    
    # get rsq
    params = Parameters()
    for param in param_names:
        params.add(param, value=results.loc[param])
        
    ss_total = np.sum((median_fluorescence - median_fluorescence.mean())**2)
    ss_error = np.sum((median_fluorescence - bindingCurveObjectiveFunction(params, concentrations))**2)
    results.loc['rsq']  = 1-ss_error/ss_total

    # save some parameters
    results.loc['numClusters'] = numTests
    results.loc['numIter'] = len(singles)
    results.loc['fractionOutlier'] = 1 - not_outliers.sum()/float(len(singles))
    results.loc['flag'] = flag
    

    if plot:
        if plot_dists:
            # plot histogram of parameters
            for param in param_names:
                if fitParameters.loc['vary', param]:
                    plt.figure(figsize=(4,3))
                    sns.distplot(singles.loc[:, param].dropna().values, hist_kws={'histtype':'stepfilled'}, color='b')
                    ax = plt.gca()
                    ylim = ax.get_ylim()
                    plt.plot([results.loc[param]]*2, ylim, 'k--', alpha=0.5)
                    plt.plot([results.loc['%s_lb'%param]]*2, ylim, 'k:', alpha=0.5)
                    plt.plot([results.loc['%s_ub'%param]]*2, ylim, 'k:', alpha=0.5)
                    plt.ylabel('prob density')
                    plt.xlabel(param)
                    plt.tight_layout()
           
        more_concentrations = np.logspace(-2, 4, 50)
        plt.figure(figsize=(4,4))
        plt.errorbar(concentrations, subSeries.median(), yerr=[eminus, eplus], fmt='.', elinewidth=1, capsize=2, capthick=1, color='k', linewidth=1)
        plt.plot(more_concentrations, bindingCurveObjectiveFunction(params, more_concentrations), 'b', label='weighted fit')
        # lower bound
        params_lb = Parameters()
        params_lb.add('fmax', value=np.percentile(singles.loc[:, 'fmax'], 97.5))
        params_lb.add('dG', value=np.percentile(singles.loc[:, 'dG'], 2.5))
        params_lb.add('fmin', value=np.percentile(singles.loc[:, 'fmin'], 97.5))
        lowerbound = bindingCurveObjectiveFunction(params_lb, more_concentrations)

        params_ub = Parameters()
        params_ub.add('fmax', value=np.percentile(singles.loc[:, 'fmax'], 2.5))
        params_ub.add('dG', value=np.percentile(singles.loc[:, 'dG'], 97.5))
        params_ub.add('fmin', value=np.percentile(singles.loc[:, 'fmin'], 2.5))
        upperbound = bindingCurveObjectiveFunction(params_ub, more_concentrations)
        
        plt.fill_between(more_concentrations, lowerbound, upperbound, color='0.5', label='95% conf int', alpha=0.5)
        
        #plt.plot(more_concentrations, bindingCurveObjectiveFunction(params2, more_concentrations), 'r', label='unweighted')
        ax = plt.gca()
        ax.set_xscale('log')
        plt.xlabel('concentration (nM)')
        plt.ylabel('normalized fluorescence')
        plt.tight_layout()
    return results

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

def plotFmaxMinDist(fDist, params, ax=None, color=None):
    fDist.dropna(inplace=True)
    fmax_lb, fmax_initial, fmax_upperbound = params
    if ax is None:
        fig = plt.figure(figsize=(4,3));
        ax = fig.add_subplot(111)
    if color is None:
        color = 'r'
    sns.distplot(fDist, color=color, hist_kws={'histtype':'stepfilled'}, ax=ax)
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

def getBoundsGivenDistribution(values, label=None, saturation_level=None,
                               use_max_prob=None):
    if saturation_level is None:
        saturation_level = 1 # i.e. assume these binders are 100% bound
    if use_max_prob is None:
        use_max_prob = False
        
    fitParameters = pd.Series(index=['lowerbound', 'initial', 'upperbound'])
    fDist = seqfun.remove_outlier(values)
    fitParameters.loc['lowerbound'] = fDist.min()
    fitParameters.loc['upperbound'] = fDist.max()/saturation_level
    if use_max_prob:
        maxProb = findMaxProbability(fDist)
        if maxProb is not None:
            fitParameters.loc['initial'] = maxProb
        else:
            fitParameters.loc['initial'] = fDist.median()
    else:
        fitParameters.loc['initial'] = fDist.median()
    ax = plotFmaxMinDist(fDist, fitParameters);
    if label is not None:
        ax.set_xlabel(label)
    return fitParameters
    
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
    params = Parameters()

    if use_actual:
        params.add('sigma', value=stds_actual.iloc[0], min=0)
        params.add('c',     value=stds_actual.min(),   min=0)
        minimize(parameters.sigma_by_n_fit, params,
                                       args=(stds_actual.index,),
                                       kws={'y':stds_actual.values,
                                            'weights':weights},
                                       xtol=1E-6, ftol=1E-6, maxfev=10000)
    else:
        params.add('sigma', value=stds.iloc[0], min=0)
        params.add('c',     value=stds.min(),   min=0)
        minimize(parameters.sigma_by_n_fit, params,
                                       args=(stds.index,),
                                       kws={'y':stds.values,
                                            'weights':None},
                                       xtol=1E-6, ftol=1E-6, maxfev=10000)
        params.add('min_sigma', value=tight_binders.fmax.std(), vary=False)
        
        # plot binders
        plt.figure(figsize=(3,3))
        sns.distplot(tight_binders.fmax, bins=10, hist_kws={'histtype':'stepfilled'},
                     color='grey')
        plt.xlabel('median fmax of tight binders')
        plt.ylabel('number of variants')
        plt.tight_layout()
        ax = plt.gca()
        ax.tick_params(right='off', top='off')
    # plot data
    plt.figure(figsize=(4,3));
    
    if use_actual:
        plt.scatter(stds_actual.index, stds_actual, color='k', marker='o',
                    label='actual');
        plt.plot(stds.index, stds, 'r:', label='simulated')
        plt.plot(stds_actual.index, parameters.sigma_by_n_fit(params, stds_actual.index), 'c',
                 label='fit')
        max_x = np.max(stds_actual.index.tolist())
    else:
        plt.scatter(stds.index, stds, color='k', marker='o', label='simulated');
        plt.scatter(stds_actual.index, stds_actual,  color='r', marker='x', label='actual');
        plt.plot(stds.index, parameters.sigma_by_n_fit(params, stds.index), 'c', label='fit')
        plt.plot([0, np.max(stds.index.tolist())], [params['min_sigma'].value]*2, 'r:')
        max_x = np.max(stds.index.tolist())
   
    # plot fit
    ax = plt.gca()
    ax.tick_params(right='off', top='off')
    ax.set_position([0.2, 0.2, 0.5, 0.75])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    plt.xlim(0, max_x)
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

def onRateByVariant(reducedCPsignaFile, times, annotatedClusterFile):
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
            reducedCPsignalFile)
    
    allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    table = pd.concat([pd.read_table(annotatedClusterFile, index_col='tileID',
                            usecols=['tileID', 'variant_number']),
                        bindingSeriesNorm], axis=1)
    
    grouped = table.groupby('variant_number')
    variants = np.unique(table.variant_number)
    print '\tDividing table into groups...'
    actual_variants = []
    groups = []
    for name, group in grouped:
        if name in variants:
            actual_variants.append(name)
            groups.append(group)
    return
