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
                self.find_Kd_from_frac_bound_concentration(.95,
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
    
class fmaxDistAny():
    # for fitting stde of fmaxes
    def __init__(self, params=None):
        self.params = params
        
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
    fracbound = (fmin + fmax*concentrations/
                 (concentrations + np.exp(dG/parameters.RT)/
                  parameters.concentration_units))

    if data is None:
        return fracbound
    elif weights is None:
        return fracbound - data
    else:
        return (fracbound - data)*weights
    
def fitSingleCurve(concentrations, fluorescence, fitParameters, func=None,
                          errors=None, plot=None, log_axis=None, do_not_fit=None):
    if do_not_fit is None:
        do_not_fit = False # i.e. if you don't want to actually fit but still want to return a value
    if plot is None:
        plot = False
    if log_axis is None:
        log_axis = True
    if func is None:
        func = bindingCurveObjectiveFunction
    
    # fit parameters
    param_names = fitParameters.columns.tolist()
    
    # initiate output structure  
    index = (param_names + ['%s_stde'%param for param in param_names] +
             ['rsq', 'exit_flag', 'rmse'])
    final_params = pd.Series(index=index)

    # return here if you don't want to actually fit
    if do_not_fit:
        final_params.loc['exit_flag'] = -1
        return final_params
    
    # store fit parameters in class for fitting
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
    
    # weighted fit if errors are given
    if errors is not None:
        eminus, eplus = errors
        weights = 1/(eminus+eplus)
        if np.isnan(weights).any():
            weights = None
    else:
        eminus, eplus = [[np.nan]*len(concentrations)]*2
        weights = None
    
    # make sure fluorescence doesn't have NaN terms
    index = np.array(np.isfinite(fluorescence))
    concentrations = concentrations[index]
    fluorescence   = fluorescence[index]
    if weights is not None:
        weights[index]
    
    # do the fit
    results = minimize(func, params,
                       args=(concentrations,),
                       kws={'data':fluorescence, 'weights':weights},
                       xtol=1E-6, ftol=1E-6, maxfev=10000)
    
    # find rqs
    ss_total = np.sum((fluorescence - fluorescence.mean())**2)
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
    
    return final_params

def findErrorBarsBindingCurve(subSeries):
    try:
        eminus, eplus = np.asarray([np.abs(subSeries.loc[:, i].dropna().median() -
                                           bootstrap.ci(subSeries.loc[:, i].dropna(),
                                                        np.median, n_samples=1000))
                                    for i in subSeries]).transpose()
    except:
        eminus, eplus = [np.ones(subSeries.shape[1])*np.nan]*2
    return eminus, eplus

def enforceFmaxDistribution(median_fluorescence, fitParameters, verbose=None):
    # decide whether to enforce fmax distribution or let it float
    # cutoff is whether the last point of the (median) fluorescence is
    # above the lower bound for fmax.
    if verbose is None:
        verbose = False
    
    median_fluorescence = median_fluorescence.astype(float).values
    if median_fluorescence[-1] < fitParameters.loc['lowerbound', 'fmax']:
        redoFitFmax = True
        if verbose:
            print (('last concentration is below lb for fmax (%4.2f out of '
                   '%4.2f (%d%%). Doing bootstrapped fit with fmax'
                   'samples from dist')
                %(median_fluorescence[-1],
                  fitParameters.loc['lowerbound', 'fmax'],
                  median_fluorescence[-1]*100/
                    fitParameters.loc['lowerbound', 'fmax']))
    else:
        redoFitFmax = False
        if verbose:
            print (('last concentration is above lb for fmax (%4.2f out of %4.2f '+
                   '(%d%%). Proceeding by varying fmax')
                %(median_fluorescence[-1],
                  fitParameters.loc['lowerbound', 'fmax'],
                  median_fluorescence[-1]*100/
                    fitParameters.loc['lowerbound', 'fmax']))
    return redoFitFmax


def bootstrapCurves(concentrations, subSeries, fitParameters, fmaxDist=None,
                    default_errors=None, verbose=None, n_samples=None,
                    enforce_fmax=None, func=None):
    # set defaults for various parameters
    if n_samples is None:
        n_samples = 100
        
    if verbose is None:
        verbose = False

    if enforce_fmax is None:
        enforce_fmax = True # by default, enfoce fmax etc
    
    if enforce_fmax and fmaxDist is None:
        print ('Error: if you wish to enforce fmax, need to define "fmaxDist"\n'
               'which is a instance of a normal distribution with mean and sigma\n'
               'defining the expected distribution of fmax')
    
    if func is None:
        func = bindingCurveObjectiveFunction
    # estimate weights to use in weighted least squares fitting
    numTests = len(subSeries)
    if default_errors is None:
        default_errors = np.ones(len(concentrations))*np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eminus, eplus = findErrorBarsBindingCurve(subSeries)
    except:
        numTestsAny = np.array([len(subSeries.loc[:, col].dropna()) for col in subSeries])
        eminus = eplus = default_errors/np.sqrt(numTestsAny)

    
    # find number of samples to bootstrap
    if numTests <10 and np.power(numTests, numTests) <= n_samples:
        # then do all possible permutations
        if verbose:
            print ('Doing all possible %d product of indices'
                   %np.power(numTests, numTests))
        indices = [list(i) for i in itertools.product(*[subSeries.index]*numTests)]
    else:
        if verbose:
            print ('making %d randomly selected (with replacement) '
                   'bootstrapped median binding curves')%n_samples
        indices = np.random.choice(subSeries.index,
                                   size=(n_samples, len(subSeries)), replace=True)
  
    # if last point in binding series is below fmax constraints, do by method B
    median_fluorescence = subSeries.median()
    if enforce_fmax:
        enforce_fmax = enforceFmaxDistribution(median_fluorescence, fitParameters, verbose=verbose)
    
    # proceed with bootstrapping. Enforce fmax if initially told to and cutoff was not met
    fitParameters = fitParameters.copy()
    if enforce_fmax:
        # make sure fmax does not vary and find random variates
        # of fmax distribution
        if 'vary' not in fitParameters.index:
            fitParameters.loc['vary'] = True
        fitParameters.loc['vary', 'fmax'] = False
        fmaxes = fmaxDist.rvs(n_samples)

    singles = {}
    for i, clusters in enumerate(indices):
        if verbose:
            if i%(n_samples/10.)==0:
                print 'working on %d out of %d, %d%%'%(i, n_samples, i/float(n_samples)*100)
        if enforce_fmax:
            fitParameters.loc['initial', 'fmax'] = fmaxes[i]
        # fit single curve
        fluorescence = subSeries.loc[clusters].median()
        index = np.isfinite(fluorescence)
        if index.sum() > 3:
            singles[i] = fitSingleCurve(concentrations[index.values],
                                        fluorescence.loc[index],
                                        fitParameters,
                                        errors=[eminus[index.values], eplus[index.values]],
                                               plot=False,
                                               func=func)
        else:
            singles[i] = fitSingleCurve(concentrations[index.values],
                                        fluorescence.loc[index],
                                        fitParameters,
                                        do_not_fit=True)
    singles = pd.concat(singles, axis=1).transpose()
    
    param_names = fitParameters.columns.tolist()
    
    # I'm just not sure this is legit
    #not_outliers = ~seqfun.is_outlier(singles.dG)

    data = np.hstack([np.percentile(singles.loc[:, param], [50, 2.5, 97.5])
                       for param in param_names])
    index = np.hstack([['%s%s'%(param_name, s) for s in ['', '_lb', '_ub']]
                       for param_name in param_names])
                       
    results = pd.Series(index = index, data=data)
    
    # get rsq
    params = Parameters()
    for param in param_names:
        params.add(param, value=results.loc[param])
        
    ss_total = np.sum((median_fluorescence - median_fluorescence.mean())**2)
    ss_error = np.sum((median_fluorescence - func(params, concentrations))**2)
    results.loc['rsq']  = 1-ss_error/ss_total

    # save some parameters
    results.loc['numClusters'] = numTests
    results.loc['numIter'] = (singles.exit_flag > 0).sum()
    #results.loc['fractionOutlier'] = 1 - not_outliers.sum()/float(len(singles))
    results.loc['flag'] = enforce_fmax
    
    return results, singles

def plotFitDistributions(results, singles, fitParameters):
    for param in fitParameters.columns.tolist():
    
        plt.figure(figsize=(4,3))
        sns.distplot(singles.loc[:, param].dropna().values,
                     hist_kws={'histtype':'stepfilled'}, color='b')
        ax = plt.gca()
        ylim = ax.get_ylim()
        plt.plot([results.loc[param]]*2, ylim, 'k--', alpha=0.5)
        plt.plot([results.loc['%s_lb'%param]]*2, ylim, 'k:', alpha=0.5)
        plt.plot([results.loc['%s_ub'%param]]*2, ylim, 'k:', alpha=0.5)
        plt.ylabel('prob density')
        plt.xlabel(param)
        plt.tight_layout()
    return

def plotFitCurve(concentrations, bindingSeries, results,
                          fitParameters, log_axis=None, func=None,
                          fittype=None, errors=None, default_errors=None):
    # default is to log axis
    if log_axis is None:
        log_axis = True
        
    # default is binding curve
    if func is None:
        func = bindingCurveObjectiveFunction
    
    if fittype is None:
        fittype = 'binding'
        
    if len(bindingSeries.shape) == 1:
        fluorescence = bindingSeries
    else:
        fluorescence = bindingSeries.median()
    
    # get error
    numTests = np.array([len(bindingSeries.loc[:, col].dropna()) for col in bindingSeries])
    if errors is None:
        if default_errors is None:
            default_errors = np.ones(len(concentrations))*np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eminus, eplus = findErrorBarsBindingCurve(bindingSeries)
        except:
            eminus = eplus = default_errors/np.sqrt(numTests)
        if np.all(np.isnan(eminus)) or np.all(np.isnan(eplus)):
            eminus = eplus = default_errors/np.sqrt(numTests)
    
    # plot binding points
    plt.figure(figsize=(4,4));
    plt.errorbar(concentrations, fluorescence,
                 yerr=[eminus, eplus], fmt='.', elinewidth=1,
                 capsize=2, capthick=1, color='k', linewidth=1)
    
    # plot fit
    if log_axis:
        ax = plt.gca()
        ax.set_xscale('log')
        more_concentrations = np.logspace(np.log10(concentrations.min()/2),
                                          np.log10(concentrations.max()*2),
                                          100)
    else:
        more_concentrations = np.linspace(concentrations.min(),
                                          concentrations.max(), 100)
    param_names = fitParameters.columns.tolist()
    params = Parameters()
    for param in param_names:
        params.add(param, value=results.loc[param])
    fit = func(params, more_concentrations)
    plt.plot(more_concentrations, fit, 'r')

    try:
        # find upper bound
        params_ub = Parameters()
        if fittype == 'binding':
            ub_vec = ['_ub', '_lb', '']
            lb_vec = ['_lb', '_ub', '']
        elif fittype == 'off':
            ub_vec = ['_ub', '_lb', '_ub']
            lb_vec = ['_lb', '_ub', '_lb']
        elif fittype == 'on':
            ub_vec = ['_ub', '_ub', '_ub']
            lb_vec = ['_lb', '_lb', '_lb']

        for param in ['%s%s'%(param, suffix) for param, suffix in
                      itertools.izip(param_names, ub_vec)]:
            name = param.split('_')[0]
            params_ub.add(name, value=results.loc[param])
        ub = func(params_ub, more_concentrations)
    
        # find lower bound
        params_lb = Parameters()
        for param in ['%s%s'%(param, suffix) for param, suffix in
                      itertools.izip(param_names, lb_vec)]:
            name = param.split('_')[0]
            params_lb.add(name, value=results.loc[param])
        lb = func(params_lb, more_concentrations)
        
        # plot upper and lower bounds
        plt.fill_between(more_concentrations, lb, ub, color='0.5',
                         label='95% conf int', alpha=0.5)
    except:
        pass
    ax = plt.gca()
    ax.tick_params(right='off', top='off')
    plt.xlim(more_concentrations[[0, -1]])
    if fittype=='binding':
        plt.xlabel('concentration (nM)')
    else:
        plt.xlabel('time (s)')
    plt.ylabel('normalized fluorescence')
    plt.tight_layout()

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

def useSimulatedOrActual(variant_table, concentrations):
    # if at least 20 data points have at least 10 counts in that bin, use actual
    # data. This statistics seem reasonable for fitting
    parameters = fittingParameters(concentrations=concentrations)
    index = variant_table.dG_init < parameters.maxdG
    counts, binedges = np.histogram(variant_table.loc[index].numTests,
                                    np.arange(1, variant_table.numTests.max()))
    if (counts > 10).sum() >= 20:
        use_actual = True
    else:
        use_actual = False
    return use_actual

def plotSigmaByN(stds_actual, params, min_sigma=None):

    x = stds_actual.index
    y = stds_actual.values

    labels = ['actual', 'fit']
    fmt = ['ko', 'c']

    # plot data
    plt.figure(figsize=(4,3))
    plt.plot(x,       y,       fmt[0], label=labels[0]);
    
    fmaxDist = fmaxDistAny()
    x_fit = np.arange(1, x.max())
    y_fit = fmaxDist.sigma_by_n_fit(params, x_fit)
    plt.plot(x_fit,       y_fit,     fmt[1], label=labels[1]);
            
    # plot fit
    ax = plt.gca()
    ax.tick_params(right='off', top='off')
    ax.set_position([0.2, 0.2, 0.5, 0.75])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    plt.xlim(0, x.max())
    plt.ylim(0, ylim[-1])
    if min_sigma is not None:
        plt.plot(xlim, [min_sigma]*2, 'r:')
        
    # fit 1/sqrt(n)
    plt.xlabel('number of tests')
    plt.ylabel('std of fit fmaxes in bin')
    return
    

def findFinalBoundsParameters(variant_table, concentrations):
    parameters = fittingParameters(concentrations=concentrations)

    # actual data
    tight_binders = variant_table.loc[variant_table.dG_init <= parameters.maxdG]
    stds_actual = pd.Series(index=np.unique(tight_binders.numTests))
    weights     = pd.Series(index=np.unique(tight_binders.numTests))
    for n in stds_actual.index:
        stds_actual.loc[n] = seqfun.remove_outlier(
            tight_binders.loc[tight_binders.numTests==n].fmax_init).std()
        weights.loc[n] = np.sqrt((tight_binders.numTests==n).sum())
    stds_actual.dropna(inplace=True)
    weights = weights.loc[stds_actual.index]

    x = stds_actual.index
    y = stds_actual.values
    weights_fit = weights

    params = fitSigmaDist(x, y, weights=weights_fit)
    
    # save fitting parameters
    params.add('median', value=np.average(tight_binders.fmax_init,
                                          weights=np.sqrt(tight_binders.numTests)))
    
    plotSigmaByN(stds_actual, params)
        
    fmaxDist = fmaxDistAny(params=params)
    return fmaxDist

def findFinalBoundsParametersSimulated(variant_table, table, concentrations, return_vals=None):
    if return_vals is None:
        return_vals = False
    parameters = fittingParameters(concentrations)
    good_variants = (variant_table.dG_init < parameters.maxdG)
    tight_binders = variant_table.loc[good_variants]
    good_clusters = pd.Series(np.in1d(table.variant_number,
                                      variant_table.loc[good_variants].index),
                      index=table.index)
    other_binders = table.loc[good_clusters, ['fmax']]
    # for each n, choose n variants
    stds = pd.Series(index=np.arange(1, 101, 4))
    for n in stds.index:
        print n
        n_reps = np.ceil(float(len(other_binders))/n)
        index = np.random.permutation(np.tile(np.arange(n_reps), n))[:len(other_binders)]
        other_binders.loc[:, 'faux_variant'] = index
        stds.loc[n] = seqfun.remove_outlier(other_binders.groupby('faux_variant').
                                            median().fmax).std()
    stds.dropna(inplace=True)
    
    
    x = stds.index
    y = stds.values
    if return_vals:
        return x, y
    
    params = fitSigmaDist(x, y, weights=None)
    
    # save fitting parameters
    params.add('median', value=np.average(tight_binders.fmax_init,
                                          weights=np.sqrt(tight_binders.numTests)))
    min_sigma = seqfun.remove_outlier(tight_binders.fmax_init).std()
    params.add('min_sigma', value=min_sigma, vary=False)
    
    plotSigmaByN(stds, params, min_sigma=min_sigma)
        
    fmaxDist = fmaxDistAny(params=params)
    return fmaxDist

def fitSigmaDist(x, y, weights=None):
    fmaxDist = fmaxDistAny()
    params = Parameters()
    params.add('sigma', value=y.max(), min=0)
    params.add('c',     value=y.min(),   min=0)
    minimize(fmaxDist.sigma_by_n_fit, params,
                                   args=(x,),
                                   kws={'y':y,
                                        'weights':weights},
                                   xtol=1E-6, ftol=1E-6, maxfev=10000)
    return params

    # save fitting parameters
    params.add('median', value=np.average(tight_binders.fmax_init,
                                          weights=np.sqrt(tight_binders.numTests)))
    min_sigma = seqfun.remove_outlier(tight_binders.fmax_init).std()
    params.add('min_sigma', value=min_sigma, vary=False)
