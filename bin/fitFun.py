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
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1})

class fittingParameters():
    """
    stores some parameters and functions
    """
    def __init__(self, concentrations=None, params=None, fitParameters=None,
                 default_errors=None):

        
        # save the units of concentration given in the binding series
        self.concentration_units = 1E-9 # i.e. nM
        self.RT = 0.582
        
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
        self.cutoff_kd = 5000
        self.cutoff_dG = self.find_dG_from_Kd(self.cutoff_kd)

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
    
        
def bindingCurveObjectiveFunction(params, concentrations, data=None, weights=None):
    """  Return fit value, residuals, or weighted residuals of a binding curve.
    
    Hill coefficient 1. """

    parameters = fittingParameters()
    
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    dG   = parvals['dG']
    fmin = parvals['fmin']
    fracbound = (fmin + fmax*concentrations/
                 (concentrations + np.exp(dG/parameters.RT)/
                  parameters.concentration_units))
    
    # return fit value of data is not given
    if data is None:
        return fracbound
    
    # return residuals if data is given
    elif weights is None:
        return fracbound - data
    
    # return weighted residuals if data is given
    else:
        return (fracbound - data)*weights
    
def fitSingleCurve(x, fluorescence, fitParameters, func=None,
                          errors=None, plot=None, log_axis=None, do_not_fit=None):
    """ Fit an objective function to data, weighted by errors. """
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
        eminus, eplus = [[np.nan]*len(x)]*2
        weights = None
    
    # make sure fluorescence doesn't have NaN terms
    index = np.array(np.isfinite(fluorescence))
    x = x[index]
    fluorescence = fluorescence[index]
    if weights is not None:
        weights = weights[index]
    
    # do the fit
    results = minimize(func, params,
                       args=(x,),
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
    """ Return bootstrapped confidence intervals on columns of an input data matrix.
    
    Assuming rows represent replicate measurments, i.e. clusters. """
    try:
        # bootstrap medians 
        eminus, eplus = np.asarray([np.abs(subSeries.loc[:, i].dropna().median() -
                                           bootstrap.ci(subSeries.loc[:, i].dropna(),
                                                        np.median, n_samples=1000))
                                    for i in subSeries]).transpose()
    except:
        # if boostrapping fails, return NaN array
        eminus, eplus = [np.ones(subSeries.shape[1])*np.nan]*2
    return eminus, eplus

def enforceFmaxDistribution(median_fluorescence, fmaxDist, verbose=None, cutoff=None):
    """ Decide whether to enforce fmax distribution (on binding curves) or let it float.
    
    Cutoff is whether the last point of the (median) fluorescence is above the
    lower bound for fmax. """
    
    if verbose is None:
        verbose = False
    
    if cutoff is None:
        cutoff = 0.025 # only 2.5% of distribution falls beneath this value
    
    lowerbound = fmaxDist.ppf(cutoff)
    
    median_fluorescence = median_fluorescence.astype(float).values
    if median_fluorescence[-1] < lowerbound:
        redoFitFmax = True
        if verbose:
            print (('last concentration is below lb for fmax (%4.2f out of '
                   '%4.2f (%d%%). Doing bootstrapped fit with fmax'
                   'samples from dist')
                %(median_fluorescence[-1], lowerbound,
                  median_fluorescence[-1]*100/lowerbound))
    else:
        redoFitFmax = False
        if verbose:
            print (('last concentration is above lb for fmax (%4.2f out of %4.2f '+
                   '(%d%%). Proceeding by varying fmax')
                %(median_fluorescence[-1], lowerbound,
                  median_fluorescence[-1]*100/lowerbound))
    return redoFitFmax


def bootstrapCurves(x, subSeries, fitParameters, fmaxDist=None,
                    default_errors=None, use_default=None, verbose=None, n_samples=None,
                    enforce_fmax=None, func=None):
    """ Bootstrap fit of a model to multiple measurements of a single molecular variant. """
    
    # set defaults for various parameters
    if n_samples is None:
        n_samples = 100
        
    if verbose is None:
        verbose = False

    # if last point in binding series is below fmax constraints, do by method B
    median_fluorescence = subSeries.median()
        
    if enforce_fmax is None:
        # if enforce_fmax is not set, decide based on median fluorescence in last binding point
        enforce_fmax = enforceFmaxDistribution(median_fluorescence,
                                               fmaxDist, verbose=verbose)
    else:
        if verbose:
            print "using enforced fmax because of user settings"
            
    if enforce_fmax and (fmaxDist is None):
        print ('Error: if you wish to enforce fmax, need to define "fmaxDist"\n'
               'which is a instance of a normal distribution with mean and sigma\n'
               'defining the expected distribution of fmax')
        
    if use_default is None:
        use_default = False # if flagged, use only default errors
    
    # if func is not given, assume fit to binding curve.
    if func is None:
        func = bindingCurveObjectiveFunction
        
    # estimate weights to use in weighted least squares fitting
    if default_errors is None:
        default_errors = np.ones(len(x))*np.nan
    if not use_default:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eminus, eplus = findErrorBarsBindingCurve(subSeries)
        except:
            use_default=True
            
    # option to use only default errors provdided for quicker runtime
    if use_default:
        numTestsAny = np.array([len(subSeries.loc[:, col].dropna()) for col in subSeries])
        eminus = eplus = default_errors/np.sqrt(numTestsAny)
    
    # find number of samples to bootstrap
    numTests = len(subSeries)
    if numTests <10 and np.power(numTests, numTests) <= n_samples:
        # then do all possible permutations
        if verbose:
            print ('Doing all possible %d product of indices'
                   %np.power(numTests, numTests))
        indices = [list(i) for i in itertools.product(*[subSeries.index]*numTests)]
    else:
        # do at most 'n_samples' number of iterations
        if verbose:
            print ('making %4.0f randomly selected (with replacement) '
                   'bootstrapped median binding curves')%n_samples
        indices = np.random.choice(subSeries.index,
                                   size=(n_samples, len(subSeries)), replace=True)

    # Enforce fmax if initially told to and cutoff was not met
    fitParameters = fitParameters.copy()
    if enforce_fmax:
        # make sure fmax does not vary and find random variates
        # of fmax distribution
        if 'vary' not in fitParameters.index:
            fitParameters.loc['vary'] = True
        fitParameters.loc['vary', 'fmax'] = False
        fmaxes = fmaxDist.rvs(n_samples)
    
    # proceed with bootstrapping
    singles = {}
    for i, clusters in enumerate(indices):
        if verbose:
            if i%(n_samples/10.)==0:
                print 'working on %d out of %d, %d%%'%(i, n_samples, i/float(n_samples)*100)
        if enforce_fmax:
            fitParameters.loc['initial', 'fmax'] = fmaxes[i]
        
        # find median fluorescence
        fluorescence = subSeries.loc[clusters].median()
        
        # only fit if at least 3 measurements
        index = np.isfinite(fluorescence)
        if index.sum() <= 3:
            do_not_fit = True
        else:
            do_not_fit = False
        
        # do an iteration of fitting
        singles[i] = fitSingleCurve(x[index.values],
                                    fluorescence.loc[index],
                                    fitParameters,
                                    errors=[eminus[index.values], eplus[index.values]],
                                    plot=False,
                                    func=func,
                                    do_not_fit=do_not_fit)
    # concatenate all resulting iterations
    singles = pd.concat(singles, axis=1).transpose()
    
    # save results
    param_names = fitParameters.columns.tolist()
    data = np.hstack([np.percentile(singles.loc[:, param], [50, 2.5, 97.5])
                       for param in param_names])
    index = np.hstack([['%s%s'%(param_name, s) for s in ['', '_lb', '_ub']]
                       for param_name in param_names])
    results = pd.Series(index=index, data=data)
    
    # get rsq
    params = Parameters()
    for param in param_names:
        params.add(param, value=results.loc[param])
        
    ss_total = np.sum((median_fluorescence - median_fluorescence.mean())**2)
    ss_error = np.sum((median_fluorescence - func(params, x))**2)
    results.loc['rsq']  = 1-ss_error/ss_total

    # save some parameters
    results.loc['numClusters'] = numTests
    results.loc['numIter'] = (singles.exit_flag > 0).sum()
    results.loc['flag'] = enforce_fmax
    
    return results, singles

def plotFitDistributions(results, singles, fitParameters):
    """ Plot a distribtion of fit parameters. """
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

def returnParamsFromResults(final_params, param_names=None):
    # make a parameters structure that works for objective functions
    if param_names is None:
        param_names = ['fmax', 'dG', 'fmin']
    params = Parameters()
    for param in param_names:
        params.add(param, value=final_params.loc[param])
    return params

def plotFitCurve(concentrations, subSeries, results,
                          fitParameters, log_axis=None, func=None, use_default=None,
                          fittype=None, errors=None, default_errors=None, ax=None):
    # default is to log axis
    if log_axis is None:
        log_axis = True
        
    # default is binding curve
    if func is None:
        func = bindingCurveObjectiveFunction
    
    if fittype is None:
        fittype = 'binding'
    
    if use_default is None:
        use_default = False

    if default_errors is None:
        default_errors = np.ones(len(concentrations))*np.nan

    if len(subSeries.shape) == 1:
        fluorescence = subSeries
        use_default = True
        numTests = np.array([1 for col in subSeries])
    else:
        fluorescence = subSeries.median()
        numTests = np.array([len(subSeries.loc[:, col].dropna()) for col in subSeries])
    
    # option to use only default errors provdided for quicker runtime
    if not use_default:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eminus, eplus = findErrorBarsBindingCurve(subSeries)
        except:
            use_default=True
        if np.all(np.isnan(eminus)) or np.all(np.isnan(eplus)):
            use_default=True
            
    if use_default:
        eminus = eplus = default_errors/np.sqrt(numTests)
    
    # plot binding points
    if ax is None:
        fig = plt.figure(figsize=(2.5,2.3));
        ax = fig.add_subplot(111)
    ax.errorbar(concentrations, fluorescence,
                 yerr=[eminus, eplus], fmt='.', elinewidth=1,
                 capsize=2, capthick=1, color='k', linewidth=1)
    
    # plot fit
    if log_axis:
        ax.set_xscale('log')
        more_concentrations = np.logspace(np.log10(concentrations.min()/2),
                                          np.log10(concentrations.max()*2),
                                          100)
    else:
        more_concentrations = np.linspace(concentrations.min(),
                                          concentrations.max(), 100)
    param_names = fitParameters.columns.tolist()
    params = returnParamsFromResults(results, param_names)
    fit = func(params, more_concentrations)
    ax.plot(more_concentrations, fit, 'r')

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

    ax.tick_params(right='off', top='off')
    ax.tick_params(which="minor", right='off', top='off')
    ylim = ax.get_ylim()
    plt.ylim(0, ylim[1])
    plt.xlim(more_concentrations[[0, -1]])
    if fittype=='binding':
        plt.xlabel('concentration (nM)')
    else:
        plt.xlabel('time (s)')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('normalized fluorescence')
    plt.subplots_adjust(bottom=0.26, left=0.26, top=0.97)

def errorPropagationKdFromKoffKobs(koff, kobs, c, sigma_koff, sigma_kobs):
    koff = koff.astype(float)
    kobs = kobs.astype(float)
    sigma_koff = sigma_koff.astype(float)
    sigma_kobs = sigma_kobs.astype(float)
    sigma_kd = np.sqrt(
        (( c*kobs/(kobs-koff)**2)*sigma_koff)**2 +
        ((-c*koff/(kobs-koff)**2)*sigma_kobs)**2)
    return sigma_kd

def errorProgagationKdFromdG(dG, sigma_dG):
    dG = dG.astype(float)
    sigma_dG = sigma_dG.astype(float)
    parameters = fittingParameters()
    sigma_kd = parameters.find_Kd_from_dG(dG)/parameters.RT*sigma_dG
    return sigma_kd
