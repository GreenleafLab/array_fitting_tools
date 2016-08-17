from lmfit import minimize, Parameters, report_fit, conf_interval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scikits.bootstrap import bootstrap
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
import itertools
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

def getInitialFitParameters(concentrations):
    """ Return fitParameters object with minimal constraints.
    
    Input: concentrations
    Uses concencetration to provide constraints on dG
    """
    parameters = fittingParameters(concentrations=concentrations)
    
    fitParameters = pd.DataFrame(index=['lowerbound', 'initial', 'upperbound'],
                                 columns=['fmax', 'dG', 'fmin'])
    
    # find fmin
    param = 'fmin'
    fitParameters.loc[:, param] = [0, 0, np.inf]

    # find fmax
    param = 'fmax'
    fitParameters.loc[:, param] = [0, np.nan, np.inf]
    
    # find dG
    fitParameters.loc[:, 'dG'] = [parameters.find_dG_from_Kd(
        parameters.find_Kd_from_frac_bound_concentration(frac_bound, concentration))
             for frac_bound, concentration in itertools.izip(
                                    [0.99, 0.5, 0.01],
                                    [concentrations[0], concentrations[-1], concentrations[-1]])]
 
    return fitParameters

def objectiveFunctionOffRates(params, times, data=None, weights=None, index=None, bleach_fraction=1, image_ns=None):
    """ Return fit value, residuals, or weighted residuals of off rate objective function. """
    if index is None:
        index = np.ones(len(times)).astype(bool)
    if image_ns is None:
        image_ns = np.arange(len(times))
        
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    koff = parvals['koff']
    fmin = parvals['fmin']
    fracbound = (fmin +
                 (fmax - fmin)*np.exp(-koff*times)*
                 np.power(bleach_fraction,image_ns))

    # return fit value of data is not given
    if data is None:
        return fracbound[index]
    
    # return residuals if data is given
    elif weights is None:
        return (fracbound - data)[index]
    
    # return weighted residuals if data is given
    else:
        return ((fracbound - data)*weights)[index]  
    
    
def objectiveFunctionOnRates(params, times, data=None, weights=None, index=None,  bleach_fraction=1, image_ns=None):
    """ Return fit value, residuals, or weighted residuals of on rate objective function. """
    if index is None:
        index = np.ones(len(times)).astype(bool)
    if image_ns is None:
        image_ns = np.arange(len(times))
        
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    koff = parvals['kobs']
    fmin = parvals['fmin']
    fracbound = fmin + (fmax*(1 - np.exp(-koff*times)*np.power(bleach_fraction,image_ns)));

    # return fit value of data is not given
    if data is None:
        return fracbound[index]
    
    # return residuals if data is given
    elif weights is None:
        return (fracbound - data)[index]
    
    # return weighted residuals if data is given
    else:
        return ((fracbound - data)*weights)[index]  
        
def bindingCurveObjectiveFunction(params, concentrations, data=None, weights=None, index=None,
                                  slope=0, fit_slope=False):
    """  Return fit value, residuals, or weighted residuals of a binding curve.
    
    Hill coefficient 1. """
    if index is None:
        index = np.ones(len(concentrations)).astype(bool)
        
    parameters = fittingParameters()
    
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    dG   = parvals['dG']
    fmin = parvals['fmin']
    if fit_slope:
        slope = parvals['slope']
    fracbound = (fmin + fmax*concentrations/
                 (concentrations + np.exp(dG/parameters.RT)/
                  parameters.concentration_units)) + slope*concentrations
    
    # return fit value of data is not given
    if data is None:
        return fracbound[index]
    
    # return residuals if data is given
    elif weights is None:
        return (fracbound - data)[index]
    
    # return weighted residuals if data is given
    else:
        return ((fracbound - data)*weights)[index]
    
def convertFitParametersToParams(fitParameters):
    """ Return lmfit params structure starting with descriptive dataframe. """
    param_names = fitParameters.columns.tolist()
    # store fit parameters in class for fitting
    params = Parameters()
    for param in param_names:
        if 'vary' in fitParameters.loc[:, param].index.tolist():
            vary = fitParameters.loc['vary', param]
        else:
            vary = True
        if 'lowerbound' in fitParameters.loc[:, param].index.tolist():
            lowerbound = fitParameters.loc['lowerbound', param]
        else:
            lowerbound = -np.inf
        if 'upperbound' in fitParameters.loc[:, param].index.tolist():
            upperbound = fitParameters.loc['upperbound', param]
        else:
            upperbound = np.inf
        params.add(param, value=fitParameters.loc['initial', param],
                   min = lowerbound,
                   max = upperbound,
                   vary= vary)
    return params

def fitSingleCurve(x, fluorescence, fitParameters, func,
                          errors=None, do_not_fit=None, kwargs=None):
    """ Fit an objective function to data, weighted by errors. """
    if do_not_fit is None:
        do_not_fit = False # i.e. if you don't want to actually fit but still want to return a value

    if kwargs is None:
        kwargs = {}
    
    # fit parameters
    params = convertFitParametersToParams(fitParameters)
    param_names = fitParameters.columns.tolist()
    
    # initiate output structure  
    index = (param_names + ['%s_stde'%param for param in param_names] +
             ['rsq', 'exit_flag', 'rmse'])
    final_params = pd.Series(index=index)

    # return here if you don't want to actually fit
    if do_not_fit:
        final_params.loc['exit_flag'] = -1
        return final_params
    
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
    kwargs = kwargs.copy()
    kwargs.update({'data':fluorescence, 'weights':weights, 'index':index}) 

    # do the fit
    results = minimize(func, params,
                       args=(x,),
                       kws=kwargs,
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
    eminus=[]
    eplus = [] 
    for i in subSeries:
        vec = subSeries.loc[:, i].dropna()
        success = True
        if len(vec) > 1:
            try:
                bounds = bootstrap.ci(vec, np.median, n_samples=1000)
            except IndexError:
                success = False
        else:
            success = False
            
        if success:
            eminus.append(vec.median() - bounds[0])
            eplus.append(bounds[1] - vec.median())
        else:
            eminus.append(np.nan)
            eplus.append(np.nan)
    eminus = pd.Series(eminus, index=subSeries.columns)
    eplus = pd.Series(eplus, index=subSeries.columns)

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
                    enforce_fmax=None, func=None, kwargs={}):
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
            if enforce_fmax:
                print "using enforced fmax because of user settings"
            else:
                print "not enforcing fmax because of user settings"
            
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
                                    func=func,
                                    do_not_fit=do_not_fit, kwargs=kwargs)
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

def fitSetClusters(concentrations, subBindingSeries, fitParameters, print_bool=None,
                   change_params=None, func=None, kwargs=None):
    """ Fit a set of binding curves. """
    if print_bool is None: print_bool = True

    #print print_bool
    singles = []
    for i, idx in enumerate(subBindingSeries.index):
        if print_bool:
            num_steps = max(min(100, (int(len(subBindingSeries)/100.))), 1)
            if (i+1)%num_steps == 0:
                print ('working on %d out of %d iterations (%d%%)'
                       %(i+1, len(subBindingSeries.index), 100*(i+1)/
                         float(len(subBindingSeries.index))))
                sys.stdout.flush()
        fluorescence = subBindingSeries.loc[idx]
        singles.append(perCluster(concentrations, fluorescence, fitParameters,
                                  change_params=change_params, func=func, kwargs=kwargs))

    return pd.concat(singles)

def perCluster(concentrations, fluorescence, fitParameters, plot=None, change_params=None, func=None,
               fittype=None, kwargs=None, verbose=False):
    """ Fit a single binding curve. """
    if plot is None:
        plot = False
    if func is None:
        func = bindingCurveObjectiveFunction
    if change_params is None:
        change_params = True
    try:
        if change_params:
            a, b = np.percentile(fluorescence.dropna(), [0, 100])
            fitParameters = fitParameters.copy()
            fitParameters.loc['initial', 'fmax'] = b
        #index = np.isfinite(fluorescence)
        fluorescence = fluorescence[:len(concentrations)]
        single = fitSingleCurve(concentrations,
                                                       fluorescence,
                                                       fitParameters, func, kwargs=kwargs)
    except IndexError as e:
        if verbose: print e
        print 'Error with %s'%fluorescence.name
        single = fitSingleCurve(concentrations,
                                                       fluorescence,
                                                       fitParameters, func,
                                                       do_not_fit=True)
    if plot:
        print "plotting.plotFitCurve(concentrations, fluorescence, single, param_names=fitParameters.columns.tolist(), func=func, fittype=fittype, kwargs=kwargs)"             
    return pd.DataFrame(columns=[fluorescence.name],
                        data=single).transpose()

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
    """ Given results, convert to lmfit params structure for fitting. """
    if param_names is None:
        param_names = ['fmax', 'dG', 'fmin']
    params = Parameters()
    for param in param_names:
        params.add(param, value=final_params.loc[param])
    return params

def returnParamsFromResultsBounds(final_params, param_names, ub_vec):
    params_ub = Parameters()
    for param in ['%s%s'%(param, suffix) for param, suffix in
                  itertools.izip(param_names, ub_vec)]:
        name = param.split('_')[0]
        params_ub.add(name, value=final_params.loc[param])
    return params_ub

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

