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

def objectiveFunction(params, concentrations, data=None, weights=None):
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    dG   = parvals['dG']
    fmin = parvals['fmin']
    fracbound = fmax*concentrations/(concentrations+np.exp(dG/0.582)/1e-9)+fmin

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
    jcb.loc['dG']   = -beta*parvals['fmax']*concentrations*np.exp(parvals['dG']/alpha)/(alpha*(beta*concentrations+np.exp(parvals['dG']/alpha))**2)
    jcb.loc['fmin'] = 1
    
    return jcb.values.astype(float)

def fitSingleBindingCurve(concentrations, fluorescence, fitParameters, errors=None, plot=None, return_results=None):
    if plot is None:
        plot = False
    if return_results is None:
        return_results = False
    param_names = ['fmax', 'dG', 'fmin']
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
        
    results = minimize(objectiveFunction, params, args=(concentrations,), kws={'data':fluorescence, 'weights':weights}, xtol=1E-6, ftol=1E-6, maxfev=10000)
    
    # find rqs
    ss_total = np.sum((fluorescence - fluorescence.mean())**2)
    ss_error = np.sum((results.residual)**2)
    rsq = 1-ss_error/ss_total
    
    # plot binding curve
    if plot:
        print 'weighted fit:'
        print(report_fit(params))
        
        more_concentrations = np.logspace(-2, 4, 50)
        plt.figure(figsize=(4,4))
        plt.errorbar(concentrations, fluorescence, yerr=[eminus, eplus], fmt='.', elinewidth=1, capsize=2, capthick=1, color='k', linewidth=1)
        plt.plot(more_concentrations, objectiveFunction(params, more_concentrations), 'b', label='weighted fit')
        ax = plt.gca()
        ax.set_xscale('log')
        plt.legend(loc='upper left')
    
    # save params in structure
    final_params = pd.Series(index=param_names+['%s_stde'%param for param in param_names])
    for param in param_names:
        final_params.loc[param] = params[param].value
        final_params.loc['%s_stde'%param] = params[param].stderr
    final_params.loc['rsq'] = rsq
    final_params.loc['exit_flag'] = results.ier
    
    if return_results:
        #ci = conf_interval(results, sigmas=(0.95, 0.674))
        return results
    else:
        return final_params
    
def fitBindingCurveIfRailed(concentrations, fluorescence, fitParameters, errors=None, numTests=None, plot=None, numSamples=None):
    if plot is None:
        plot = False
    if numSamples is None:
        numSamples = 20
    # anyways given this empircal distirbution, sample from fmaxs
    
    fmax_sample = np.linspace(fitParameters.loc['lowerbound', 'fmax'],
                              fitParameters.loc['upperbound', 'fmax'], numSamples)
    results = {}
    for fmax_fixed in fmax_sample:
    
        fitParametersNew = fitParameters.copy()
        fitParametersNew.loc[:, 'fmax'] = [np.nan, fmax_fixed, np.nan, False]
        params = fitSingleBindingCurve(concentrations, fluorescence, fitParametersNew, errors=errors) 
        results[fmax_fixed] = params
        
    results = pd.concat(results, axis=1).transpose()
    weights = weightingFunction(fmax_sample, numTests)
    ds = DescrStatsW(results.dG, weights)
    
    if plot:
        xlim = [0, 2]
        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(211)
        ax.plot(fmax_sample, weights)
        ax.set_xlim(xlim)
        ax.set_ylabel('prob density')
    
        ax = fig.add_subplot(212)
        ax.scatter(fmax_sample, results.dG, c=weights, cmap='coolwarm')
        ax.set_xlim(xlim)
        ax.plot(xlim, [ds.mean]*2, 'k--', linewidth=1)
        lb, ub = ds.zconfint_mean()
        ax.plot(xlim, [lb]*2, '0.5', linewidth=1, linestyle=':')
        ax.plot(xlim, [ub]*2, '0.5', linewidth=1, linestyle=':')
        ax.set_ylabel('fit dG (kcal/mol)')
        ax.set_xlabel('fmax')
        ax.set_ylim(-12, -6)
        plt.subplots_adjust(left=0.15)
    
    return pd.Series(index=['dG', 'dG_stde'], data=[ds.mean, ds.std_mean*np.sqrt(numSamples)/np.sqrt(numTests)] )
    
    
def weightingFunction(fmax, numTests=None, mu=None, sigma=None):
    if sigma is None:
        sigma =  0.13677539
    if mu is None:
        mu = 0.9569374963533428
    if numTests is None:
        numTests = 1
    else:
        a, b, c = [0.13677539,  0.14555823,  0.06151002]
        sigma =  a*np.exp(-b*numTests)+c
    
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((fmax-mu)/(2*sigma))**2)
    
def processData():
    
    concentrationCols = IMlibs.formatConcentrations(concentrations)
    fitFilteredTable = IMlibs.filterFitParameters(IMlibs.filterStandardParameters(table))
    fitFilteredTable.loc[:, concentrationCols] = np.divide(fitFilteredTable.loc[:, concentrationCols], np.vstack(fitFilteredTable.all_cluster_signal))
    
    grouped = fitFilteredTable.groupby('variant_number')
    default_errors = grouped[concentrationCols].std().mean(axis=0)
    
    numTests = 6
    variantList = variant_table.dG.loc[variant_table.numTests*variant_table.fitFraction == numTests]
    variantList.sort()
    for variant in variantList.iloc[::310].index:
        subSeries = subtable.loc[subtable.variant_number==variant, concentrationCols]
        
        initial_points = variant_table.loc[variant, ['fmax', 'dG', 'fmin']]
        fitParameters.loc[:, 'fmax'] = parameters.find_fmax_bounds_given_n(len(subSeries))
    
        # set intial guesses for fmax and dG. Fmin stays!!
        for param_name in ['fmax', 'dG']:
            if (initial_points.loc[param_name] < fitParameters.loc['upperbound', param_name]-eps and
                initial_points.loc[param_name] > fitParameters.loc['lowerbound', param_name]+eps):
                print 'changing %s'%param_name
                fitParameters.loc['initial', param_name] = initial_points.loc[param_name]

        fitBindingCurve.bootstrapCurves(subSeries, fitParameters, concentrations,
                                        parameters,
                                    default_errors=parameters, plot=True,
                                    plot_dists=True, n_samples=100, eps=None, min_fraction_railed=None)

    variant = 16
    variant = 1


    plt.figure(); plt.errorbar(singles.dG, estimates.dG, yerr=estimates.dG_stde, xerr=singles.dG_stde, fmt='.', ecolor='k', color='w', alpha=0.5);
    plt.scatter(singles.dG, estimates.dG, c=[(fitFilteredTable.variant_number==variant).sum() for variant in vec.iloc[::780].index], vmin=1, vmax=10);
    plt.plot([-12, -6], [-12, -6])
    
def bootstrapCurves(subSeries, fitParameters, concentrations, parameters, default_errors=None, plot=None,plot_dists=None, n_samples=None, eps=None, min_fraction_railed=None, ):
    if n_samples is None:
        n_samples = 1000
    if eps is None:
        eps = (fitParameters.loc['upperbound', 'fmax'] - fitParameters.loc['lowerbound', 'fmax'])/100.
    if min_fraction_railed is None:
        min_fraction_railed = 0.25
    if plot is None:
        plot = False
    if plot_dists is None:
        plot_dists = plot
    
    # estimate weights to use in weighted least squares fitting
    concentrationCols = subSeries.columns
    numTests = len(subSeries)
    if numTests > 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eminus, eplus = np.asarray([np.abs(subSeries.loc[:, i].median() - bootstrap.ci(subSeries.loc[:, i], np.median, n_samples=1000)) for i in concentrationCols]).transpose()
    elif default_errors is None:
        eminus = eplus = np.ones(len(concentrations))*np.nan
    else:
        eminus = eplus = default_errors/np.sqrt(numTests)

    
    singles = {}
    if numTests <10 and np.power(numTests, numTests) <= n_samples:
        # then do all possible permutations
        if plot_dists: print 'Doing all possible %d product of indices'%np.power(numTests, numTests)
        indices = [list(i) for i in itertools.product(*[subSeries.index]*numTests)]
    else:
        if plot_dists:
            print 'making %d randomly selected (with replacement) bootstrapped median binding curves'%n_samples
        indices = np.random.choice(subSeries.index, size=(n_samples, len(subSeries)), replace=True)
    n_tests = len(indices)
    
    # if n_tests is less than 5
    for i, clusters in enumerate(indices):
        if plot_dists:
            if i%(n_tests/10.)==0: print 'working on %d out of %d, %d%%'%(i, n_tests, i/float(n_tests)*100)
        fluorescence = subSeries.loc[clusters].median()
        singles[i] = fitSingleBindingCurve(concentrations, fluorescence, fitParameters, errors=[eminus, eplus], plot=False)
        
        # check if after 100 tries, the fmax_ub is railed.
        if i==min(9, n_tests-1):
            current_fmaxes = pd.concat(singles, axis=1).transpose().fmax
            num_railed = (np.abs(current_fmaxes-fitParameters.loc['lowerbound', 'fmax']) < eps).sum() + (np.abs(current_fmaxes-fitParameters.loc['upperbound', 'fmax']) < eps).sum()
            num_tests = len(current_fmaxes)
            if num_railed/float(num_tests) > min_fraction_railed:
                # problem encountered. do everything again with set fmax.
                if plot_dists:
                    print 'After %d iterations, fmax is railed. \n\t%d (%d%%) of fits are within eps of bounds.'%(num_tests, num_railed, num_railed/float(num_tests)*100)
                    print 'Starting bootstrapped fit again with fmax sampled from dist'%fitParameters.loc['initial', 'fmax']
                redoFitFmax = True
                redoFitFmin = False
                break
            else:
                if plot_dists:
                    print 'After %d iterations: median fmax is %4.2f. \n\t%d (%d%%) of fits are within eps of bounds.'%(num_tests, current_fmaxes.median(), num_railed, num_railed/float(num_tests)*100)
                redoFitFmax = False
                redoFitFmin = False
            # now check fmins
            current_fmins = pd.concat(singles, axis=1).transpose().fmin
            num_railed = ((np.abs(current_fmins-fitParameters.loc['lowerbound', 'fmin']) < eps).sum() +
                (np.abs(current_fmins-fitParameters.loc['upperbound', 'fmin']) < eps).sum())
            num_tests = len(current_fmins)
            if num_railed/float(num_tests) > min_fraction_railed:
                if plot_dists:
                    print 'After %d iterations, fmin is railed. \n\t%d (%d%%) of fits are within eps of bounds.'%(num_tests, num_railed, num_railed/float(num_tests)*100)
                    print 'Starting bootstrapped fit again with fmin set to %4.3f'%fitParameters.loc['initial', 'fmin']
                redoFitFmin = True
                break
            else:
                if plot_dists:
                    print 'After %d iterations: median fmin is %4.2f. \n\t%d (%d%%) of fits are within eps of bounds.'%(num_tests, current_fmaxes.median(), num_railed, num_railed/float(num_tests)*100)
                redoFitFmin = False
    if redoFitFmax:        
        singles = {}
        fitParametersNew = fitParameters.copy()
        if 'vary' not in fitParametersNew.index:
            fitParametersNew.loc['vary'] = True
        fitParametersNew.loc['vary', 'fmax'] = False
        fmaxes = parameters.find_fmax_bounds_given_n(numTests, return_dist=True).rvs(n_tests)
        for i, (clusters, fmax) in enumerate(itertools.izip(indices, fmaxes)):
            if plot_dists:
                if i%(n_tests/10.)==0: print 'working on %d out of %d, %d%%'%(i, n_tests, i/float(n_tests)*100)
            fluorescence = subSeries.loc[clusters].median()
            fitParametersNew.loc['initial', 'fmax'] = fmax
            singles[i] = fitSingleBindingCurve(concentrations, fluorescence, fitParametersNew, errors=[eminus, eplus], plot=False)
    
    if redoFitFmin:
        singles = {}
        fitParametersNew = fitParameters.copy()
        fitParametersNew.loc['vary'] = True
        fitParametersNew.loc['vary', 'fmin'] = False
        for i, clusters in enumerate(indices):
            if plot_dists:
                if i%(n_tests/10.)==0: print 'working on %d out of %d, %d%%'%(i, n_tests, i/float(n_tests)*100)
            fluorescence = subSeries.loc[clusters].median()
            singles[i] = fitSingleBindingCurve(concentrations, fluorescence, fitParametersNew, errors=[eminus, eplus], plot=False)
    
             
    singles = pd.concat(singles, axis=1).transpose()
    
    param_names = ['fmax', 'dG', 'fmin']
    not_outliers = ~seqfun.is_outlier(singles.dG)
    data = np.ravel([[np.percentile(singles.loc[not_outliers, param], idx) for param in param_names] for idx in [50, 2.5, 97.5]])


    index = param_names + ['%s_lb'%param for param in param_names] + ['%s_ub'%param for param in param_names]
    results = pd.Series(index = index, data=data)
    results.loc['flag'] = '%d%d'%(redoFitFmax, redoFitFmin)
    if redoFitFmax:
        fmax_lb, fmax_initial, fmax_ub = parameters.find_fmax_bounds_given_n(numTests)
        results.loc[['fmax_lb', 'fmin_ub']] = [fmax_lb, fmax_ub]
 
    if plot:
        if plot_dists:
            # plot histogram of parameters
            for param in param_names:
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

        # also plot binding curve
        params = Parameters()
        for param in param_names:
            params.add(param, value=singles.loc[:, param].median())
                       
        more_concentrations = np.logspace(-2, 4, 50)
        plt.figure(figsize=(4,4))
        plt.errorbar(concentrations, subSeries.median(), yerr=[eminus, eplus], fmt='.', elinewidth=1, capsize=2, capthick=1, color='k', linewidth=1)
        plt.plot(more_concentrations, objectiveFunction(params, more_concentrations), 'b', label='weighted fit')
        # lower bound
        params_lb = Parameters()
        params_lb.add('fmax', value=np.percentile(singles.loc[:, 'fmax'], 97.5))
        params_lb.add('dG', value=np.percentile(singles.loc[:, 'dG'], 2.5))
        params_lb.add('fmin', value=np.percentile(singles.loc[:, 'fmin'], 97.5))
        lowerbound = objectiveFunction(params_lb, more_concentrations)

        params_ub = Parameters()
        params_ub.add('fmax', value=np.percentile(singles.loc[:, 'fmax'], 2.5))
        params_ub.add('dG', value=np.percentile(singles.loc[:, 'dG'], 97.5))
        params_ub.add('fmin', value=np.percentile(singles.loc[:, 'fmin'], 2.5))
        upperbound = objectiveFunction(params_ub, more_concentrations)
        
        plt.fill_between(more_concentrations, lowerbound, upperbound, color='0.5', label='95% conf int', alpha=0.5)
        
        #plt.plot(more_concentrations, objectiveFunction(params2, more_concentrations), 'r', label='unweighted')
        ax = plt.gca()
        ax.set_xscale('log')
        plt.xlabel('concentration (nM)')
        plt.ylabel('normalized fluorescence')
        plt.tight_layout()
    return results

    
def oldFunctionForPlottingFmax():
    # empirically, fmax follows exponential distribution a*exp(-b*x)+c, with a, b, c = [ 0.13677539,  0.14555823,  0.06151002]
    # from constrained fit on variant_table (normalized) after fit filters, with maxdG < 0.
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    binwidth = 1
    maxdG = -10
    vec = np.array([fitConstrained.loc[((variant_table.fitFraction*variant_table.numTests)>=i)&
        ((variant_table.fitFraction*variant_table.numTests)<i+binwidth)&
        (fitConstrained.dG < maxdG), 'fmax'].std()
                    for i in range(0, 50, binwidth)])
    popt, pcov = curve_fit(func, np.arange(0, 50, binwidth)[index], vec[index])
    plt.figure(figsize=(4,3));
    plt.plot(range(0, 50, binwidth), vec,  'ko');
    plt.plot(range(0, 50, binwidth), vec[1]/np.sqrt(np.arange(0, 50, binwidth)), 'r');
    plt.plot(np.arange(0, 50, binwidth)[index], func(np.arange(0, 50, binwidth)[index], *popt), 'grey')
    plt.xlabel('number of tests')
    plt.ylabel('std of fit fmaxes in bin')
    plt.tight_layout()
