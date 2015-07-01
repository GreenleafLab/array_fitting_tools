from lmfit import minimize, Parameters, report_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scikits.bootstrap import bootstrap

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

def fitSingleBindingCurve(concentrations, fluorescence, fitParameters, errors=None):
    params = Parameters()
    for param in ['fmax', 'dG', 'fmin']:
        params.add(param, value=fitParameters.loc['initial', param],
                   min = fitParameters.loc['lowerbound', param],
                   max = fitParameters.loc['upperbound', param],
                   vary=fitParameters.loc['vary', param])

    params2 = Parameters()
    for param in ['fmax', 'dG', 'fmin']:
        params2.add(param, value=fitParameters.loc['initial', param],
                   min = fitParameters.loc['lowerbound', param],
                   max = fitParameters.loc['upperbound', param],
                   vary=fitParameters.loc['vary', param])
    if errors is not None:
        eminus, eplus = errors
        weights = 1/(eminus+eplus)
    else:
        eminus, eplus = [[np.nan]*len(concentrations)]*2
        weights = None
    results_weighted = minimize(objectiveFunction, params, args=(concentrations,), kws={'data':fluorescence, 'weights':weights}, xtol=1E-6, ftol=1E-6, maxfev=10000)
    results = minimize(objectiveFunction, params2, args=(concentrations,), kws={'data':fluorescence}, xtol=1E-6, ftol=1E-6, maxfev=10000)

    #res1 = Minimizer(objectiveFunction, params2, fcn_args=(concentrations,), fcn_kws={'data':fluorescence, 'weights':weights})
    #out1 = res1.leastsq(Dfun=jacobianBindingCurve, col_deriv=1, xtol=1E-6, ftol=1E-6, maxfev=10000)
    
    print 'weighted fit:'
    print(report_fit(params))
    print ''
    print 'unweighted:'
    print(report_fit(params2))
    
    more_concentrations = np.logspace(-2, 4, 50)
    plt.figure(figsize=(4,4))
    plt.errorbar(concentrations, fluorescence, yerr=[eminus, eplus], fmt='.', elinewidth=1, capsize=2, capthick=1, color='k', linewidth=1)
    plt.plot(more_concentrations, objectiveFunction(params, more_concentrations), 'b', label='weighted fit')
    plt.plot(more_concentrations, objectiveFunction(params2, more_concentrations), 'r', label='unweighted')
    ax = plt.gca()
    ax.set_xscale('log')
    plt.legend(loc='upper left')
    
    final_params = pd.DataFrame(index=['value', 'stderr'], columns=['fmax', 'dG', 'fmin'])
    for param in final_params:
        final_params.loc['value', param] = params[param].value
        final_params.loc['stderr', param] = params[param].stderr
    return final_params
    
def fitBindingCurveIfRailed(concentrations, fluorescence, fitParameters, errors=None):
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
    
    # anyways given this empircal distirbution, sample from fmaxs
    
    fmax_sample = np.linspace(fitParameters.loc['lowerbound', 'fmax'],
                              fitParameters.loc['upperbound', 'fmax'],20)
    results = {}
    for fmax_fixed in fmax_sample:
    
        fitParametersNew = fitParameters.copy()
        fitParametersNew.loc[:, 'fmax'] = [fmax_fixed, fmax_fixed, fmax_fixed+0.01, False]
        print fmax_fixed
        params = fitBindingCurve.fitSingleBindingCurve(concentrations, fluorescence, fitParametersNew, errors=[eminus, eplus]) 
        results[fmax_fixed] = params.loc['value']
    
    
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
    bindingSeriesNormAll = np.divide(fitFilteredTable.loc[:, concentrationCols], np.vstack(fitFilteredTable.all_cluster_signal))
    
    variant = 45098
    variant = 16
    variant = 1
    subSeries = bindingSeriesNormAll.loc[fitFilteredTable.variant_number==variant]
    fluorescence = subSeries.median()
    eminus, eplus = np.asarray([subSeries.loc[:, i].median() - bootstrap.ci(subSeries.loc[:, i], np.median) for i in concentrationCols]).transpose()
    eplus = -eplus
    
    fitBindingCurve.fitSingleBindingCurve(concentrations, fluorescence, fitParameters, errors=[eminus, eplus])