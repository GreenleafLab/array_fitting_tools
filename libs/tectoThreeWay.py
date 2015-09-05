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
from fitFun import fittingParameters

 
def objectiveFunction(params, y=None, return_weighted=None, return_pred=None):
    if return_pred is None:
        return_pred = False
    if return_weighted is None:
        return_weighted = False
    parameters = fittingParameters()
    
    parvals = params.valuesdict()

    diff = y.dG.copy()
    for idx in y.index:
        length, seq, pos = idx
        
        term1 = parvals['bind_%d'%length]
        
        seq = list(seq)
        if pos == 0:
            term2 = parameters.RT*np.log(parvals['%s%s_%s'%(seq[0], seq[2], seq[1])])
        elif pos == 1:
            term2 = parameters.RT*np.log(parvals['%s%s_%s'%(seq[2], seq[1], seq[0])])
        elif pos == 2:
            term2 = parameters.RT*np.log(parvals['%s%s_%s'%(seq[1], seq[0], seq[2])])
            
        diff.loc[idx] = term1 + term2 
    
    if return_pred:
        return diff.astype(float)
    elif return_weighted:
        return ((diff - y.dG)*y.weight).astype(float)
    else:
        return (diff - y.dG).astype(float)

    
def fitThreeWay(y, weight=None):
    if weight is None: weight = False
    lengths = y.index.levels[0].tolist()
    seqs = y.index.levels[1].tolist()
    
    # store fit parameters in class for fitting
    params = Parameters()
    for length in lengths:
        params.add('bind_%d'%length, value=-9, 
                       min = -16,
                       max = -4)
    
    for seq in seqs:
        permute_0 = '%s%s_%s'%(seq[0], seq[2], seq[1])
        permute_1 = '%s%s_%s'%(seq[2], seq[1], seq[0])
        permute_2 = '%s%s_%s'%(seq[1], seq[0], seq[2])
        params.add(permute_0,
                       value=0.1, 
                       min = 0,
                       max = 1)
        params.add(permute_1,
                       value=0.1, 
                       min = 0,
                       max = 1)
        
        params.add(permute_2,
                       expr='1-%s-%s'%(permute_0, permute_1))
    
    func = objectiveFunction
    results = minimize(func, params,
                       args=(y.loc[y.variance > 0],),
                       kws={'return_weighted':weight})
    
    # find rsq
    ss_total = np.sum((y.dG - y.dG.mean())**2)
    ss_error = np.sum((results.residual)**2)
    rsq = 1-ss_error/ss_total
    rmse = np.sqrt(ss_error)
    
    ## plot residuals
    #plt.figure()
    #im = plt.scatter(y.dropna().dG, func(params, y=y.dropna(), return_pred=True),
    #            c = y.dropna().weight, cmap='coolwarm')
    #plt.colorbar(im)
    
    param_names = params.valuesdict().keys()
    index = (param_names + ['%s_stde'%param for param in param_names] +
             ['rsq', 'exit_flag', 'rmse'])
    final_params = pd.Series(index=index)
    for param in param_names:
        final_params.loc[param] = params[param].value
        final_params.loc['%s_stde'%param] = params[param].stderr
    final_params.loc['rsq'] = rsq
    final_params.loc['exit_flag'] = results.ier
    final_params.loc['rmse'] = rmse
    
    return final_params