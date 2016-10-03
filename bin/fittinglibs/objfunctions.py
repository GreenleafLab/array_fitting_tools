import numpy as np
import pandas as pd


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
    
def powerlaw(params, x, y=None, weights=None, index=None):
    """"""
    if index is None: index = np.ones(len(x)).astype(bool)
    parvals = params.valuesdict()
    c = parvals['c']
    k = parvals['exponent']
    A = parvals['amplitude']

    y_pred = A*np.power(x, k) + c
    if y is None:
        return y_pred[index]
    elif weights is None:
        return (y - y_pred)[index]
    else:
        return ((y - y_pred)*weights)[index]
    
def exponential(params, x, y=None, weights=None):
    """"""
    parvals = params.valuesdict()
    c = parvals['c']
    k = parvals['exponent']
    A = parvals['amplitude']

    y_pred = A*np.exp(k*x) + c
    if y is None:
        return y_pred
    elif weights is None:
        return (y - y_pred)
    else:
        return (y - y_pred)*weights    
    