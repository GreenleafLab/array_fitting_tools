import numpy as np
import pandas as pd
from fittinglibs.fitting import fittingParameters
  

def rates_off(params, times, data=None, weights=None, index=None, bleach_fraction=1, image_ns=None):
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
    
    
def rates_on(params, times, data=None, weights=None, index=None,  bleach_fraction=1, image_ns=None):
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
        
def binding_curve(params, concentrations, data=None, weights=None, index=None):
    """  Return fit value, residuals, or weighted residuals of a binding curve.
    
    Hill coefficient 1. """
    if index is None:
        index = np.ones(len(concentrations)).astype(bool)
        
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
        return fracbound[index]
    
    # return residuals if data is given
    elif weights is None:
        return (fracbound - data)[index]
    
    # return weighted residuals if data is given
    else:
        return ((fracbound - data)*weights)[index]
    
def binding_curve_linear(params, concentrations, data=None, weights=None, index=None):
    """  Return fit value, residuals, or weighted residuals of a binding curve.
    
    Hill coefficient 1. """
    if index is None:
        index = np.ones(len(concentrations)).astype(bool)
        
    parameters = fittingParameters()
    
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    dG   = parvals['dG']
    fmin = parvals['fmin']
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

    
def binding_curve_nonlinear(params, concentrations, data=None, weights=None, index=None):
    """  Return fit value, residuals, or weighted residuals of a binding curve with nonlinear, nonspecific term.
    
    Hill coefficient 1. """
    if index is None:
        index = np.ones(len(concentrations)).astype(bool)
        
    parameters = fittingParameters()
    
    parvals = params.valuesdict()
    fmax = parvals['fmax']
    dG   = parvals['dG']
    fmin = parvals['fmin']
    dG_ns = parvals['dGns']

    Kd = np.exp(dG/parameters.RT)/parameters.concentration_units
    Kd_ns = np.exp(dG_ns/parameters.RT)/parameters.concentration_units
    fracbound = (fmin +
                 fmax*(concentrations/(concentrations + Kd) +
                       concentrations**2/(concentrations + Kd)/Kd_ns))
    
    # return fit value of data is not given
    if data is None:
        return fracbound[index]
    
    # return residuals if data is given
    elif weights is None:
        return (fracbound - data)[index]
    
    # return weighted residuals if data is given
    else:
        return ((fracbound - data)*weights)[index]


