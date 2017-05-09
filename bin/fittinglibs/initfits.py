from lmfit import minimize, Parameters, report_fit, conf_interval
import numpy as np
import pandas as pd
import sys
import warnings
import itertools
import scipy.stats as st
import ipdb
from fittinglibs import objfunctions, variables

class FitParams():
    """Class with attributes objective function, initial params, upper/lowerbounds on params."""
    def __init__(self, func_name, x, init_kws={}):
        self.func = getattr(objfunctions, func_name)
        self.param_names = self.func(None, None, return_param_names=True)
        self.x = x
        # go through and get defaults for each of these fit params
        fit_parameters = {}
        for param_name in self.param_names:
            fit_parameters[param_name] = getattr(self, 'get_init_%s'%param_name)()
        self.fit_parameters = fit_parameters
        self.update_fit_parameters(**init_kws)
            
    def update_initfit_params(self, **init_kws):
        """Given kwargs, update the fit_parameter values."""
        fit_parameters = self.fit_parameters
        for param_name in self.param_names:
            fit_param_dict = fit_parameters[param_name]
            if param_name in init_kws.keys():
                for key, val in init_kws[param_name].items():
                    fit_param_dict[key] = val
            fit_parameters[param_name] = fit_param_dict
        self.fit_parameters = fit_parameters      
        
    def get_init_params(self, fit_parameters=None):
        """Return lmfit Parameters class."""
        if fit_parameters is None: fit_parameters = self.fit_parameters
        params = Parameters()
        for param_name, init_dict in fit_parameters.items():
            params.add(param_name, value=init_dict['initial'],
                   min = init_dict['lowerbound'],
                   max = init_dict['upperbound'],
                   vary= init_dict['vary'])
        return params
    
    def get_init_dG(self, **kwargs):
        """Given the initial concentrations, find an initial dG value."""
        parameters = variables.fittingParameters()
        lb   = parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.99, self.x[0]))
        init = parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.5,  self.x[-1]))
        ub   = parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.01, self.x[-1]))
        return {'lowerbound':lb, 'initial':init, 'upperbound':ub, 'vary':True}
        
    def get_init_fmax(self):
        """Find initial fmax values."""
        return {'lowerbound':0, 'initial':np.nan, 'upperbound':np.inf, 'vary':True}

    def get_init_fmin(self):
        """Find initial fmin values."""
        return {'lowerbound':0, 'initial':0, 'upperbound':np.inf, 'vary':True}
        
