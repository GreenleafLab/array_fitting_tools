#!/usr/bin/env python

# Author: Sarah Denny, Stanford University 

# Parameters to analyze the images from RNA array experiment 11/11/14


##### IMPORT #####
import numpy as np
import pandas as pd

class Parameters():
    
    """
    stores file names
    """
    def __init__(self, concentrations, f_abs_green, f_abs_red, f_abs_green_nonbinders):
        
        # save the units of concentration given in the binding series
        self.concentration_units = 1E-9 # i.e. nM
        self.RT = 0.582
        self.concentrations = concentrations
        
        # initial binding curve settings
        self.frac_bound_upperbound = 0.001
        self.frac_bound_lowerbound = 0.999
        
        self.fitParameters = pd.DataFrame(columns=['fmax', 'dG', 'fmin'],
                                          index=['lowerbound', 'initial', 'upperbound'])
        
        currParam = 'dG'
        self.fitParameters[currParam]['lowerbound'] = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(self.frac_bound_lowerbound, self.concentrations[0]))
        self.fitParameters[currParam]['initial'] = self.find_dG_from_Kd(self.concentrations[-1])
        self.fitParameters[currParam]['upperbound'] = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(self.frac_bound_upperbound, self.concentrations[-1]))
        
        currParam = 'fmin'  
        self.fitParameters[currParam]['lowerbound'] = 0
        self.fitParameters[currParam]['initial'] = 0
        self.fitParameters[currParam]['upperbound'] = np.nan # this is defined per cluster
        
        currParam = 'fmax'
        self.fdr_cutoff = 0.05
        self.mx_factor_fmax = 10
        self.fitParameters[currParam]['lowerbound']= self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff)
        self.fitParameters[currParam]['initial']= np.nan # this will be defined per cluster
        self.fitParameters[currParam]['upperbound'] = self.find_fmax_upperbound(f_abs_green, self.mx_factor_fmax)
        
        # save parameters for barcode mapping
        self.barcode_col = 7 # 1 indexed
        self.sequence_col = 5 # 1 indexed
        
        # fit stability
        self.vary_fmax_lowerbounds = np.linspace(0, 1000, 11)
        self.vary_mx_factor_fmax = np.logspace(-1, 4, 11)
        self.vary_fmax_upperbounds = np.array([self.find_fmax_upperbound(f_abs_green, mx_factor_fmax) for mx_factor_fmax in self.vary_mx_factor_fmax])
        self.vary_dG_initial = np.linspace(self.fitParameters['dG']['lowerbound'],
                                           self.fitParameters['dG']['upperbound'], 11)
        
        self.scale_factor = self.find_scale_factor(f_abs_green, f_abs_red, subset_index=f_abs_green>self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff) )
        

    def find_fmax_upperbound(self, f_abs_green, scale_factor=None):
        if scale_factor is None:
            scale_factor = 10 # multiplying factor on max of f_abs_green to constrain f_max upper bound
        return np.max(f_abs_green)*scale_factor
    
    def find_fmax_lowerbound(self, null_scores, FDR):
        return np.percentile(null_scores, 100*(1-FDR))

    def find_scale_factor(self, f_abs_green, f_abs_red, subset_index=None):
        if subset_index is None:
            subset_index = np.arange(len(f_abs_green)) # subset of fluorescence values to compare. Should give the subset you think is bound.
        return np.median((f_abs_green/f_abs_red)[subset_index])
    
    def find_dG_from_Kd(self, Kd):
        return self.RT*np.log(Kd*self.concentration_units)
    
    def find_Kd_from_frac_bound_concentration(self, frac_bound, concentration):
        return concentration/float(frac_bound) - concentration
    
    
    
    