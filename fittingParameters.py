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
    def __init__(self, concentrations=None, f_abs_green_max=None,
                 f_abs_red=None, f_abs_green_nonbinders=None):        
        # save the units of concentration given in the binding series
        self.concentration_units = 1E-9 # i.e. nM
        self.RT = 0.582

        # save parameters for barcode mapping
        self.barcode_col = 7 # 1 indexed
        self.sequence_col = 5 # 1 indexed            
        
        # initial binding curve settings
        self.frac_bound_upperbound = 0.001
        self.frac_bound_lowerbound = 0.999
        self.fdr_cutoff = 0.05
        
        self.fitParameters = pd.DataFrame(columns=['fmax', 'dG', 'fmin'],
                                          index=['lowerbound', 'initial', 'upperbound'])
        # make sure everything is an array
        if (f_abs_green_max is not None and concentrations is not None and
            f_abs_red is not None and f_abs_green_nonbinders is not None):
            f_abs_green_max = np.array(f_abs_green_max)
            self.concentrations = concentrations
        
            currParam = 'dG'
            self.fitParameters[currParam]['lowerbound'] = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(self.frac_bound_lowerbound, self.concentrations[0]))
            self.fitParameters[currParam]['initial'] = self.find_dG_from_Kd(self.concentrations[-1])
            self.fitParameters[currParam]['upperbound'] = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(self.frac_bound_upperbound, self.concentrations[-1]))
            
            currParam = 'fmin'  
            self.fitParameters[currParam]['lowerbound'] = 0
            self.fitParameters[currParam]['initial'] = 0
            self.fitParameters[currParam]['upperbound'] = 70 # this is futher reducedor relaxed if the first point in the binding curve was fit
            
            currParam = 'fmax'
            self.mx_factor_fmax = 100
            self.fitParameters[currParam]['lowerbound']= self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff)
            self.fitParameters[currParam]['initial']= np.nan # this will be defined per cluster
            self.fitParameters[currParam]['upperbound'] = self.find_fmax_upperbound(f_abs_green_max, self.mx_factor_fmax)
            
            # estimate conversion of f_abs_red to f_abs_green
            self.scale_factor = self.find_scale_factor(f_abs_green_max, f_abs_red, subset_index=f_abs_green_max>self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff) )
        
            # fit stability
            self.vary_fmax_lowerbounds = np.linspace(0, 1000, 11)
            self.vary_mx_factor_fmax = np.logspace(-1, 4, 11)
            self.vary_fmax_upperbounds = np.array([self.find_fmax_upperbound(f_abs_green_max, mx_factor_fmax) for mx_factor_fmax in self.vary_mx_factor_fmax])
            
            self.vary_scale_factor = np.logspace(-2, 2, 11) # to vary fmax initial guess
            
            self.vary_dG_initial = np.linspace(self.fitParameters['dG']['lowerbound'],
                                               self.fitParameters['dG']['upperbound'], 11)
            self.vary_fmin_upperbound = np.logspace(1, np.log10(5000), 10)



    def find_fmax_upperbound(self, f_abs_green, scale_factor=None):
        if scale_factor is None:
            scale_factor = 100 # multiplying factor on max of f_abs_green to constrain f_max upper bound
        return np.percentile(f_abs_green[np.isfinite(f_abs_green)], 99.99)*scale_factor
    
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
    
    
    
    