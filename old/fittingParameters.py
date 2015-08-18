#!/usr/bin/env python

# Author: Sarah Denny, Stanford University 

# Parameters to analyze the images from RNA array experiment 11/11/14


##### IMPORT #####
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seqfun
import scipy.stats as st
import IMlibs
#import lmfit


class Parameters():
    
    """
    stores file names
    """
    def __init__(self, concentrations=None, f_abs_green_max=None,
                 f_abs_red=None, f_abs_green_nonbinders=None, fittype=None, table=None):
        if fittype is None: fittype='binding'
        if f_abs_red is None and f_abs_green_max is not None:
            f_abs_red = pd.Series(1, index=f_abs_green_max)
        
        # save the units of concentration given in the binding series
        self.concentration_units = 1E-9 # i.e. nM
        self.RT = 0.582

        # save parameters for barcode mapping
        self.barcode_col = 7 # 1 indexed
        self.sequence_col = 5 # 1 indexed            
        
        # initial binding curve settings
        self.frac_bound_upperbound = 0.01
        self.frac_bound_lowerbound = 0.99
        self.fdr_cutoff = 0.05

        
        self.fitParameters = pd.DataFrame(columns=['fmax', 'dG', 'fmin', 'toff', 'ton'],
                                          index=['lowerbound', 'initial', 'upperbound'])
        if concentrations is not None:
            self.concentrations = concentrations
            self.maxdG = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(0.9,
                                                                                         concentrations[-1]))
            self.mindG = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(0.01,
                                                                                         concentrations[0]))
        # make sure everything is an array

        if fittype == 'binding':
            if concentrations is not None:        
                currParam = 'dG'
                self.fitParameters[currParam]['lowerbound'] = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(self.frac_bound_lowerbound, self.concentrations[0]))
                self.fitParameters[currParam]['initial'] = self.find_dG_from_Kd(self.concentrations[-1])
                self.fitParameters[currParam]['upperbound'] = self.find_dG_from_Kd(self.find_Kd_from_frac_bound_concentration(self.frac_bound_upperbound, self.concentrations[-1]))
                
            currParam = 'fmin'  
            self.fitParameters[currParam]['lowerbound'] = 0
            self.fitParameters[currParam]['initial'] = 0
            self.fitParameters[currParam]['upperbound'] = 1 # this is futher reducedor relaxed if the first point in the binding curve was fit
            
            currParam = 'fmax'
            self.mx_factor_fmax = 10
            #self.fitParameters[currParam]['lowerbound']= self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff)
            self.fitParameters[currParam]['lowerbound']= 0
            self.fitParameters[currParam]['initial']= 1 # this will be defined per cluster
            #self.fitParameters[currParam]['initial']= 1
            if f_abs_green_max is None:
                self.fitParameters[currParam]['upperbound'] = self.mx_factor_fmax*self.fitParameters[currParam]['initial']
            else:
                self.fitParameters[currParam]['upperbound'] = self.find_fmax_upperbound(f_abs_green_max, self.mx_factor_fmax)
            #self.fitParameters[currParam]['upperbound']= self.find_fmax_upperbound(f_abs_green_max/f_abs_red, self.mx_factor_fmax)
            
        
        if fittype == 'offrate':    
            currParam = 'toff'
            self.fitParameters[currParam]['lowerbound'] = 1E0
            self.fitParameters[currParam]['initial']= 1E6 
            self.fitParameters[currParam]['upperbound'] = 1E6
            
            currParam = 'fmin'  
            self.fitParameters[currParam]['lowerbound'] = 0
            self.fitParameters[currParam]['initial'] = 0
            self.fitParameters[currParam]['upperbound'] = self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff) # this is futher reducedor relaxed if the first point in the binding curve was fit
            
            currParam = 'fmax'
            self.fitParameters[currParam]['lowerbound']= 0
            self.fitParameters[currParam]['initial']= np.nan # this will be defined per cluster
            self.fitParameters[currParam]['upperbound'] = self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff) # this is futher reducedor relaxed if the first point in the binding curve was fit
        
        if fittype == 'onrate':
            currParam = 'ton'
            self.fitParameters[currParam]['lowerbound'] = 1E-1
            self.fitParameters[currParam]['initial']= 1E4 
            self.fitParameters[currParam]['upperbound'] = 1E6
            
            currParam = 'fmin'  
            self.fitParameters[currParam]['lowerbound'] = 0
            self.fitParameters[currParam]['initial'] = np.nan # this will be defined per cluster
            self.fitParameters[currParam]['upperbound'] = self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff) # this is futher reducedor relaxed if the first point in the binding curve was fit
            
            currParam = 'fmax'
            self.fitParameters[currParam]['lowerbound']= 0
            self.fitParameters[currParam]['initial']= np.nan # this will be defined per cluster
            self.fitParameters[currParam]['upperbound'] = self.find_fmax_lowerbound(f_abs_green_nonbinders, self.fdr_cutoff) # this is futher reducedor relaxed if the first point in the binding curve was fit
                        
                
                
            # estimate conversion of f_abs_red to f_abs_green
        if (f_abs_green_max is not None and f_abs_red is not None):
            self.scale_factor = self.find_scale_factor(f_abs_green_max, f_abs_red)
        
            # fit stability
            self.vary_fmax_lowerbounds = np.linspace(0, 1000, 11)
            self.vary_mx_factor_fmax = np.logspace(-1, 4, 11)
            self.vary_fmax_upperbounds = np.array([self.find_fmax_upperbound(f_abs_green_max, mx_factor_fmax) for mx_factor_fmax in self.vary_mx_factor_fmax])
            
            self.vary_scale_factor = np.logspace(-2, 2, 11) # to vary fmax initial guess
            
            self.vary_dG_initial = np.linspace(self.fitParameters['dG']['lowerbound'],
                                               self.fitParameters['dG']['upperbound'], 11)
            self.vary_fmin_upperbound = np.logspace(1, np.log10(5000), 10)

        # adding these functions to estimate the fmax given the table and the number of measurements
        if table is not None:
            
            grouped = table.groupby('variant_number')
            concentrationCols = IMlibs.formatConcentrations(concentrations)
            param_names = ['fmax', 'dG', 'fmin'] 
            all_fit_params = pd.concat([grouped['fmax'].count(), grouped[param_names].median()], axis=1);
            all_fit_params.columns = np.hstack(['number', param_names]) 
            
            tight_binders_fit_params = all_fit_params.loc[all_fit_params.dG < self.maxdG]
            
            per_variant_params = ['fmax']
            per_variant = pd.DataFrame(index = np.unique(tight_binders_fit_params.number),
                                       columns=(['weight'] +
                                                ['%s_mean'%s for s in per_variant_params] +
                                                ['%s_std'%s for s in per_variant_params]),
                                       dtype=float)
            for n in np.unique(tight_binders_fit_params.number):
                per_variant.loc[n, 'weight'] = np.sqrt((tight_binders_fit_params.number==n).sum())
                for idx in per_variant_params:
                    outliers = seqfun.is_outlier(tight_binders_fit_params.loc[tight_binders_fit_params.number==n, idx])
                    distribution = tight_binders_fit_params.loc[tight_binders_fit_params.number==n, idx].loc[~outliers]
                    per_variant.loc[n, '%s_mean'%idx] = distribution.mean()
                    per_variant.loc[n, '%s_std'%idx] = distribution.std()

            per_variant.dropna(inplace=True, axis=0)

            self.tight_binders_fit_params = tight_binders_fit_params
            self.per_variant = per_variant
            self.default_errors = grouped[concentrationCols].std().mean()
                
            start = 2 # start fit with at least three vclusters/variant
            self.fits = pd.DataFrame(index=per_variant_params, columns=['sigma', 'c'])
            for param in per_variant_params:
                params = lmfit.Parameters()
                params.add('sigma', value=per_variant.loc[start, '%s_std'%param]/np.sqrt(start), min=0)
                params.add('c', value=per_variant.iloc[-10:].mean().loc['%s_std'%param], min=0)
                
                results = lmfit.minimize(self.objectiveFunction, params,
                                   args=(np.array(per_variant.loc[start:].index),),
                                   kws={'y':per_variant.loc[start:, '%s_std'%param].values,
                                        'weights':per_variant.loc[start:, 'weight'].values},
                                   xtol=1E-6, ftol=1E-6, maxfev=10000)
                
                popt = [params[name].value for name in params.keys()]
                
                #popt, pcov = curve_fit(self.fitFunction,
                #                           np.array(per_variant.loc[start:].index),
                #                           per_variant.loc[start:, '%s_std'%param].values,
                #                           p0=[1],
                #                           sigma=1/per_variant.loc[start:, 'weight'].values)
                self.fits.loc[param] = popt
            
            # updat the initial fit parameters given table info
            loose_binders = all_fit_params.loc[all_fit_params.dG > self.mindG]
            self.fitParameters.loc[:, 'fmin'] = np.percentile(loose_binders.fmin, [0.5, 50, 99.5])
            self.fitParameters.loc[:, 'fmax'] = np.percentile(tight_binders_fit_params.fmax, [0.5, 50, 99.5])
            self.fitParameters.loc['vary'] = True
            self.fitParameters.loc['vary', 'fmin'] = False
            
            # plot data
            plt.figure(figsize=(4,3));
            plt.plot(per_variant.index, per_variant.loc[:, 'fmax_std'],  'ko');
            
            # fit exponential function
            plt.plot(per_variant.index, self.fitFunction(per_variant.index, *popt), 'c')
            
            # fit 1/sqrt(n)
            plt.xlabel('number of tests')
            plt.ylabel('std of fit fmaxes in bin')
            plt.legend()
            plt.tight_layout()
            



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

    def find_Kd_from_dG(self, dG):
        return np.exp(dG/self.RT)/self.concentration_units
    
    def find_Kd_from_frac_bound_concentration(self, frac_bound, concentration):
        return concentration/float(frac_bound) - concentration
    
    def fitFunction(self, x, a, b):
        return a/np.sqrt(x) + b
    
    def objectiveFunction(self, params, x, y=None, weights=None):
        parvals = params.valuesdict()
        sigma = parvals['sigma']
        c   = parvals['c']
        fit = sigma/np.sqrt(x) + c
        if y is None:
            return fit
        else:
            return (y - fit)*weights
    

    
    
    def find_fmax_bounds_given_n(self, n, alpha=None, return_dist=None):
        if alpha is None: alpha = 0.99
        if return_dist is None: return_dist = False
        popts = self.fits.loc['fmax']
        sigma = self.fitFunction(n, *popts)
        fmax_median = self.tight_binders_fit_params.fmax.median()
        
        if return_dist:
            return st.norm(loc=fmax_median, scale=sigma)
        else:
            interval = st.norm.interval(alpha, loc=fmax_median, scale=sigma)
            return interval[0], fmax_median, interval[1]
    
    def find_default_error_given_n(self, n, alpha=None):
        if alpha is None: alpha = 0.99
        concentrationCols = IMlibs.formatConcentrations(self.concentrations)
        errors = pd.DataFrame(index=concentrationCols, columns=['eminus', 'eplus'])
        for concentrationCol in concentrationCols:
            popts = self.fits.loc[concentrationCol]
            sigma = self.fitFunction(n, *popts)
            mean  = self.tight_binders_fit_params.loc[:, concentrationCol].median()
            errors.loc[concentrationCol] = st.norm.interval(alpha, loc=mean, scale=sigma)
        return errors   
    