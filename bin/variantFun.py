from lmfit import minimize, Parameters, report_fit, conf_interval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scikits.bootstrap import bootstrap
from statsmodels.stats.weightstats import DescrStatsW
import warnings
import itertools
import os
import seqfun
import IMlibs
import scipy.stats as st
import fitFun
import fileFun
import plotFun
from plotFun import fix_axes
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1})

def findExtInList(directory, ext):
    if os.path.isdir(directory):
        files = os.listdir(directory)
        return [os.path.join(directory, i) for i in files if i.find(ext)>-1 and i.find(ext)==len(i)-len(ext)]
    else:
        print 'No directory named: %s'%directory
        return []

def loadFile(directory, ext):
    filenames = findExtInList(directory, ext)
    if len(filenames)==1:
        data = fileFun.loadFile(filenames[0])
        print 'Loaded file: %s'%filenames[0]
    else:
        data = None
        if len(filenames) > 1:
            print 'More than one file found: %s'%('\t\n'.join(filenames))
        else:
            print 'Could not find extension %s in directory %s'%(ext, directory)
    return data


def initialize(directory):
    """ Find the variant table, binding series, and cluster table in directory. """
    variant_table = loadFile(directory, 'normalized.CPvariant')
    binding_series = loadFile(directory, 'normalized.CPseries.pkl')
    cluster_table = loadFile(directory, 'normalized.CPfitted.pkl')
    annotated_clusters = loadFile(os.path.join(directory, '../../seqData/'),'CPannot.pkl')
    time_series = loadFile(directory, 'CPtimeseries.pkl')

    # x
    times = loadFile(directory, 'times')
    concentrations = loadFile(os.path.join(directory, '..'), 'concentrations.txt')
    timedict = loadFile(directory, 'timeDict.p')
        
    return variant_table, binding_series, cluster_table, annotated_clusters, time_series, times, concentrations, timedict



class perVariant():
    def __init__(self, variant_table, annotated_clusters, binding_series, x, cluster_table=None):
        self.binding_series = binding_series
        self.annotated_clusters = annotated_clusters.loc[:, 'variant_number']
        self.variant_table = variant_table
        self.x = x
        self.cluster_table = cluster_table
    
    def getVariantBindingSeries(self,variant ):
        index = self.annotated_clusters == variant
        return self.binding_series.loc[index]
    
    def plotBindingCurve(self, variant):
        subSeries = self.getVariantBindingSeries(variant)
        concentrations = self.x
        variant_table = self.variant_table
        
        # plot
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        plotFun.plotFitCurve(concentrations,
                            subSeries,
                            variant_table.loc[variant],
                            ax=ax)
        
    def plotOffrateCurve(self, variant, annotate=False):
        subSeries = self.getVariantBindingSeries(variant)
        times = self.x
        variant_table = self.variant_table
        
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        
        fitParameters = pd.DataFrame(columns=['fmax', 'koff', 'fmin'])
        plotFun.plotFitCurve(times, subSeries, variant_table.loc[variant], fitParameters, ax=ax,
                 log_axis=False, func=fitFun.objectiveFunctionOffRates, fittype='off')
        if annotate:
            annotationText = ['koff= %4.2e (%4.2e, %4.2e)'%(variant_table.loc[variant].koff,
                                                                  variant_table.loc[variant].koff_lb,
                                                                  variant_table.loc[variant].koff_ub),
                              'fmax= %4.2f (%4.2f, %4.2f)'%(variant_table.loc[variant].fmax,
                                                                variant_table.loc[variant].fmax_lb,
                                                                variant_table.loc[variant].fmax_ub),
                              'Nclusters= %d'%variant_table.loc[variant].numTests,
                              'pvalue= %.1e'%variant_table.loc[variant].pvalue,
                              'average Rsq= %4.2f'%variant_table.loc[variant].rsq,
                              ]
            ax.annotate('\n'.join(annotationText), xy=(.25, .95), xycoords='axes fraction',
                        horizontalalignment='left', verticalalignment='top')
            
    def plotClusterOffrates(self, variant, cluster=None, idx=None):
        times = self.x
        
        if cluster is not None:
            fluorescence = subSeries.loc[cluster]
        else:
            subSeries = self.getVariantBindingSeries(variant)
            if idx is not None:
                fluorescence = subSeries.iloc[idx]
            else:
                fluorescence = subSeries.iloc[0]
        cluster = fluorescence.name
        cluster_table = self.cluster_table
        
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        
        fitParameters = pd.DataFrame(columns=['fmax', 'koff', 'fmin'])
        plotFun.plotFitCurve(times, fluorescence, cluster_table.loc[cluster], fitParameters, ax=ax,
                 log_axis=False, func=fitFun.objectiveFunctionOffRates, fittype='off')

    def plotClusterBinding(self, variant, cluster=None, idx=None):
        concentrations = self.x
        
        if cluster is not None:
            fluorescence = subSeries.loc[cluster]
        else:
            subSeries = self.getVariantBindingSeries(variant)
            if idx is not None:
                fluorescence = subSeries.iloc[idx]
            else:
                fluorescence = subSeries.iloc[0]
        cluster = fluorescence.name
        cluster_table = self.cluster_table
        
        # plot
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        plotFun.plotFitCurve(concentrations,
                            fluorescence,
                            cluster_table.loc[cluster],
                            ax=ax)
     
    def plotBootstrappedDist(self, variant, param, log_axis=False):
        variant_table = self.variant_table
        cluster_table = self.cluster_table
        subSeries = self.getVariantBindingSeries(variant)
        
        params = cluster_table.loc[subSeries.index, param]
        
        # make bootstrapped dist
        if log_axis:
            vec = np.log10(params.dropna())
            med = np.log10(variant_table.loc[variant, param])
            ub = np.log10(variant_table.loc[variant, param+'_ub'])
            lb = np.log10(variant_table.loc[variant, param+'_lb'])
            xlabel = 'log '+param
        else:
            vec = params.dropna()
            med = variant_table.loc[variant, param]
            ub = variant_table.loc[variant, param+'_ub']
            lb = variant_table.loc[variant, param+'_lb']
            xlabel = param
        plt.figure(figsize=(4,3))
        sns.distplot(vec, color='r', kde=False)
        plt.axvline(med, color='0.5', linestyle='-')
        plt.axvline(lb, color='0.5', linestyle=':')
        plt.axvline(ub, color='0.5', linestyle=':')
        plt.xlabel(xlabel)
        fix_axes(plt.gca())
        plt.tight_layout()
        
        
class perFlow():
    def __init__(self, affinityData, offRate):
        self.affinityData = affinityData
        self.offRate = offRate
        self.all_variants = pd.concat([affinityData.variant_table, offRate.variant_table], axis=1).index
    
    def getGoodVariants(self, ):
        variants = (pd.concat([self.affinityData.variant_table.pvalue < 0.01,
                               self.offRate.variant_table.pvalue < 0.01,
                               self.offRate.variant_table.fmax_lb>0.2], axis=1)).all(axis=1)
        return variants
    
    def plotDeltaGDoubleDagger(self, variant=None, dG_cutoff=None, plot_on=False, params=['koff', 'dG']):
        parameters = fitFun.fittingParameters()
        koff = self.offRate.variant_table.loc[self.all_variants, params[0]]
        dG = self.affinityData.variant_table.loc[self.all_variants, params[1]]

        if variant is None:
            variant = 34936
        
        # find dG predicted from off and on rates
        dG_dagger = parameters.find_dG_from_Kd(koff.astype(float))
        kds = parameters.find_Kd_from_dG(dG)
        dG_dagger_on = parameters.find_dG_from_Kd((koff/kds).astype(float))
        
        # find the variants to plot based on goodness of fit and dG cutoff if given
        variants = self.getGoodVariants()
        if dG_cutoff is not None:
            variants = variants&(dG < dG_cutoff)
        
        # deced which y to plot based on user input
        x = (dG - dG.loc[variant]).loc[variants]
        if plot_on:
            y = -(dG_dagger_on - dG_dagger_on.loc[variant]).loc[variants]
        else:
            y = (dG_dagger - dG_dagger.loc[variant]).loc[variants]
        
        # plot
        fig = plt.figure(figsize=(3,3));
        ax = fig.add_subplot(111, aspect='equal')
        
        xlim = [min(x.min(), y.min()), max(x.max(), y.max())]
        im = plt.hexbin(x, y,  extent=xlim+xlim, gridsize=100, cmap='Spectral_r', mincnt=1)
        #sns.kdeplot(x, z,  cmap="Blues", shade=True, shade_lowest=False)
        slope, intercept, r_value, p_value, std_err = st.linregress(x,y)
        
        # offset

        num_variants = 100
        offset = (x - y).mean()
        
        xlim = np.array(ax.get_xlim())
        plt.plot(xlim, xlim*slope + intercept, 'k--', linewidth=1)
        if plot_on:
            plt.plot(xlim, [y.mean()]*2, 'r', linewidth=1)
        else:
            plt.plot(xlim, xlim, 'r', linewidth=1)
        plt.xlabel('$\Delta \Delta G$')
        plt.ylabel('$\Delta \Delta G_{off}\dagger$')
        
        #plt.colorbar(im)
        plt.tight_layout()
        fix_axes(ax)

        annotationText = ['slope = %4.2f '%(slope),
                          'intercept = %4.2f'%intercept,
                          'pvalue = %4.1e'%p_value
                          ]
        ax.annotate('\n'.join(annotationText), xy=(.05, .95), xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top')
        
        return x, y
    
    def plotKdVersusKoff(self, ):
        parameters = fitFun.fittingParameters()
        results_off = self.offRate.variant_table.loc[self.all_variants]
        dG = self.affinityData.variant_table.dG.loc[self.all_variants]
        
        plt.figure(figsize=(4,4))
        plt.hexbin(dG, results_off.koff, yscale='log', mincnt=1, cmap='Spectral_r')
        plt.xlabel('$\Delta G$ (kcal/mol)')
        plt.ylabel('$k_{off}$ (s)')
        fix_axes(plt.gca())
        plt.tight_layout()


    def plotEquilibrationTimes(self, concentration, wait_time, initial=1E-9):
        parameters = fitFun.fittingParameters()
        variants = self.getGoodVariants()
        koff = self.offRate.variant_table.loc[variants].koff
        dG = self.affinityData.variant_table.loc[variants].dG
        kds = parameters.find_Kd_from_dG(dG.astype(float))*1E-9
        
        # for meshgrid
        kobs_bounds = [-5, 0]
        koff_bounds = [-4, -2]
        xx, yy = np.meshgrid(np.linspace(*koff_bounds, num=200),
                             np.linspace(*kobs_bounds, num=200))
        
        # for each kobs = x(i,j)+y(i,j), plot fraction equilibrated
        labels = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1-1E-12])
        min_kobs = kobs_bounds[0] + np.log10(concentration/initial)
        plt.figure(figsize=(4,4))
        cs =plt.contour(xx, yy, 100*fraction_equilibrated(np.power(10, xx)+np.power(10, yy),
                                                      wait_time),
                        labels*100, colors='k', linewidths=1)
        plt.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
        
        plt.hexbin(np.log10(koff), np.log10(concentration*koff/kds), mincnt=1,
                   cmap='Spectral_r', extent=koff_bounds+[min_kobs, min_kobs+2], gridsize=150)
        plt.xlabel('log$(k_{off})$')
        plt.ylabel('log$([$flow$] k_{on})$')
        plt.title('%.2e'%(concentration*1E9))
        plt.ylim(min_kobs, min_kobs+2)
        ax = fix_axes(plt.gca())
        plt.tight_layout()
    
class compareFlow():
    def __init__(self, affinityData1, affinityData2):
        self.expt1 = affinityData1
        self.expt2 = affinityData2
        self.all_variants = pd.concat([affinityData1.variant_table, affinityData2.variant_table], axis=1).index

    def getGoodVariants(self, ):
        variants = (pd.concat([self.expt1.variant_table.pvalue < 0.01,
                               self.expt2.variant_table.pvalue < 0.01], axis=1)).all(axis=1)
        return variants

    def compareParam(self, param, log_axes=False, filter_pvalue=False):
        x = self.expt1.variant_table.loc[self.all_variants, param]
        y = self.expt2.variant_table.loc[self.all_variants, param]
        
        if log_axes:
            x = np.log10(x)
            y = np.log10(y)
        
        if filter_pvalue:
            variants = self.getGoodVariants()
            x = x.loc[variants]
            y = y.loc[variants]
            
        plt.figure(figsize=(4,4))
        plt.hexbin(x, y, cmap='Spectral_r', mincnt=1)
        fix_axes(plt.gca())
        
    

def fraction_equilibrated(kobs, time_waited):
    return 1-np.exp(-kobs*time_waited)
 

    
def _makebootstrappeddist(vec, n_samples=1000, statfunction=np.median):
    indices = np.random.choice(vec.index, [n_samples, len(vec)])
    bootstrapped_vec = []
    for index in indices:
        bootstrapped_vec.append(statfunction(vec.loc[index]))
    return np.array(bootstrapped_vec)
                                
        