"""
Sarah Denny
Stanford University

"""

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import warnings
import argparse
import itertools  
import seaborn as sns
import scipy.spatial.distance as ssd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib as mpl
from matplotlib import gridspec
from joblib import Parallel, delayed
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1})

import fitFun
import seqfun

def fix_axes(ax):
    ax.tick_params(which='minor', top='off', right='off')
    ax.tick_params(top='off', right='off', pad=2, labelsize=10, labelcolor='k')
    return ax

def get_c(x, y, distance_threshold=None):
    """ Given two arrays x and y, return the number of points within a certain distance of each point."""
    if distance_threshold is None:
        distance_threshold = min(x.std(), y.std())/10
    distance_mat = ssd.squareform(ssd.pdist(pd.concat([x, y], axis=1)))
    c = ((distance_mat < distance_threshold).sum(axis=1) - 1)/2
    c = (c-c.min())/(c.max()-c.min()).astype(float)
    return c


def my_smoothed_scatterplot(x,y, color=None,**kwargs):
    """ given x and y, plot a scatterplot with color according to density."""
    c = get_c(x, y)
    if 'cmap' not in kwargs.keys():
        if color is not None:
            cmap = sns.dark_palette(color, as_cmap=True)
        else:
            cmap = None
        plt.scatter(x, y, c=c, cmap=cmap,
                edgecolors='none', marker='.', rasterized=True, **kwargs)
    else:
        plt.scatter(x, y, c=c, 
                edgecolors='none', marker='.', rasterized=True, **kwargs)

    return



def plotDataErrorbars(x, subSeries, ax=None, capsize=2):
    """ Find errorbars on set of cluster fluorescence and plot. """
    
    # set errors to NaN unless successfully find them later on with bootstrapping
    default_errors = np.ones(len(x))*np.nan

    # if subseries is a dataframe, find errors along columns. if verctor, no errors will be found.
    if len(subSeries.shape) == 1:
        fluorescence = subSeries
        use_default = True
        numTests = np.array([1 for col in subSeries])
    else:
        fluorescence = subSeries.median()
        use_default = False
        numTests = np.array([len(subSeries.loc[:, col].dropna()) for col in subSeries])
    
    # option to use only default errors provdided for quicker runtime
    if not use_default:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eminus, eplus = fitFun.findErrorBarsBindingCurve(subSeries)

    # if ax is given, plot to that. else
    if ax is None:
        ax = plt.gca()
    
    # plot binding points
    ax.errorbar(x, fluorescence,
                 yerr=[eminus, eplus], fmt='.', elinewidth=1,
                 capsize=capsize, capthick=1, color='k', linewidth=1)
    return ax

def plotFit(x, params, func, ax=None, fit_kwargs=None):
    """ given x values, lmfit params,  and fitting function, calculate fit and plot to current axes."""
    if fit_kwargs is None:
        fit_kwargs = {}
    fit = func(params, x, **fit_kwargs)
    
    if ax is None:
        ax = plt.gca()
    ax.plot(x, fit, 'r')
    return ax

def plotFitBounds(x, params_lb, params_ub, func, ax=None, fit_kwargs=None):
    """ given x values, lmfit params,  and fitting function, calculate fit and plot to current axes."""
    if fit_kwargs is None:
        fit_kwargs = {}
    ub = func(params_ub, x, **fit_kwargs)
    lb = func(params_lb, x, **fit_kwargs)
    
    # plot upper and lower bounds
    if ax is None:
        ax = plt.gca()
    ax.fill_between(x, lb, ub, color='0.5',
                         label='95% conf int', alpha=0.5)
    return ax

def plotFitCurve(x, subSeries, results, param_names=None, ax=None, log_axis=True,
                 func=fitFun.bindingCurveObjectiveFunction, fittype='binding', kwargs=None):
    if kwargs is None:
        kwargs = {}
    
    # these are useful definitions for three commonly used fitting functions
    if fittype == 'binding':
        param_names_tmp = ['fmax', 'dG', 'fmin']
        ub_vec = ['_ub', '_lb', '']
        lb_vec = ['_lb', '_ub', '']
        capsize = 2
        log_axis = True
        xlabel = 'concentration (nM)'
    elif fittype == 'off':
        param_names_tmp = ['fmax', 'koff', 'fmin']
        ub_vec = ['_ub', '_lb', '_ub']
        lb_vec = ['_lb', '_ub', '_lb']
        capsize = 0
        log_axis = False
        xlabel = 'time (s)'
    elif fittype == 'on':
        param_names_tmp = ['fmax', 'kobs', 'fmin']
        ub_vec = ['_ub', '_ub', '_ub']
        lb_vec = ['_lb', '_lb', '_lb']
        capsize = 2
        log_axis=False
        xlabel = 'time (s)'
    elif fittype == 'binding_linear':
        param_names_tmp = ['fmax', 'dG', 'fmin', 'slope']
        ub_vec = ['_ub', '_ub', '_ub', '_ub']
        lb_vec = ['_lb', '_lb', '_lb', '_lb']
        capsize = 2
        log_axis = True
        xlabel = 'concentration (nM)'
        
    # allow custom definition of param_names with fitParameters 
    if param_names is None:
        param_names = param_names_tmp
    
    # get params for fit function
    params = fitFun.returnParamsFromResults(results, param_names)
    
    # gerenate x values for fit function
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
    
    # generate x values for fit function
    if log_axis:
        more_x = np.logspace(np.log10(x.min()/10), np.log10(x.max()*2), 100)
    else:
        more_x = np.linspace(x.min(), x.max(), 100)
    
    # plot the data
    plotDataErrorbars(x, subSeries, ax, capsize=capsize)
    
    # plot fit
    plotFit(more_x, params, func, ax=ax, fit_kwargs=kwargs)
    
    # plot upper and lower bounds
    all_param_names = [['%s%s'%(param, s) for param, s in itertools.izip(param_names, vec)]
                       for vec in [ub_vec, lb_vec]]
    if np.all(np.in1d(all_param_names, results.index.tolist())):
        params_ub = fitFun.returnParamsFromResultsBounds(results, param_names, ub_vec)
        params_lb = fitFun.returnParamsFromResultsBounds(results, param_names, lb_vec)
        plotFitBounds(more_x, params_lb, params_ub, func, ax=ax, fit_kwargs=kwargs)

    # format
    ylim = ax.get_ylim()
    xlim = more_x[[0,-1]]
    plt.ylim(0, ylim[1])
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    if log_axis:
        plt.xscale('log')
    else:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('normalized fluorescence')
    fix_axes(ax)
    plt.tight_layout()
    return



def plotFmaxVsKd(variant_table, cutoff, subset=None, kde_plot=False,
                 plot_fmin=False, xlim=None, ylim=None):
    if subset is None:
        if kde_plot: subset = True
        else: subset = False
    
    parameters = fitFun.fittingParameters()
    
    kds = parameters.find_Kd_from_dG(variant_table.dG_init)
    if plot_fmin:
        fmax = variant_table.fmin_init
        ylabel_text = 'min'
    else:
        fmax = variant_table.fmax_init
        ylabel_text = 'max'
        
    # find extent in x
    if xlim is None:
        log_kds = np.log10(kds)
        kds_bounds = [np.floor(log_kds.min()), log_kds.median() + log_kds.std()*3]
    else:
        kds_bounds = [np.log10(i) for i in xlim]
    
    #find extent in y
    if ylim is None:
        fmax_bounds = [0, fmax.median() + 3*fmax.std()]
    else:
        fmax_bounds = ylim
    
    # initiate plot
    plt.figure(figsize=(3,3))
    ax = plt.gca()
    if subset:
        x = kds.iloc[::100]
        y = fmax.iloc[::100]
    else:
        x = kds
        y = fmax
    
    if kde_plot:   
        sns.kdeplot(np.log10(x),
                    y,
                    shade=True, shade_lowest=False, n_levels=20, clip=[kds_bounds, fmax_bounds],
                    cmap="binary")
        xticks = ax.get_xticks()
        ax.set_xticklabels(['$10^%d$'%x for x in xticks])
        cutoff = np.log10(cutoff)
    else:
        ax.hexbin(x,
                  y, xscale='log',
                  extent=np.hstack([kds_bounds, fmax_bounds]),
                  cmap="Spectral_r", mincnt=1)
    
    fix_axes(ax)
    plt.xlabel('$K_d$ (nM)')
    
    plt.ylabel('initial $f_{%s}$'%ylabel_text)
    ylim=ax.get_ylim()
    plt.plot([cutoff]*2, ylim, 'r:', label='cutoff for 95% bound')
    plt.tight_layout()

def plotFmaxStdeVersusN(fmaxDist, stds_object, maxn, ax=None):
    # plot
    x = stds_object.index.tolist()
    y = stds_object.loc[:, 'std']
    params = fmaxDist.params
    x_fit = np.arange(1, maxn)
    y_fit = fmaxDist.sigma_by_n_fit(params, x_fit)
    if ax is None:
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        marker='o'
        color='k'
        linestyle='-'
        linecolor = 'c'
    else:
        marker='.'
        color='0.5'
        linestyle=':'
        linecolor =color     
    ax.scatter(x, y, s=10, marker=marker, color=color)
    ax.plot(x_fit, y_fit, linestyle=linestyle, color=linecolor)
    plt.xlabel('number of measurements')
    plt.ylabel('standard deviation of median fmax')
    plt.xlim(0, maxn)
    fix_axes(ax)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    return ax

def plotFmaxOffsetVersusN(fmaxDist, stds_object, maxn, ax=None):
    x = stds_object.index.tolist()
    y = stds_object.offset.values
    yerr = stds_object.offset_stde.values

    if ax is None:
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        marker='o'
        color='k'
        linestyle='-'
        linecolor = 'c'
    else:
        marker='.'
        color='0.5'
        linestyle=':'
        linecolor =color    

    plt.errorbar(x, y, yerr=yerr, fmt='.', color=color, marker=marker, markersize=3)
    plt.axhline(0, color=linecolor, linestyle=linestyle)
    #plt.plot(x, y_smoothed, 'c')
    plt.xlabel('number of measurements')
    plt.ylabel('offset')
    plt.xlim(0, maxn)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    fix_axes(plt.gca())
    return

def plotNumberVersusN(n_tests, maxn):
    plt.figure(figsize=(4, 3))
    sns.distplot(n_tests, bins=np.arange(maxn), kde=False, color='grey',
                 hist_kws={'histtype':'stepfilled'})
    plt.xlabel('number of measurements')
    plt.ylabel('count')
    plt.xlim(0, maxn)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    fix_axes(plt.gca())
    

def plotFmaxInit(variant_table):
    fmax_subset = variant_table.loc[~variant_table.flag.astype(bool)].fmax
    bounds = [fmax_subset.median()-3*fmax_subset.std(), fmax_subset.median() + 3*fmax_subset.std()]
    parameters = fitFun.fittingParameters()
    detection_limit = -6.93
    cmap = sns.diverging_palette(220, 20, center="dark", as_cmap=True)
    index = variant_table.loc[variant_table.numClusters >= 5].index
    
    x = parameters.find_Kd_from_dG(variant_table.loc[index].dG_init.astype(float))
    y = parameters.find_Kd_from_dG(variant_table.loc[index].dG.astype(float))
    
    xlim = [1, 1E5]
    fig = plt.figure(figsize=(4.5,3.75))
    ax = fig.add_subplot(111, aspect='equal')
    im = ax.scatter(x, y, marker='.', alpha=0.5,
                    c=variant_table.loc[index].fmax_init, vmin=bounds[0], vmax=bounds[1], cmap=cmap, linewidth=0)
    plt.plot(xlim, xlim, 'c:', linewidth=1)
    plt.plot([detection_limit]*2, xlim, 'r:', linewidth=1)
    plt.plot(xlim, [detection_limit]*2, 'r:', linewidth=1)
    plt.xlim(xlim); plt.xlabel('$K_d$ initial (kcal/mol)')
    plt.ylim(xlim); plt.ylabel('$K_d$ final (kcal/mol)')
    plt.colorbar(im, label='fmax initial')
    ax.tick_params(top='off', right='off')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    return

def plotBoundFluorescence(signal, bounds):
    """ Plot histogram of all RNA fluorescence and bounds imposed on distribution."""
    lowerbound, upperbound  = bounds
    binwidth = (upperbound - lowerbound)/50.
    plt.figure(figsize=(4,3))
    sns.distplot(signal.dropna(), bins = np.arange(signal.min(), signal.max()+binwidth, binwidth), color='seagreen')
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.plot([lowerbound]*2, ylim, 'k:')
    plt.plot([upperbound]*2, ylim, 'k:')
    plt.xlim(0, upperbound + 2*signal.std())
    plt.xlabel('all cluster fluorescence')
    plt.ylabel('probability density')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fix_axes(ax)
    plt.tight_layout()

def plotErrorInBins(variant_table, xdelta=None):
    parameters = fitFun.fittingParameters()
    variant_table = variant_table.astype(float)
    errors = variant_table.dG_ub - variant_table.dG_lb
    plotErrorBars(parameters.find_Kd_from_dG(variant_table.dG),
                  variant_table.numTests, errors, xdelta=xdelta)
    return

def plotPercentErrorInBins(variant_table, xdelta=None):
    parameters = fitFun.fittingParameters()
    variant_table = variant_table.astype(float)
    errors = ((parameters.find_Kd_from_dG(variant_table.dG_ub) -
               parameters.find_Kd_from_dG(variant_table.dG_lb))/
               parameters.find_Kd_from_dG(variant_table.dG))*100
    ylim = [0, 250]
    yticks = np.arange(0, 250, 50)
    ylabel = 'percent error on Kd'
    plotErrorBars(parameters.find_Kd_from_dG(variant_table.dG),
                  variant_table.numTests, errors,
                  ylim=ylim, yticks=yticks,
                  ylabel=ylabel, xdelta=xdelta)

def plotErrorBars(kds, numTests, errors, ylim=None, yticks=None, ylabel=None, xdelta=None):

    binedges = np.power(10., [0, 1, 2, 3, 4, 5])
    binned_Kds = np.digitize(kds, binedges)


    
    if ylim is None:
        ylim = [0, 1.5]
    if yticks is None:
        yticks = np.arange(0, 1.5, 0.25)
    if ylabel is None:
        ylabel = 'confidence interval width (kcal/mol)'
    if xdelta is None:
        xdelta = 5

    numbers = np.unique(numTests)
    xticks = np.arange(0, len(numbers), xdelta)[1:]
    xticklabels = ['%d'%n  for n in xticks ]

    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(len(binedges)-1, 1)
    for i, bin_idx in enumerate(np.arange(1, len(binedges))):
        index = binned_Kds==bin_idx
        ax = fig.add_subplot(gs[i])
        sns.barplot(x=numTests.loc[index],
                    y=errors.loc[index],
                    ax=ax,
                    order=numbers,
                    color="r",
                    edgecolor="r",
                    error_kw={'elinewidth':0.5})
        ax.set_xticks(xticks)
        if bin_idx == len(binedges)-1:
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('number of tests')
        else:
            ax.set_xticklabels('')
            ax.set_ylabel('')
            
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)

        ax.tick_params(top='off', right='off')
        ax.annotate('$10^%d \leq K_d \leq 10^%d$ nM'%(np.log10(binedges[i]),
                                                   np.log10(binedges[i+1])),
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top',
                    fontsize=12)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.annotate(ylabel, rotation=90,
                 xy=(0.05, 0.5),
                 xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=12)
    return


        
def plotNumberInBins(variant_table, xdelta=None):
    if xdelta is None:
        xdelta=5
    parameters = fitFun.fittingParameters()
    kds = parameters.find_Kd_from_dG(variant_table.dG.astype(float))
    numTests = variant_table.numTests
    
    binedges = np.power(10., [0, 1, 2, 3, 4, 5])
    binned_Kds = np.digitize(kds, binedges)

    numbers = np.unique(numTests)
    xticks = np.arange(0, len(numbers), xdelta)[1:]
    xticklabels = ['%d'%n  for n in xticks ]
    ylabel = 'number of variants'
        
    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(len(binedges)-1, 1)
    for i, bin_idx in enumerate(np.arange(1, len(binedges))):
        index = binned_Kds==bin_idx
        ax = fig.add_subplot(gs[i])
        plt.hist(numTests.loc[index].values,
                 bins=numbers,
                 color=sns.xkcd_rgb['charcoal'],
                 rwidth=0.8,
                 alpha=0.5)
        ax.set_xticks(xticks)
        ax.set_xlim(0, xticks[-1]+1)
        if bin_idx == len(binedges)-1:
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('number of tests')
        else:
            ax.set_xticklabels('')
            ax.set_ylabel('')
        #
        #ylim = ax.get_ylim()
        ##ax.set_ylim(ylim)
        #delta = np.ceil(ylim[1]/100.)
        #ax.set_yticks(np.arange(0, delta*100, np.around(delta/4.)*100))

        ax.tick_params(top='off', right='off')
        ax.annotate('$10^%d \leq K_d \leq 10^%d$ nM'%(np.log10(binedges[i]),
                                                   np.log10(binedges[i+1])),
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top',
                    fontsize=12)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.annotate(ylabel, rotation=90,
                 xy=(0.05, 0.5),
                 xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=12)
    return

def plotNumberTotal(variant_table, binedges=None, variant_table2=None):
    bins = np.arange(120)
    fig = plt.figure(figsize=(3.5,2.5))
    hist, binedges, patches = plt.hist(variant_table.numTests.astype(float).values,
                            bins=bins, histtype='stepfilled', alpha=0.5, color=sns.xkcd_rgb['navy blue'])
    plt.plot((binedges[:-1]+binedges[1:])*0.5, hist, linewidth=1, alpha=0.1, color=sns.xkcd_rgb['navy blue'])
    #sns.distplot(variant_table.numTests.astype(float).values, bins=bins,
    #            hist_kws={'histtype':'stepfilled'}, color='grey')
    plt.xlabel('# tests')
    plt.ylabel('# variants')
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params(top='off', right='off')

    if variant_table2 is not None:
        hist, binedges, patches = plt.hist(variant_table2.numTests.astype(float).values,
                                           bins=bins, histtype='stepfilled', alpha=0.5, color=sns.xkcd_rgb['dark cyan'])
        plt.plot((binedges[:-1]+binedges[1:])*0.5, hist, linewidth=1, alpha=0.1, color=sns.xkcd_rgb['dark cyan'])

    ylim = ax.get_ylim()
    plt.plot([5]*2, ylim, 'k--', linewidth=1, alpha=0.5)

def findKdAndError(variant_table):
    parameters = fitFun.fittingParameters()
    kds = parameters.find_Kd_from_dG(variant_table.dG.astype(float))
    error = [(fitFun.errorProgagationKdFromdG(variant_table.dG,
                                             variant_table.dG - variant_table.dG_lb)/
              kds),
             (fitFun.errorProgagationKdFromdG(variant_table.dG,
                                              variant_table.dG_ub - variant_table.dG)/
              kds)]
    fractional_error = pd.concat([variant_table.numTests,
                                  pd.concat(error, axis=1).mean(axis=1)],
        axis=1, keys=['numTests', 'fractional_error'])
    return fractional_error

def plotErrorTotal(variant_table, variant_table2=None):
    
    parameters = fitFun.fittingParameters()
    
    fig = plt.figure(figsize=(4,2.5))
    ax = fig.add_subplot(111)
    order = np.arange(100)

    if variant_table2 is not None:
        index = variant_table2.dG < parameters.find_dG_from_Kd(5000)
        fractional_error = findKdAndError(variant_table2)
        sns.boxplot(x="numTests", y="fractional_error", data=fractional_error.loc[index],
                    whis=1.5, order=order, fliersize=0,
                    linewidth=0.5, color=sns.xkcd_rgb['gold'], ax=ax, showcaps=False,
                    showfliers=False)

    index = variant_table.dG < parameters.find_dG_from_Kd(5000)
    #fractional_error = findKdAndError(variant_table)
    sns.boxplot(x="numTests", y="fractional_error", data=fractional_error.loc[index],
                whis=1.5, order=order, fliersize=0,
                linewidth=0.5, color='grey', ax=ax, showcaps=False, showfliers=False)
    
    if len(order) < 100:
        binwidth = 5
    elif len(order) < 200:
        binwidth = 10
    elif len(order) < 300:
        binwidth = 15
    elif len(order) < 400:
        binwidth = 20
    ax = plt.gca()
    ax.tick_params(top='off', right='off')

    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::binwidth])
    ax.set_xticklabels(['%d'%x for x in order[::binwidth]])

    plt.xlim(0, 60)
    plt.xlabel('# measurements')
    plt.ylabel('fractional error on $K_d$')
    plt.tight_layout()
    ylim = ax.get_ylim()
    plt.plot([5]*2, ylim, 'k--', linewidth=1, alpha=0.5)

def plotFractionFit(variant_table, binedges=np.arange(-12, -6, 0.5), param='dG_init', pvalue_threshold=0.05):
    # plot
    binwidth=0.01
    bins=np.arange(0,1+binwidth, binwidth)
    plt.figure(figsize=(4, 3.5))
    plt.hist(variant_table.loc[variant_table.pvalue <= pvalue_threshold].fitFraction.values,
             alpha=0.5, color='red', bins=bins, label='passing cutoff')
    plt.hist(variant_table.loc[variant_table.pvalue > pvalue_threshold].fitFraction.values,
             alpha=0.5, color='grey', bins=bins,  label='fails cutoff')
    plt.ylabel('number of variants')
    plt.xlabel('fraction fit')
    plt.legend(loc='upper left')
    plt.tight_layout()
    fix_axes(plt.gca())    

    subtable = pd.DataFrame(index=variant_table.index,
                            columns=['binned_dGs', 'pvalueFilter'],
                            data=np.column_stack([np.digitize(variant_table.loc[:, param], binedges),
                                                  variant_table.pvalue <= pvalue_threshold]))
    g = sns.factorplot(x="binned_dGs", y="pvalueFilter", data=subtable,
                order=np.unique(subtable.binned_dGs),
                color="r", kind='bar');
    g.set(ylim=(0, 1.1), ylabel='fraction pass pvalue cutoff');
    g.set_xticklabels(['<%4.1f'%binedges[0]] +
        ['%3.1f:%3.1f'%(i, j) for i, j in itertools.izip(binedges[:-1], binedges[1:])]+
        ['>%4.1f'%binedges[-1]], rotation=90)
    g.set(xticks=np.arange(len(binedges)+1))
    g.fig.subplots_adjust(hspace=.2, bottom=0.35)
    fix_axes(plt.gca())
    
def histogramKds(variant_table):
    """ Find density of set of 'background' Kds to define cutoff. """
    parameters = fitFun.fittingParameters()
    kds = parameters.find_Kd_from_dG(variant_table.dG.astype(float))
    cutoff = np.percentile(kds, 1)
    cutoff_1 = np.percentile(kds, 0.1)
    binedges = np.linspace(0, 6, 100)
    plt.figure(figsize=(4,3))
    plt.hist(np.log10(kds).values, binedges, histtype='stepfilled', color='grey', normed=True,
             alpha=0.5)

    ax = plt.gca()
    ax.tick_params(top='off', right='off')
    xticks = np.arange(0, 6)
    plt.xticks(xticks, ['$10^{%d}$'%x for x in xticks])
    ylim = ax.get_ylim()
    plt.plot([np.log10(cutoff)]*2, ylim, 'r:', linewidth=1)
    plt.plot([np.log10(cutoff_1)]*2, ylim, 'r--', linewidth=1)
    plt.xlabel('fit $K_d$')
    plt.ylabel('probability')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.annotate(('cutoff (1%% FDR) = %d nM, %4.2f kcal/mol\n'
                  'cutoff (0.1%% FDR) = %d nM, %4.2f kcal/mol')
                  %(cutoff, parameters.find_dG_from_Kd(cutoff),
                    cutoff_1, parameters.find_dG_from_Kd(cutoff_1)),
                 xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=10)
    plt.tight_layout()
    
def plotScatterPlotColoredByFlag(results, results_dropped, concentrations, numPointsLost, plotAll=None):
    if plotAll is None:
        plotAll = False
    parameters = fitFun.fittingParameters()
    # use the flag to determine the color
    c = results_dropped.flag.copy()
    c.loc[results.flag == 1] = -1
    
    kd_original = parameters.find_Kd_from_dG(results.dG.astype(float))
    kd_dropped  = parameters.find_Kd_from_dG(results_dropped.dG.astype(float))
    
    cmap = sns.diverging_palette(20, 220, center="dark", as_cmap=True, )

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, aspect='equal')
    xlim = [1E0, 5E2]
    
    index = (results.flag == 0)&(results_dropped.flag==1)
    plt.scatter(kd_original.loc[index], kd_dropped.loc[index], marker='.', alpha=0.8, s=20,
                facecolors="#26AFE5", edgecolors="#26AFE5", linewidth=0.1)   

    if plotAll:
        index = (results.flag == 0)&(results_dropped.flag==0)
        plt.scatter(kd_original, kd_dropped, marker='.', alpha=0.5, s=20,
                    facecolors="k", edgecolors="k", linewidth=0.1)
        
        index = (results.flag == 1)&(results_dropped.flag==1)
        plt.scatter(kd_original, kd_dropped, marker='.', alpha=0.5, s=20,
                    facecolors="EF4036", edgecolors="EF4036", linewidth=0.1)
        plotted_x = np.log10(kd_original)
        plotted_y = np.log10(kd_dropped)
    else:
        plotted_x = np.log10(kd_original.loc[index])
        plotted_y = np.log10(kd_dropped.loc[index])
    
    #plt.xticks(np.arange(xlim[0], xlim[1]))
    plt.xlim(xlim); plt.xlabel('$K_d$ (nM) all')
    plt.ylim(xlim); plt.ylabel('$K_d$ (nM) dropped')
    plt.plot(xlim, xlim, 'r', linewidth=0.5)
    #plt.plot([concentrations[-1]]*2, xlim, 'k:', linewidth=0.5)
    plt.plot(xlim, [concentrations[-numPointsLost]]*2, 'k:', linewidth=0.5)

    ax.tick_params(top='off', right='off')
    ax.tick_params(which="minor", top='off', right='off')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.annotate('$R^2$=%4.2f'%(st.pearsonr(plotted_x,plotted_y)[0]**2),
                 xy=(0.95, 0.05),
                 xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=10)
    plt.tight_layout()
    return


def plotDeltaAbsFluorescence(bindingSeries, bindingSeriesBackground, concentrations=None):
    


    if concentrations is None:
        concentrations = ['%4.2fnM'%d for d in 2000*np.power(3., np.arange(0, -8, -1))][::-1]
    numconcentrations = len(concentrations)
    
    cNorm = mpl.colors.Normalize(vmin=0, vmax=numconcentrations-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=False))
    
    
    fig = plt.figure(figsize=(5,8))
    gs = gridspec.GridSpec(numconcentrations, 1)
    bins = np.arange(0, 4000, 10)
    for i in range(numconcentrations):
        ax = fig.add_subplot(gs[i])
        # plot actual
        try:
            counts, xbins, patches = ax.hist(bindingSeries.iloc[:, i].dropna().values,
                                             bins=bins,
                                             histtype='stepfilled', alpha=0.1,
                                             color=scalarMap.to_rgba(i))
            if (counts == 0).sum() > 0:
                index2 = np.arange(0, np.ravel(np.where(counts == 0))[0])
            else:
                index2 = np.arange(len(counts))
            ax.plot(((xbins[:-1] + xbins[1:])*0.5)[index2], counts[index2],
                color=scalarMap.to_rgba(i), label=concentrations[i])
            
            # plot background
            counts, xbins, patches = ax.hist(bindingSeriesBackground.iloc[:, i].dropna().values,
                                             bins=bins,
                                             histtype='stepfilled', alpha=0.1,
                                             color='0.5')
            if (counts == 0).sum() > 0:
                index2 = np.arange(0, np.ravel(np.where(counts == 0))[0])
            else:
                index2 = np.arange(len(counts))
            ax.plot(((xbins[:-1] + xbins[1:])*0.5)[index2], counts[index2],
                color='0.5', label=concentrations[i])
        except:
            pass
        # set labels
        ax.set_yscale('log')
        ax.set_ylim(1, 10**5)
        ax.set_yticks(np.power(10, range(0, 5)))
        ax.tick_params(top='off', right='off')
        ax.tick_params(which="minor", top='off', right='off')
        #ax.set_ylabel('number of clusters')
        if i == numconcentrations-1:
            ax.set_xlabel('absolute fluorescence')
        else:
            ax.set_xticklabels([])
        ax.annotate('%4.1f nM'%(concentrations[i]),
                    xy=(0.95, 0.90),
                    xycoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top',
                    fontsize=12)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.annotate('number of clusters', rotation=90,
                 xy=(0.05, 0.5),
                 xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=12)
    pass

def plotManyExperiments(deltaGs, expts=None, xlim=[-13 -5]):
    if expts is None:
        expts = deltaGs.columns.tolist()

    numgrid = len(expts)-1
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(numgrid, numgrid)
    for i in np.arange(0, len(expts)):
        for j in np.arange(0, i):
            index = deltaGs.loc[:, expts].dropna().index
            x = deltaGs.loc[index, expts[j]]
            y = deltaGs.loc[index, expts[i]]
            ax = fig.add_subplot(gs[i-1, j])
            ax.hexbin(deltaGs.loc[:, expts[j]], deltaGs.loc[:, expts[i]],
                     cmap='Spectral_r', mincnt=1)
            ax.set_ylim(xlim)
            ax.set_xlim(xlim)
            ax.plot(xlim, xlim, 'k')
            fix_axes(ax)
            if i==numgrid:
                ax.set_xlabel(expts[j])
            else:
                ax.set_xticklabels([])
            if j==0:
                ax.set_ylabel(expts[i])
            else:
                ax.set_yticklabels([])
            ax.annotate('R2=%4.2f'%st.pearsonr(x, y)[0]**2, xy=(0.95, 0.05),
                 xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=11)

def plotReplicatesKd(variant_tables,
                   log=None, vmax=None, scatter=None, enforce_numTests=None):

    if log is None:
        log = False
    if log:
        bins = 'log'
    else:
        bins = None
    if scatter is None:
        scatter = False
    if enforce_numTests is None:
        enforce_numTests = True
    
    parameters = fitFun.fittingParameters()
    cutoff = parameters.cutoff_dG
    
    if enforce_numTests:
        index = (pd.concat(variant_tables, axis=1).loc[:, 'numTests'] >=5).all(axis=1)
    else:
        index = pd.concat(variant_tables, axis=1).index
    
    x = parameters.find_Kd_from_dG(variant_tables[0].loc[index].dG.astype(float))
    y = parameters.find_Kd_from_dG(variant_tables[1].loc[index].dG.astype(float))
        
    fig = plt.figure(figsize=(4,2.5))
    ax = fig.add_subplot(111, aspect='equal')
    if scatter:
        ax.scatter(x, y, alpha=0.1, marker='.', c='k', s=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        im = ax.hexbin(x, y, bins=bins, mincnt=1, xscale='log', yscale='log',
                       cmap='Spectral_r',
                       #extent=[-12, cutoff, -12, cutoff],
                        gridsize=150)
    
    plt.xlabel('$K_d$ rep 1 (nM)')
    plt.ylabel('$K_d$ rep 2 (nM)')
    ax.tick_params(right='off', top='off')
    
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    
    xlim = np.array([np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])])
    
    index_sub = (pd.concat(variant_tables, axis=1).loc[index, 'dG'] < cutoff).all(axis=1)
    slope, intercept, r_value, p_value, std_err = st.linregress(
        np.log10(x.loc[index_sub]),
        np.log10(y.loc[index_sub]))
    plt.plot(xlim, np.power(10, slope*np.log10(xlim)+intercept), 'r:', linewidth=1)

    plt.annotate('$R^2$=%4.2f'%(r_value**2),
                 xy=(0.15, 0.01),
                 xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='bottom',
                 fontsize=12)
    
    plt.xlim(xlim)
    plt.ylim(xlim)

    ax.tick_params(top='off', right='off')
    ax.tick_params(which='minor', top='off', right='off')
    #plt.xticks(xticks, ['$10^{%d}$'%x for x in xticks])
    #plt.yticks(xticks, ['$10^{%d}$'%x for x in xticks])
    
    plt.plot([parameters.find_Kd_from_dG(cutoff)]*2, ylim, 'k--', linewidth=1)
    plt.plot(xlim, [parameters.find_Kd_from_dG(cutoff)]*2, 'k--', linewidth=1)
    if not scatter:
        plt.colorbar(im)
    #plt.yticks(np.arange(0, 60, 10))
    plt.tight_layout()

def plotResidualsKd(variant_tables):
    parameters = fitFun.fittingParameters()
    cutoff = np.log10(parameters.cutoff_kd)
    index = (pd.concat(variant_tables, axis=1).loc[:, 'numTests'] >=5).all(axis=1)
    
    x = np.log10(parameters.find_Kd_from_dG(variant_tables[0].loc[index].dG.astype(float)))
    y = np.log10(parameters.find_Kd_from_dG(variant_tables[1].loc[index].dG.astype(float)))

    numTests_x = variant_tables[0].loc[index].numTests
    numTests_y = variant_tables[1].loc[index].numTests

    index = (x < cutoff)&(y < cutoff)
    z =  y - (y - x).loc[index].mean()
    residuals = pd.concat([(x - y), (x+y)/2.], axis=1, keys=['diff', 'value'])
    plt.figure(figsize=(4,3))
    plt.hist(residuals.loc[residuals.value < np.log10(5000), 'diff'],
             bins=np.linspace(-1, 1, 100), histtype='stepfilled', alpha=0.5,
             color=sns.xkcd_rgb['navy blue'])
    
    residuals = pd.concat([np.power(10, x) - np.power(10, y),
                           (np.power(10, x) + np.power(10, y))/2.],
        axis=1, keys=['diff', 'value'])
    plt.figure(figsize=(3,3))
    plt.hist((residuals.loc[residuals.value < 5000, 'diff']/
              residuals.loc[residuals.value < 5000, 'value']).values,
             bins=np.linspace(-2, 1, 100), histtype='stepfilled', alpha=0.5,
             color=sns.xkcd_rgb['navy blue'])
    plt.xlabel('fraction of Kd')
    plt.ylabel('number of variants')
    ax = plt.gca()
    ax.tick_params(right='off', top='off')
    plt.tight_layout()
    
def plotNumberOfTilesFitRates(tileMap, finalTimes):
    fig = plt.figure(figsize=(5,7));
    gs = gridspec.GridSpec(1, 2, wspace=0.05, width_ratios=[2,1],
                           bottom=0.15, left=0.15, top=0.95, right=0.95)
    ax = fig.add_subplot(gs[0,0])
    sns.heatmap(tileMap.transpose(),  linewidths=.5, cbar=False, ax=ax,
                yticklabels=finalTimes.astype(int).values)
    ax.set_xlabel('tile')
    ax.set_ylabel('time (s)')
    
    color = sns.cubehelix_palette()[-1]      
    ax = fig.add_subplot(gs[0,1])
    ax.barh(np.arange(tileMap.shape[1]), (tileMap>0).sum(axis=0)[::-1],
            facecolor=color, edgecolor='w', linewidth=0.5,
            )
    ax.set_ylim(0, tileMap.shape[1]-1)
    ax.set_xlim(0, tileMap.shape[0])
    ax.set_yticks(np.arange(tileMap.shape[1])+0.5)
    ax.set_yticklabels([])
    majorLocator   = mpl.ticker.MultipleLocator(5)
    majorFormatter = mpl.ticker.FormatStrFormatter('%d')
    minorLocator   = mpl.ticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.set_xlabel('# tiles')
    fix_axes(ax)
    sns.despine()


def plotTimeDeltaDist(time_deltas, min_time_delta):
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    binwidth = 0.1 # seconds
    binstart = min_time_delta-binwidth*10
    binend = min_time_delta+binwidth*100
    bins = np.arange(binstart, binend+binwidth, binwidth)
    sns.distplot(time_deltas, ax=ax, bins=bins, kde=False, hist_kws={'histtype':'stepfilled'})
    ax.axvline(min_time_delta, color='k', linewidth=1, linestyle=':' )
    plt.xlabel('time deltas (s)')
    plt.xlim(binstart, binend)
    fix_axes(ax)
    plt.tight_layout()
    

def plotTimesScatter(timeMap, finalTimes):
    color = sns.cubehelix_palette()[-1]      
    fig = plt.figure(figsize=(5,3));
    for i, time in enumerate(finalTimes):
        y = timeMap.iloc[:, i].dropna()
        x = i + st.norm.rvs(loc=0, scale=0.1, size=len(y))
        plt.scatter(x, y, marker='o', s=10, facecolors='none', edgecolors=color)
        plt.scatter(i, time, marker='x', s=30, facecolors='none', edgecolors=color, linewidth=1.25)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    plt.xlim(-len(finalTimes)*0.01, len(finalTimes)*1.01)
    plt.xticks(np.arange(len(finalTimes)))
    plt.xlabel('time points')

    plt.ylim(0 - (ylim[1]-ylim[0])*0.01, ylim[1])
    plt.ylabel('time (s)')
    
    ax.tick_params(top='off', right='off')
    ax.tick_params(which='minor', top='off', right='off')
    plt.tight_layout()
    majorLocator   = mpl.ticker.MultipleLocator(5)
    minorLocator   = mpl.ticker.MultipleLocator(1)
    majorFormatter = mpl.ticker.FormatStrFormatter('%d')
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)    
    fix_axes(ax)

def plotTimesOriginal(timeDelta):
    plt.figure(figsize=(5, 3))
    colors = sns.color_palette('Paired', 10) + sns.color_palette('Paired', 10)
    tiles = np.sort(timeDelta.keys())
    for i, tile in enumerate(tiles):
        times = timeDelta[tile]
        x = np.arange(len(times))
        y = times
        if i%3 == 0:
            fmt = ':'
            marker = 'o'
        elif (i-1)%3==0:
            fmt = '-'
            marker = '<'
        else:
            fmt = '--'
            marker = '*'
        plt.plot(x, y, fmt, color=colors[i], marker=marker, label=tile, linewidth=1)
    plt.legend(loc='upper left', ncol=2)
    plt.xticks(np.arange(np.max([len(vec) for vec in timeDelta.values()])))
    plt.xlabel('time points')
    plt.ylabel('time (s)')
    ax = plt.gca()
    ax.tick_params(top='off', right='off')
    plt.tight_layout()
    fix_axes(ax)
    
def plotReplicates(variant_tables, cutoff=None, relative=None, variant=None,
                   log=None, vmax=None):
    if cutoff is None:
        cutoff = -7
    if relative is None:
        relative = False
    if variant is None:
        variant = 34429
    if log is None:
        log = False
    if log:
        bins = 'log'
    else:
        bins = None
    
    index = ((pd.concat(variant_tables, axis=1).loc[:, 'dG'] < -7).all(axis=1)&
        ((pd.concat(variant_tables, axis=1).loc[:, 'numTests']) >=5).all(axis=1))
    
    x = variant_tables[0].loc[index].dG
    y = variant_tables[1].loc[index].dG
    
    if relative:
        x = x - x.loc[variant]
        y = y - y.loc[variant]
        text = '$\Delta\Delta$'
        
    else:
        text = '$\Delta$'
    
    fig = plt.figure(figsize=(4,2.5))
    ax = fig.add_subplot(111, aspect='equal')
    im = ax.hexbin(x, y, bins=bins, mincnt=0, vmin=0, vmax=vmax,
           #extent=[-12, cutoff, -12, cutoff],
           gridsize=75)
    
    plt.xlabel('%s$G$ rep 1 (kcal/mol)'%text)
    plt.ylabel('%s$G$ rep 2 (kcal/mol)'%text)
    ax.tick_params(right='off', top='off')
    
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    
    slope, intercept, r_value, p_value, std_err = st.linregress(x,y)
    plt.plot(xlim, slope*xlim+intercept, 'r:', linewidth=1)

    plt.annotate('$R^2$=%4.2f'%(r_value**2),
                 xy=(0.95, 0.05),
                 xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=12)
    plt.xlim(xlim[0], x.max())
    plt.ylim(ylim[0], y.max())
    plt.colorbar(im)
    #plt.yticks(np.arange(0, 60, 10))
    plt.tight_layout()

def plotColoredByLength(variant_tables, cutoff=None, density=None):
    if cutoff is None:
        cutoff = -7
    if density is None:
        density=False
    index = ((pd.concat(variant_tables, axis=1).loc[:, 'dG'] < cutoff).all(axis=1)&
        ((pd.concat(variant_tables, axis=1).loc[:, 'numTests']) >=5).all(axis=1))

    colors =  ["#e74c3c", "#3498db", "#34495e", "#9b59b6","#2ecc71"]
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_subplot(111)
    for i, length in enumerate([10, 9, 11]):
        index2 = index&(variant_tables[0].length == length)
        if density:
            plt.hexbin(variant_tables[0].loc[index2].dG, variant_tables[1].loc[index2].dG,
                       alpha=0.5,
                       label='%dbp'%length,
                       cmap=sns.light_palette(colors[i], as_cmap=True),
                       mincnt=1,
                       bins='log')
        else:
            plt.scatter(variant_tables[0].loc[index2].dG, variant_tables[1].loc[index2].dG,
                    marker='.', alpha=0.5, facecolors=colors[i], edgecolors='none', label='%dbp'%length)
    plt.legend()
    
    plt.xlabel('$\Delta$$G$ rep 1 (kcal/mol)')
    plt.ylabel('$\Delta$$G$ rep 2 (kcal/mol)')
    ax.tick_params(right='off', top='off')
    ax.set_position([0.2, 0.25, 0.5, 0.65])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    plt.xlim(xlim[0], cutoff)
    plt.ylim(ylim[0], cutoff)
    #plt.tight_layout()
    
def plotDeltaDeltaGByLength(variant_tables, cutoff=None, normed=None):
    if cutoff is None:
        cutoff = -7
    if normed is None:
        normed = False
    index = ((pd.concat(variant_tables, axis=1).loc[:, 'dG'] < cutoff).all(axis=1)&
        ((pd.concat(variant_tables, axis=1).loc[:, 'numTests']) >=5).all(axis=1))
    ddG = variant_tables[1].dG - variant_tables[0].dG
    
    bins = np.linspace(ddG.loc[index].min(),ddG.loc[index].max() )
    colors =  ["#e74c3c", "#3498db", "#34495e", "#9b59b6","#2ecc71"]
    fig = plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot(111)
    for i, length in enumerate([10, 9, 11]):
        index2 = index&(variant_tables[0].length == length)
        if normed:
            sns.distplot(ddG.loc[index2], hist_kws={'histtype':'stepfilled'},
                     label='%dbp'%length, color=colors[i])
        else:
            plt.hist(ddG.loc[index2].values, bins=bins, histtype='stepfilled',
                     label='%dbp'%length, color=colors[i], alpha=0.5)
    plt.legend(loc='upper left')
    ax.tick_params(right='off', top='off')
    plt.xlabel('$\Delta\Delta$G (kcal/mol)')
    plt.ylabel('probability')
    plt.tight_layout()

def plotAbsoluteFluorescenceInLibraryBins(bindingSeriesNormLabeled, libCharFile,
                                          cutoff=None, bindingPoint=None,
                                          ylim=None):
    if bindingPoint is None:
        bindingPoint = -1
    if ylim is None: ylim = (0, 1.5)
    libChar = pd.read_table(libCharFile)
    grouped = bindingSeriesNormLabeled.groupby('variant_number')
    
    fluorescence = grouped.median().iloc[:, bindingPoint]
    data = pd.concat([libChar.length,
                      libChar.loop + '_' + libChar.receptor,
                      fluorescence], axis=1)
    data.columns = ['length', 'loop_receptor', 'fluorescence']
    
    g = sns.FacetGrid(data, col='loop_receptor', col_wrap=4,  ylim=ylim,
                      size=2, aspect=1.4,)
                      
    g.map(sns.boxplot, "length", "fluorescence",
                                 order=[8,9,10,11,12],
                                 whis=1.5, fliersize=0,
                linewidth=0.5, color='grey', showcaps=False, showfliers=False)

    # plot cutoff if present

    for ax in g.axes.flat:
        xlim = ax.get_xlim()
        ax.plot(xlim, [cutoff]*2, c="r", ls=":", linewidth=1)
        
    plt.annotate('%s nM'%fluorescence.name,
             xy=(0.95, 0.05),
             xycoords='figure fraction',
             horizontalalignment='right', verticalalignment='bottom',
             fontsize=12)

def plotLibraryFigure(matrixAll, labelMat, libChar, labels=None, positions=None,
                      allLengths=None):
    if labels is None:
        labels = ['sequence_8', 'sequence_9', 'sequence_10', 'sequence_11','sequence_12',
                    '0x1', '0x2', '0x3',
                    '1x0', '2x0', '3x0',
                    '1x1', "1x1'",
                    '1x2', "1x2'",
                    '1x3', "1x3'",
                    '2x1', "2x1'",
                    '3x1', "3x1'",
                    '2x2',  '3x3', "3x3'"]
    if positions is None:
        positions = range(-3, 5)
    if allLengths is None:
        allLengths = [8, 9, 10, 11, 12]
    numLengths = len(allLengths)
    width_keys = ['%s_0'%key for key in labels]
    width_ratios  = np.array([np.log10(len(matrixAll[labelMat[key]].dropna(axis=0, how='all')))
                                          for key in width_keys])
    with sns.axes_style("whitegrid", {'grid.linestyle': u':', 'axes.edgecolor': '0.9'}):
        fig = plt.figure(figsize=(6, 3.5))
        gs = gridspec.GridSpec(2, len(labels), wspace=0, hspace=0.05,
                               width_ratios=width_ratios,
                               height_ratios=[1,4],
                               bottom=0.25, right=0.97, left=0.1, top=0.97)
        markers = ['^', 'o', 'v']
        cmap = sns.cubehelix_palette(start=0.75, rot=1.25, light=0.40, dark=0.05, reverse=True, hue=1, as_cmap=True)
        colors= ['black', 'red brown', 'orange brown', 'greenish blue', 'dark teal']
        colors= ['black', 'red brown', 'tomato red', 'blue', 'navy blue', 'plum', 'milk chocolate' , 'medium green']
        for i, key in enumerate(labels):
            ax = fig.add_subplot(gs[1, i])
            number = pd.Series(index=positions)
            numPossible = len(matrixAll[labelMat['%s_%d'%(key, 0)]])
            for j, position in enumerate(positions):
                try:
                    a = matrixAll[labelMat['%s_%d'%(key, position)]].dropna(axis=0, how='all')
                    indices = np.hstack(np.column_stack(a.values)).astype(float)
                    index = np.isfinite(indices)
                    
                    x = np.array(a.index.tolist())/float(np.max(a.index.tolist()))
                    x = np.hstack([x]*numLengths)[index]
                    
                    y = libChar.loc[indices[index]].length.values
                    jitter = st.norm.rvs(loc=position*0.15, scale=0.05, size=len(y))
                    c = (libChar.loc[indices[index]].helix_one_length).fillna(0).values
                    if not key.find('sequence') == 0:
                        c += 1
                    else:
                        c[:] = 0
                    for k in range(5):
                        index = c == k
                        ax.scatter(x[index], (y+jitter)[index], s=1, marker='.',
                                   facecolors=sns.xkcd_rgb[colors[k]], edgecolors='none')
                    number.loc[position] = (len(a))
                except:
                    pass

            plt.ylim(7.5, 12.5)
            plt.xlim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([])
            ax.set_yticks(allLengths)
            if i != 0:
                ax.set_yticklabels([])
            
            
            if key.find('sequence')==0:
                ax.annotate('%s'%(key),
                             xy=(0.5, -0.01),
                             xycoords='axes fraction',
                             horizontalalignment='center', verticalalignment='top',
                             rotation=90,
                             fontsize=10)
            else:
                ax.annotate('%s'%(key),
                             xy=(0.5, -0.13),
                             xycoords='axes fraction',
                             horizontalalignment='center', verticalalignment='bottom',
                             rotation=90,
                             fontsize=10)   
            # plot second box
            n = number.loc[0]
            scaled_width = width_ratios.min()/np.log10(n)
            aspect = 0.1
            ax = fig.add_subplot(gs[0, i])
            ax.bar(left=[0.4], height=np.log10(numPossible), width=scaled_width*0.5,
                   facecolor='grey', edgecolor='0.9', alpha=0.5)
            ax.bar(left=[0.6], height=np.log10(n), width=scaled_width*0.5,
                   facecolor=sns.xkcd_rgb['charcoal'], edgecolor='0.9', alpha=1)
        
            
            #ax.bar(left=[0], height=numPossible/np.log10(numPossible), width=1, color='grey', alpha=0.5)
            #ax.bar(left=[0], height=n/(aspect*np.log10(numPossible)), width=aspect, color='navy', alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([0,1])
            ax.set_ylim([0, 6])
            ax.set_yticks(np.arange(6))
            ax.set_yticklabels(['$10^%d$'%y if (y-1)%2==0 else '' for y in np.arange(6)] )
            #ax.set_yscale('log')
            ax.set_xticklabels([])
            if i != 0:
                ax.set_yticklabels([])
                
def plotNormalizedFitCurve(concentrations, subSeries, result, fitParameters):
    
    fitFun.plotFitCurve(concentrations,
                        subSeries,
                        result,
                        fitParameters)
    ax = plt.gca()
    majorLocator   = mpl.ticker.MultipleLocator(result.fmax*0.5)
    minorLocator   = mpl.ticker.MultipleLocator(result.fmax*0.1)
    majorFormatter = mpl.ticker.FormatStrFormatter('%4.1f')
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_ylim(0, result.fmax*1.1)
    ax.set_yticklabels([])
    #
    #normalized_ticks = np.linspace(0, result.fmax, 5)
    #
    #ax.set_yticks(normalized_ticks)
    #ax.set_yticklabels([0, 0.5, 1])
