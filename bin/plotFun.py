"""
Sarah Denny
Stanford University

"""

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import itertools  
import seaborn as sns
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
  


def plotFmaxInit(variant_table):
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
                    c=variant_table.loc[index].fmax_init, vmin=0.5, vmax=1.5, cmap=cmap, linewidth=0)
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

def plotErrorInBins(variant_table):
    parameters = fitFun.fittingParameters()
    variant_table = variant_table.astype(float)
    errors = variant_table.dG_ub - variant_table.dG_lb
    plotErrorBars(parameters.find_Kd_from_dG(variant_table.dG),
                  variant_table.numTests, errors)
    return

def plotPercentErrorInBins(variant_table):
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
                  ylabel=ylabel)

def plotErrorBars(kds, numTests, errors, ylim=None, yticks=None, ylabel=None):

    binedges = np.power(10., [0, 1, 2, 3, 4, 5])
    binned_Kds = np.digitize(kds, binedges)

    numbers = np.unique(numTests)
    xticks = np.arange(0, len(numbers), 5)[1:]
    xticklabels = ['%d'%n  for n in xticks ]
    
    if ylim is None:
        ylim = [0, 1.5]
    if yticks is None:
        yticks = np.arange(0, 1.5, 0.25)
    if ylabel is None:
        ylabel = 'confidence interval width (kcal/mol)'
        
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


        
def plotNumberInBins(variant_table):
    parameters = fitFun.fittingParameters()
    kds = parameters.find_Kd_from_dG(variant_table.dG.astype(float))
    numTests = variant_table.numTests
    
    binedges = np.power(10., [0, 1, 2, 3, 4, 5])
    binned_Kds = np.digitize(kds, binedges)

    numbers = np.unique(numTests)
    xticks = np.arange(0, len(numbers), 5)[1:]
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
        ax.set_xlim(xticks[0]-1, xticks[-1]+1)
        if bin_idx == len(binedges)-1:
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('number of tests')
        else:
            ax.set_xticklabels('')
            ax.set_ylabel('')
        
        ylim = ax.get_ylim()
        #ax.set_ylim(ylim)
        delta = np.ceil(ylim[1]/100.)
        ax.set_yticks(np.arange(0, delta*100, np.around(delta/4.)*100))

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

def plotFractionFit(variant_table, binedges=None):
    # plot
    binwidth=0.01
    bins=np.arange(0,1+binwidth, binwidth)
    plt.figure(figsize=(4, 3.5))
    plt.hist(variant_table.loc[variant_table.pvalue <= 0.05].fitFraction.values, alpha=0.5, color='red', bins=bins)
    plt.hist(variant_table.loc[variant_table.pvalue > 0.05].fitFraction.values, alpha=0.5, color='grey', bins=bins)
    plt.ylabel('number of variants')
    plt.xlabel('fraction fit')
    plt.tight_layout()
    
    if binedges is None:
        binedges = np.arange(-12, -6, 0.5)
    subtable = pd.DataFrame(index=variant_table.index,
                            columns=['binned_dGs', 'pvalueFilter'],
                            data=np.column_stack([np.digitize(variant_table.dG, binedges),
                                                  variant_table.pvalue <= 0.05]))
    g = sns.factorplot(x="binned_dGs", y="pvalueFilter", data=subtable,
                order=np.unique(subtable.binned_dGs),
                color="r", kind='bar');
    g.set(ylim=(0, 1.1), );
    g.set_xticklabels(['%3.1f:%3.1f'%(i, j) for i, j in itertools.izip(binedges[:-1], binedges[1:])]+['>%4.1f'%binedges[-1]], rotation=90)
    g.set(xticks=np.arange(len(binedges)))
    g.fig.subplots_adjust(hspace=.2, bottom=0.35)
    
def histogramKds(variant_table):
    
    parameters = fitFun.fittingParameters()
    kds = parameters.find_Kd_from_dG(variant_table.dG.astype(float))
    cutoff = np.percentile(kds, 1)
    binedges = np.logspace(0, 6, 100)
    plt.figure(figsize=(4,3))
    plt.hist(kds.values, binedges, histtype='stepfilled', color='grey', normed=True,
             alpha=0.5)

    ax = plt.gca()
    ax.set_xscale('log')
    ax.tick_params(top='off', right='off')
    ylim = ax.get_ylim()
    plt.plot([cutoff]*2, ylim, 'r:')
    plt.xlabel('fit $K_d$')
    plt.ylabel('probability')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.annotate('cutoff = %d nM, %4.2f kcal/mol'%(cutoff,
                                                   parameters.find_dG_from_Kd(cutoff)),
                 xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=10)
    plt.tight_layout()
    

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
    