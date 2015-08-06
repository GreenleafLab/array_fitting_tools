import os
import sys
import time
import re
import uuid
import subprocess
import numpy as np
import pandas as pd
import histogram
import plotfun
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import CPlibs
import scipy.stats as st
#import heatmapfun
from scikits.bootstrap import bootstrap
from statsmodels.stats.weightstats import DescrStatsW
import IMlibs
import scipy.cluster.hierarchy as sch


class Parameters():
    def __init__(self):
    
        # save the units of concentration given in the binding series
        self.RT = 0.582  # kcal/mol at 20 degrees celsius
        self.min_deltaG = -12
        self.max_deltaG = -3
        self.concentration_units = 1E-9
        self.concentrations = np.array([2000./np.power(3, i) for i in range(8)])[::-1]
        self.lims = {'qvalue':(2e-4, 1e0),
                    'dG':(-12, -4),
                    'koff':(1e-5, 1E-2),
                    'kon' :(1e1, 1e6),
                    'kobs':(1e-5, 1e-2)}

def find_dG_from_Kd(Kd):
    parameters = Parameters()
    return parameters.RT*np.log(Kd*parameters.concentration_units)

def find_Kd_from_dG(dG):
    parameters = Parameters()
    return np.exp(dG/parameters.RT)
        
def perVariantInfo(table, variants=None):
    if variants is None:
        variants = np.arange(0, np.max(table['variant_number']))
    columns = [name for name in table][:12] + ['numTests', 'numRejects', 'dG', 'fmax', 'fmin', 'qvalue', 'dG_lb', 'dG_ub']
    newtable = pd.DataFrame(columns=columns, index=np.arange(len(variants)))
    for i, variant in enumerate(variants):
        if i%1000==0: print 'computing iteration %d'%i
        sub_table = table[table['variant_number']==variant]
        if len(sub_table) > 0:
            sub_table_filtered = filterFitParameters(sub_table)[['dG', 'fmax', 'fmin', 'qvalue']]
            sub_table_filtered = sub_table.copy()[['dG', 'fmax', 'fmin', 'qvalue']]
            newtable.iloc[i]['numTests'] = len(sub_table_filtered)
            newtable.iloc[i]['numRejects'] = len(sub_table) - len(sub_table_filtered)
            newtable.iloc[i]['dG':'qvalue'] = np.median(sub_table_filtered, 0)
            newtable.iloc[i][:'total_length'] = sub_table.iloc[0][:'total_length']
            try:
                newtable.iloc[i]['dG_lb'], newtable.iloc[i]['dG_ub'] = bootstrap.ci(sub_table_filtered['dG'], np.median)
            except IndexError: print variant
    return newtable
        

def filterFitParameters(sub_table):
    binding_curves = np.array([np.array(sub_table[i]) for i in range(8)])
    num_curves = binding_curves.shape[1]
    indices = np.arange(num_curves)[np.all((np.array(sub_table['rsq'] > 0), np.sum(np.isnan(binding_curves), 0) < 2), axis=0)]
    #indices = np.arange(num_curves)[np.array(sub_table['rsq'] > 0.5)]
    sub_table = sub_table.iloc[indices]
    return sub_table

def bindingCurve(concentrations, dG, fmax=None, fmin=None):
    if fmax is None:
        fmax = 1
    if fmin is None:
        fmin = 0
    parameters = Parameters()
    return fmax*concentrations/(concentrations + np.exp(dG/parameters.RT)/1E-9) + fmin

def offRateCurve(time, toff, fmax=None, fmin=None):
    if fmax is None:
        fmax = 1
    if fmin is None:
        fmin = 0
    return fmax*np.exp(-time/toff) + fmin


def onRateCurve(time, ton, fmax=None, fmin=None):
    if fmax is None:
        fmax = 1
    if fmin is None:
        fmin = 0
    return fmax+fmin - fmin*np.exp(-time/ton)
    
    

def plotBoxplot(data, labels):
    ax = plt.gca()
    bp = ax.boxplot(data)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('normalized insertions/peak')
    for box in bp['boxes']:
        # change outline color
        box.set( color='k', linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(color='k', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='.', color='r' ,linewidth=0, alpha=0.5)
    return

def plotCluster(series, concentrations, name=None):
    if name is None:
        name = '%4.0f'%(series['variant_number'])
    parameters = Parameters()
    series['kd'] = np.exp(series['dG']/parameters.RT)/1E-9
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.plot(concentrations, [series[i] for i in range(8)], 'o')
    ax.plot(np.logspace(-1, 4, 50), bindingCurve(np.logspace(-1, 4, 50), series['kd'], series['fmax'], series['fmin']),'k')
    ax.set_xscale('log')
    ax.set_xlabel('concentration (nM)')
    ax.set_ylabel('normalized fluorescence')
    plt.title(name)
    plt.tight_layout()
    return

def plotVariant(sub_table, concentrations, name=None, to_filter=None):
    if name is None:
        name = ''
    if to_filter is None:
        to_filter=True
    # reduce to fits that were successful
    if to_filter:
        sub_table = filterFitParameters(sub_table)
    print 'testing %d variants'%len(sub_table)
        
    # plot  
    num_concentrations = len(concentrations)
    binding_curves = np.array([np.array(sub_table[i]) for i in range(num_concentrations)])
    binding_curves_norm = (binding_curves-np.array(sub_table['fmin']))/np.array(sub_table['fmax'])
    frac_bound = np.median(binding_curves_norm, axis=1)
    [percentile25, percentile75] = np.percentile(binding_curves_norm, [25,75],axis=1)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.errorbar(concentrations, frac_bound, yerr=[frac_bound-percentile25, percentile75-frac_bound], fmt='o')
    ax.plot(np.logspace(-1, 4, 50), bindingCurve(np.logspace(-1, 4, 50), np.median(sub_table['dG'])),'k')
    ax.set_xscale('log')
    ax.set_xlabel('concentration (nM)')
    ax.set_ylabel('normalized fluorescence')
    ax.set_ylim((0, 1.5))
    plt.title(name)
    plt.tight_layout()
    
    # make histogram
    binsize = 0.2
    parameters = Parameters()
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    xbins = np.arange(parameters.min_deltaG, parameters.max_deltaG+2*binsize, binsize)-binsize*0.5
    histogram.compare([sub_table['dG']], bar=True, xbins=xbins, normalize=False )
    ax.set_xlim((parameters.min_deltaG-1, parameters.max_deltaG+1))
    ax.set_xlabel('dG (kcal/mol)')
    ax.legend_ = None
    ax.set_ylabel('number')
    plt.title(name)
    plt.tight_layout()
    return

def plotOffrateCurve(series, times):
    numTimePoints = len(times)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.scatter(times, series.loc[[i for i in range(numTimePoints)]], facecolors='none', edgecolors='k')
    ax.plot(times, offRateCurve(times, series['toff'], fmax=series['fmax'], fmin=series['fmin']), 'r', label='FDR = %4.2f'%series['qvalue'])
    ax.set_xlim((times.min(), times.max()))
    ax.set_ylim((0, np.max([600, series['fmax']])))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('f green')
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    return

def plotOffRateVariantError(subtable, times, errorbar=None):
    if errorbar is None: errorbar=True
    subtable.dropna(subset=['toff'], how='all', axis=0, inplace=True)
    subtable = subtable.loc[subtable['qvalue'].values < 0.05]
    cols = ['toff', 'fmin', 'fmax', 'fmin_var', 'fmax_var']
    subtable.loc[:, cols] = subtable.loc[:, cols].astype(float)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    for idx in subtable.index:
        series = subtable.loc[idx]
        time = times.loc[series['tile']]
        numTimePoints = len(time)
        fracbound = (series.loc[[i for i in range(numTimePoints)]] - series['fmin'])/series['fmax']
        if errorbar:
            yerr = np.sqrt((np.power((series['fmax_var'])/series['fmax']*fracbound, 2) + np.power(series['fmin']/series['fmax']*(series['fmax_var']/series['fmax']+series['fmin_var']/series['fmin']), 2)).values.astype(float))
            ax.errorbar(time, fracbound,yerr, fmt='-.', color='k', ecolor='k', alpha = 0.1, capsize=0)
    
    ax.plot(times.loc[9], offRateCurve(times.loc[9], subtable.median(axis=0)['toff']), 'r')
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlim((0, np.max(times.loc[9])))
    ax.set_xlabel('time (s)'); ax.set_ylabel('fraction bound')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    return ax

def offRateFracBoundError(series, numTimePoints):
    cols = ['fmin', 'fmax', 'fmin_var', 'fmax_var']
    series[cols] = series[cols].astype(float)
    fracbound = (series.loc[[str(i) for i in range(numTimePoints)]] - series['fmin'])/series['fmax']
    if series['fmin'] != 0 and series['fmax'] != 0:
        yerr = np.sqrt((np.power((series['fmax_var'])/series['fmax']*fracbound, 2) +
                        np.power(series['fmin']/series['fmax']*(series['fmax_var']/series['fmax']+series['fmin_var']/series['fmin']), 2)).values.astype(float))
    elif series['fmax'] != 0:
        yerr = np.sqrt((np.power((series['fmax_var'])/series['fmax']*fracbound, 2)).values.astype(float))
    else: yerr = np.nan
    return fracbound, yerr
    
def plotOffRateVariant(subtable, times, errorbar=None, plotAllTraces=None):
    if errorbar is None: errorbar=True
    if plotAllTraces is None: plotAllTraces=True
    subtable.dropna(subset=['toff'], how='all', axis=0, inplace=True)
    #subtable = IMlibs.filterFitParameters(subtable)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    fracbound = np.zeros((subtable.shape[0], times.shape[1]))
    yerr = np.zeros((subtable.shape[0], times.shape[1]))
    for i, idx in enumerate(subtable.index):
        series = subtable.loc[idx]
        time = times.loc[series['tile']]
        numTimePoints = len(time)
        fracbound[i], yerr[i] = offRateFracBoundError(series, numTimePoints)
        if plotAllTraces:
            if errorbar:
                ax.errorbar(time, fracbound[i],yerr[i], fmt='-', marker='.', color='k', ecolor='k', alpha = 0.1, capsize=0)
            else:
                ax.plot(time, fracbound[i], '.-', color='k', alpha=0.1)
    if not plotAllTraces:
        dxs = [DescrStatsW(fracbound[:, i], weights = 1/yerr[:,i]) for i in range(times.shape[1])]
        
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlim((0, np.max(times.loc[9])))
    ax.set_xlabel('time (s)'); ax.set_ylabel('fraction bound')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    maxTime = np.max(times.loc[9])
    timeBins = np.linspace(0, maxTime, len(times.loc[9]))
    timeBinCenters = (timeBins[:-1] + timeBins[1:])*0.5
    fracboundAll = pd.DataFrame(index=timeBinCenters, columns=['fracbound', 'delta'])
    whatBin = np.array([np.digitize(times.loc[subtable.loc[idx, 'tile']], right=True, bins=timeBins) for idx in subtable.index])
    for i, timeBinCenter in enumerate(timeBinCenters):
        fracboundAll.loc[timeBinCenter, 'fracbound'] = np.median(fracbound[whatBin == (i+1)])
        fracboundAll.loc[timeBinCenter, 'delta'] = np.sqrt(np.sum(np.power(yerr[whatBin == (i+1)], 2)))
    
    #ax.errorbar(fracboundAll.index,fracboundAll.loc[:, 'fracbound'], yerr= fracboundAll.loc[:, 'delta'], fmt='o', color='r', ecolor='k', capsize=0.5, alpha=0.5)
    ax.plot(fracboundAll.index,fracboundAll.loc[:, 'fracbound'], 'ro', alpha = 0.5)
    ax.plot(times.loc[9], offRateCurve(times.loc[9], subtable.median(axis=0)['toff']), 'r')
    ax.tick_params( direction='out', top='off', right='off')
    return ax, timeBinCenters

def plotBindingCurveVariant(subtable, concentrations,errorbar=None, plotAllTraces=None):
    if errorbar is None: errorbar=True
    if plotAllTraces is None: plotAllTraces=True
    parameter = 'dG'
    subtable.dropna(subset=[parameter], how='all', axis=0, inplace=True)
    #subtable = IMlibs.filterFitParameters(subtable)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    fracbound = np.zeros((subtable.shape[0], len(concentrations)))
    yerr = np.zeros((fracbound.shape))
    for i, idx in enumerate(subtable.index):
        series = subtable.loc[idx]
        numPoints = len(concentrations)
        fracbound[i], yerr[i] = offRateFracBoundError(series, numPoints)
    if plotAllTraces:
        for i in range(len(subtable)):
            if errorbar:
                ax.errorbar(concentrations, fracbound[i],yerr[i], fmt='-', marker='.', color='k', ecolor='k', alpha = 0.1, capsize=0)
            else:
                ax.plot(concentrations, fracbound[i], '.-k', alpha=0.1)
        ax.plot(concentrations, np.median(fracbound, axis=0), 'ro', alpha = 0.5)
    else:
        #weights = subtable.loc[:, 'fraction_consensus'].values/100*subtable.loc[:, 'clusters_per_barcode'].values
        #dx = DescrStatsW(fracbound, weights = weights/np.mean(weights))
        dx = DescrStatsW(fracbound) # not using any weights
        ax.errorbar(concentrations, np.median(fracbound, axis=0),
                    yerr=dx.mean - dx.zconfint_mean()[0], fmt='o',color='r', ecolor='k', alpha=0.5)
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlim((5e-1, 5e3))
    ax.set_xlabel('concentration (nM)'); ax.set_ylabel('fraction bound')
    ax.set_xscale('log')
    #ax.plot(concentrations, np.median(fracbound, axis=0), 'ro', alpha = 0.5)
    concentrationsAll = np.logspace(-1, 4, 50)
    ax.plot(concentrationsAll, bindingCurve(concentrationsAll, subtable.median(axis=0)[parameter]), 'r')
    ax.tick_params( direction='out', top='off', right='off')
    return ax, concentrationsAll

def plotOnrateCurve(series, times):
    numTimePoints = len(times)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.scatter(times, series.loc[[i for i in range(numTimePoints)]], facecolors='none', edgecolors='k')
    ax.plot(times, onRateCurve(times, series['ton'], fmax=series['fmax'], fmin=series['fmin']), 'r', label='FDR = %4.2f'%series['qvalue'])
    ax.set_xlim((times.min(), times.max()))
    ax.set_ylim((0, np.max([600, series['fmax']])))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('f green')
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    return

def plotHistogram(sub_table, parameter):
    plt.figure(figsize=(4,4))
    if parameter == 'ton':
        vmin = 0; vmax = 7
        histogram.compare([np.log10(sub_table.loc[:, 'ton'])], xbins=np.linspace(vmin, vmax, 50), bar=True,normalize=False,  cmap='autumn')
        plt.xticks(np.arange(vmin, vmax+1), ['%1.0e'%i for i in np.power(10, np.arange(vmin, vmax+1))], rotation=90 )
        plt.xlabel('lifetime of observed association rate (s)')
        plt.ylabel('fraction of total')
    if parameter == 'dG':
        vmin = -14; vmax = -4
        histogram.compare([sub_table.loc[:, 'dG']], xbins=np.linspace(vmin, vmax, 50), bar=True, normalize=False, cmap='autumn')
        plt.xlabel('delta G (kcal/mol)')
        plt.ylabel('number')
    if parameter == 'toff':
        vmin = 0; vmax = 7
        histogram.compare([np.log10(sub_table.loc[:, parameter])], xbins=np.linspace(vmin, vmax, 50), bar=True, normalize=False, cmap='autumn')
        plt.xticks(np.arange(vmin, vmax+1), ['%1.0e'%i for i in np.power(10, np.arange(vmin, vmax+1))], rotation=90 )
        plt.xlabel('lifetime of observed dissociation rate (s)')
        plt.ylabel('number of clusters')

    ax = plt.gca(); ax.legend_ = None
    plt.tight_layout()
    ax.tick_params( direction='out', top='off', right='off')        
    ax.get_ylim
    return

def getInfo(per_variant_series):
    return '_'.join(np.array(per_variant_series.loc[['variant_number','total_length', 'helix_one_length','junction_length','junction_sequence',  'helix_context', 'loop','receptor']], dtype=str)).replace('.0','')

def findVariantNumbers(table, criteria_dict):
    for name, value in criteria_dict.items():
        if name == 'offset':
            print 'selecting criteria: offset <= %d'%value
            table = table[np.abs(table.helix_two_length - table.helix_one_length) <= value]
        else:
            print 'selecting criteria: %s = %s'%(name, value)
            table = table[table[name]==value]

    return np.unique(table['variant_number']).astype(int)

def plotVariantBoxplots(table, variant_table, helix_context, total_length, loop=None, receptor=None, max_diff_helix_length=None, helix_one_length=None):
    if loop is None:
        loop = 'goodLoop'
    if receptor is None:
        receptor='R1'
    if max_diff_helix_length is None:
        max_diff_helix_length = 1

    criteria_central = np.all((np.array(variant_table['total_length']==total_length),
                               np.array(np.abs(variant_table['helix_one_length'] - variant_table['helix_two_length']) <= max_diff_helix_length),
                               np.array(variant_table['receptor'] == receptor),
                               np.array(variant_table['loop']==loop),
                               np.array(variant_table['helix_context'] == helix_context)),
                            axis=0)
    sub_table = variant_table[criteria_central]
    junction_topologies = np.array(['','B1', 'B2', 'B1_B1', 'B2_B2', 'B1_B1_B1', 'B2_B2_B2', 'M','M_B1','B2_M', 'M_M',
                                    'B2_B2_M', 'M_B1_B1', 'B2_M_M', 'M_M_B1', 'M_M_M', 'D_D'])
    delta_G_initial = variant_table.loc[0]['dG']
    delta_deltaG = ['']*len(junction_topologies)
    num_variants = np.zeros(len(junction_topologies))
    for i, topology in enumerate(junction_topologies):
        if i==0 or topology=='':
            variant_numbers = np.unique(variant_table[np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)]['variant_number'])
        else:
            variant_numbers = np.unique(variant_table[np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)]['variant_number'])
        deltaGs = np.empty(0)
        for j, variant in enumerate(variant_numbers):
            deltaGs = np.append(deltaGs, variant_table.iloc[variant]['dG'])
        delta_deltaG[i] = deltaGs - delta_G_initial
        num_variants[i] = len(variant_numbers)
        
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    plotBoxplot(delta_deltaG, junction_topologies)
    ax.set_ylabel('delta delta G')
    ax.set_ylim((-1, 5))
    plt.subplots_adjust(bottom=0.2)
    
    # plot bar plot
    
    return num_variants

def plotMarkers():
    fmt_list = ['o', '^', 'v', '<', '>', 's', 's', '*', '*', '.', '.', 'x', 'x', '+', '+', '1', '1', '2', '2', '3', '3', '4', '4']
    lengths  = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10]
    
    fig = plt.figure(figsize=(2,6))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    x = np.ones(len(lengths))
    y = lengths
    for i in range(len(lengths)):
        ax.plot(x[i], y[i], fmt_list[i], color='0.5')
    ax.set_xlim(0.75, 1.25)
    ax.set_ylim((-11, 11))
    ax.set_yticks(lengths)
    ax.set_xticks([])
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out')
    ax.set_ylabel('junction offset')
    #plt.subplots_adjust(left = 0.3, right=0.8)
    plt.tight_layout()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    #ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    
    return

def plotColors():
    norm  = colors.LogNorm(vmin=0.001, vmax=1)
    cmap = cmx.coolwarm_r
    
    fig = plt.figure(figsize=(1.5,6))
    ax1 = fig.add_subplot(111)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                   norm=norm,
                                   orientation='vertical',
                                   )
    cb1.set_label('FDR')
    plt.tight_layout()
    
    return

def makeFasta(per_variant, filename=None, image_dir=None):
    if filename is None:
        filename = 'test.fasta'
    if image_dir is None:
        image_dir = os.getcwd()
    filename = os.path.join(os.getcwd(), filename)
    f = open(filename, 'w')
    for i in range(len(per_variant)):
        header = '>'+ '_'.join(np.array(per_variant.iloc[i].loc[['variant_number','total_length', 'helix_one_length','junction_length','junction_sequence',  'helix_context', 'loop','receptor']], dtype=str)).replace('.0','')
        f.write(header + '\n')
        f.write(per_variant.iloc[i].loc['sequence'] + '\n')
    f.close()
    makeSecondaryStructurePics(filename, image_dir)
    return
    
def makeSecondaryStructurePics(filename, image_dir=None):
    if image_dir is None:
        image_dir = os.getcwd()
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    curr_dir = os.getcwd()
    os.chdir(image_dir)
    os.system("cat %s | RNAfold"%filename)
    os.chdir(curr_dir)
    return

def getMarker(series):
    fmt_list = ['o', '^', 'v', '<', '>', 's', 's', '*', '*', '.', '.', 'x', 'x', '+', '+', '1', '1', '2', '2', '3', '3', '4', '4']
    lengths  = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10]
    fmt_dict = {}
    for i, length in enumerate(lengths): fmt_dict[length] = fmt_list[i]
    fmt = fmt_dict[series['helix_two_length'] - series['helix_one_length']]
    
    unit = 0.05
    wiggle = unit*(series['helix_two_length'] - series['helix_one_length'])
    
    cNorm  = colors.Normalize(vmin=0, vmax=3)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='coolwarm')
    color = scalarMap.to_rgba(-np.log10(series['qvalue']))
    
    return fmt, wiggle, color

def plot_dG_errorbars_vs_coordinate(series, xvalue, param=None):
    if param is None: param = 'dG'
    ax = plt.gca()
    fmt, wiggle, color = getMarker(series)
    if np.isnan(series[param+'_lb']) or np.isnan(series[param+'_ub']):
        ax.plot(xvalue+wiggle, series[param], fmt, color=color)
    else:
        ax.errorbar(xvalue+wiggle, series[param], yerr=[[series[param] - series[param+'_lb']], [series[param+'_ub'] - series[param]]],
                    fmt=fmt, color=color, ecolor='k')
    return

def plot_errorbars_2D(series, param1, param2):
    ax = plt.gca()
    fmt, wiggle, color = getMarker(series)
    xerr = [[np.nan], [np.nan]]
    yerr = [[np.nan], [np.nan]]

    xerr = [[series[param1] - series[param1+'_lb']], [series[param1+'_ub'] - series[param1]]]
    yerr = [[series[param2] - series[param2+'_lb']], [series[param2+'_ub'] - series[param2]]]
    ax.errorbar(series[param1], series[param2], yerr=yerr, xerr=xerr,
                fmt=fmt, color=color, ecolor='k')
    return

def plot_expected_spread(variant_table, subset_index=None, xbins=None):
    if subset_index is None:
        subset_index = variant_table.index
    if xbins is None: xbins = np.linspace(-1, 10, 100)
    # plot average error
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    histogram.compare([variant_table.loc[subset_index, 'dG_ub'].values -
                       variant_table.loc[subset_index, 'dG_lb'].values],
        xbins = xbins, bar=True, normalize=False)
    ax.set_xlabel('Confidence interval width (kcal/mol)')
    ax.set_ylabel('Number of variants')
    ax.set_xlim((0, 7))
    ax.tick_params( direction='out', top='off', right='off')
    ax.legend_ = None
    plt.tight_layout()

    return

def plot_parameter_vs_length(series, p1, p2):
    ax = plt.gca()
    fmt, wiggle, color = getMarker(series)
    ax.plot(series[p1]+wiggle, series[p2], fmt, color=color)
    return ax

def plot_scatterplot(per_variant, param1, param2):
    parameters = Parameters()
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    for indx in per_variant.index:
        plot_errorbars_2D(per_variant.loc[indx], param1, param2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(parameters.lims[param1])
    ax.set_ylim(parameters.lims[param2])
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.tight_layout()
    ax.grid(linestyle=':', alpha=0.5)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out')
    return ax

def plot_scatterplot_errorbars(table1, table2=None, yvalues=None, yerrs=None, parameter=None, errorBar=None, labels=None ):
    if parameter is None: parameter = 'dG'
    if errorBar is None:
        if parameter == 'dG': errorBar = True
        else: errorBar = False
    if table2 is None and yvalues is None:
        print 'Error: Need to define either a table or vactor of y values'
        return
    # intialize figure
    plt.figure()
    ax = plt.gca()
    #ax.plot([-12, -6], [-12, -6], '--', c='0.25')
    #ax.set_xlim((-12,-6))
    #ax.set_ylim((-12,-6))
    ax.grid()
    
    for i in range(len(table1)):
        xvalue = table1.iloc[i].loc[parameter]
        if table2 is not None:
            yvalue = table2.iloc[i].loc[parameter]
        else: yvalue = yvalues[i]
        xerr = [[0], [0]]
        if yerrs is None:
            yerr = [[0], [0]]
        else:
            yerr = [[yerrs[i][0]], [yerrs[i][-1]]]
        if np.isnan(xvalue) or np.isnan(yvalue):
            print 'Skipping variant %s because no data associated with it'%(str(table1.iloc[i]['variant_number']))
        else:
            fmt, wiggle, color = getMarker(table1.iloc[i])
            if errorBar:
                if not np.isnan(table1.iloc[i]['dG_lb']): xerr[0][0] =   xvalue - table1.iloc[i]['dG_lb']
                if not np.isnan(table1.iloc[i]['dG_ub']): xerr[1][0] = -(xvalue - table1.iloc[i]['dG_ub'])
                if table2 is not None:
                    if not np.isnan(table2.iloc[i]['dG_lb']): yerr[0][0] =   yvalue - table2.iloc[i]['dG_lb']
                    if not np.isnan(table2.iloc[i]['dG_ub']): yerr[1][0] = -(yvalue - table2.iloc[i]['dG_ub'])
                ax.errorbar(xvalue, yvalue, fmt=fmt, color=color, ecolor='k', yerr=yerr, xerr=xerr)
            else:
                ax.plot(xvalue, yvalue, fmt, color=color)
        if labels is not None:
            label = labels[i]
            plt.annotate(label, xy=(xvalue, yvalue), xytext = (30, -20),
                         fontsize=8,
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    return

def plotScatterplot(variant_table, param1, param2, indx=None, color_param=None, vmax=None, vmin=None):
    if vmax is None: vmax = 1
    if vmin is None: vmin = 0
    if indx is None:
        indx = np.arange(len(variant_table))
    if color_param is None:
        #c = np.tile([1,0,0], reps =(len(variant_table),1))
        color_param = ['', '']
    else:
        c = variant_table[color_param[0]].loc[indx, color_param[1]]
    if color_param == 'qvalue':
        norm  = colors.LogNorm(vmin=1e-3, vmax=1e0)
        cmap = cmx.coolwarm_r
    else:
        norm  = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmx.coolwarm_r
        c = np.ones(len(variant_table))*0.5

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    norm  = colors.LogNorm(vmin=1e-3, vmax=1e0)
    cmap = cmx.coolwarm_r
    im = ax.scatter(variant_table[param1[0]].loc[indx, param1[1]].values,
                    variant_table[param2[0]].loc[indx, param2[1]].values,
                    alpha=0.1, edgecolor='0.1', c=c, cmap=cmap, norm=norm)

    ax.set_xlabel(' '.join(param1)); ax.set_ylabel(' '.join(param2))
    cbar = plt.colorbar(im)
    cbar.set_label(' '.join(color_param))
    plt.tight_layout()
    return ax
    




def plot_over_coordinate(per_variant, to_fill=None, x_param=None, x_param_name=None, sort_index=None):
    # plot variant delta G's over a coordinate given, or total length (default)
    if sort_index is None:
        per_variant.sort(['total_length', 'helix_two_length', 'dG'], inplace=True)
    else: per_variant = per_variant.iloc[sort_index]
    if to_fill is None: to_fill = False    # by default, don't plot 'landscape ' background
    if x_param is None: x_param = per_variant['total_length'].astype(int)
    if x_param_name is None: x_param_name = 'total_length'
    x_param_unique = np.unique(x_param)
    x_param = np.array(x_param) # force to be array for proper indexing later
    
    # initiate figure
    fig_width = 4 + 3./96*len(x_param_unique) - 4*3./96
    fig = plt.figure(figsize=(fig_width, 4))
    ax = fig.add_subplot(111)
    
    # plot gray boundaries
    if to_fill:
        ax.fill_between(x_param_unique, [np.mean(per_variant[per_variant['total_length']==length]['dG_lb']) for length in x_param_unique],
                        [np.mean(per_variant[per_variant['total_length']==length]['dG_ub']) for length in x_param_unique],
                        facecolor='0.5', alpha=0.05, linewidth=0)

    # plot points with errorbars
    for i in range(len(per_variant)):
        series = per_variant.iloc[i]
        plot_dG_errorbars_vs_coordinate(series, x_param[i])
    
    # plot ticks and labels
    if len(x_param_unique) < 10:
        ax.set_xticks(x_param_unique)
    ax.legend_ = None
    ax.set_xlabel(x_param_name)
    ax.set_ylabel('dG (kcal/mol)')
    ax.set_ylim((-12, -6))
    ax.set_xlim((x_param_unique[0]-1, x_param_unique[-1]+1))
    #plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.tight_layout()
    ax.grid(linestyle=':', alpha=0.5)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out')
    
    # plot bar graph
    xvalues = np.arange(len(per_variant))
    fig = plt.figure(figsize=(fig_width,3))
    ax = fig.add_subplot(111)
    ax.bar(xvalues, per_variant['numTests'], color='0.75', linewidth=0)
    if len(x_param_unique) < 10:
        ax.set_xticks(xvalues+0.4)
        ax.set_xticklabels(np.array(x_param, dtype=int), rotation=90)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out')
    ax.set_ylabel('number of tests')
    ax.set_xlabel(x_param_name)
    ax.set_xlim((xvalues[0]-0.4, xvalues[-1]+1))
    ax.set_ylim((0, 35))
    for i in xvalues:
        series = per_variant.iloc[i]
        fmt, wiggle, color = getMarker(series)
        ax.plot(i+0.4, series['numTests']+1, fmt, color=color)
    #plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.tight_layout()
    
    return
    
#start here
def plot_length_changes(table, variant_table, helix_context, topology, loop=None, receptor=None, offset=None):
    if loop is None:
        loop = 'goodLoop'
    if receptor is None:
        receptor='R1'
    if offset is None:
        offset = 0  # amount to change helix_one_length by from default
    couldPlot = True
    criteria_central = np.all((np.array(variant_table['receptor'] == receptor),
                           np.array(variant_table['loop']==loop),
                           np.array(variant_table['helix_context'] == helix_context)),
                          axis=0)
    if topology == '':
        criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)
    else:
        criteria_central = np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)
    
    if sum(criteria_central)>0:
        sub_table = variant_table[criteria_central]
        junction_sequences = np.unique(sub_table['junction_sequence'])
        total_lengths = np.array([8,9,10,11,12])
        delta_G_initial = variant_table.loc[0]['dG']
        delta_G_ub_initial = variant_table.loc[0]['dG_ub']-variant_table.loc[0]['dG']
        delta_G_lb_initial = variant_table.loc[0]['dG_lb']-variant_table.loc[0]['dG']   
        delta_deltaG = np.ones((len(junction_sequences), len(total_lengths)))*np.nan
        delta_deltaGerrub = np.ones((len(junction_sequences), len(total_lengths)))*np.nan
        delta_deltaGerrlb = np.ones((len(junction_sequences), len(total_lengths)))*np.nan 
        for i, sequence in enumerate(junction_sequences):
            for j, length in enumerate(total_lengths):
                helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
                variant_number = sub_table[np.all((np.array(sub_table['junction_sequence']==sequence),
                                                   np.array(sub_table['total_length']==length),
                                                   np.array(sub_table['helix_one_length']==helix_one_length)),axis=0)]['variant_number']
                if len(variant_number)>0:
                    delta_deltaG[i][j],delta_deltaGerrub[i][j],delta_deltaGerrlb[i][j] = get_ddG_and_Errs(sub_table, variant_number, delta_G_initial,delta_G_ub_initial, delta_G_lb_initial)

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        plotfun.plot_manylines(delta_deltaG, cmap='Paired', marker='o', labels=junction_sequences, alpha=0.5)
        ax.set_xticks(total_lengths-8)
        values = range(len(delta_deltaG))
        cm = plt.cm.Paired
        cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
        #put error bars on numbers
        for i, row in enumerate(delta_deltaG):
            ax.errorbar(total_lengths-8,row, yerr=[delta_deltaGerrub[i,:], delta_deltaGerrlb[i,:]], fmt='.', color=scalarMap.to_rgba(i),ecolor=scalarMap.to_rgba(i)) 
        ax.set_xticklabels(total_lengths.astype(str))
        ax.legend_ = None
        ax.set_xlabel('total length')
        ax.set_ylabel('delta delta G')
        ax.set_ylim((-1, 5))
        plt.tight_layout()
    else: couldPlot = False
    return couldPlot,  delta_deltaG


def plot_position_changes(table, variant_table, helix_context, topology, total_length, loop=None, receptor=None):
    if loop is None:
        loop = 'goodLoop'
    if receptor is None:
        receptor='R1'
    
    criteria_central = np.all((np.array(variant_table['receptor'] == receptor),
                           np.array(variant_table['loop']==loop),
                           np.array(variant_table['helix_context'] == helix_context),
                           np.array(variant_table['total_length'] == total_length)),
                          axis=0)
    if topology == '':
        criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)
    else:
        criteria_central = np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)
    
    sub_table = variant_table[criteria_central]
    junction_sequences = np.unique(sub_table['junction_sequence'])
    helix_one_lengths = np.arange(0, total_length)
    delta_G_initial = variant_table.loc[0]['dG']
    delta_deltaG = np.ones((len(junction_sequences), len(helix_one_lengths)))*np.nan
    for i, sequence in enumerate(junction_sequences):
        for j, length in enumerate(helix_one_lengths):
            variant_number = sub_table[np.all((np.array(sub_table['junction_sequence']==sequence),
                                               np.array(sub_table['helix_one_length']==length)),axis=0)]['variant_number']
            if len(variant_number)>0:
                delta_deltaG[i][j] = sub_table.loc[variant_number]['dG'] - delta_G_initial
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    plotfun.plot_manylines(delta_deltaG, cmap='Paired', marker='o', labels=junction_sequences, alpha=0.5)
    ax.set_xticks(helix_one_lengths)
    ax.set_xticklabels(helix_one_lengths.astype(str))
    ax.legend_ = None
    ax.set_xlabel('helix1 length')
    ax.set_ylabel('delta delta G')
    ax.set_ylim((-1, 5))
    plt.tight_layout()
    return

def plot_length_changes_helices(table, variant_table, topology, loop=None, receptor=None, offset=None):
    if loop is None:
        loop = 'goodLoop'
    if receptor is None:
        receptor='R1'
    if offset is None:
        offset = 0  # amount to change helix_one_length by from default
    couldPlot = True
    criteria_central = np.all((np.array(variant_table['receptor'] == receptor),
                           np.array(variant_table['loop']==loop)),
                          axis=0)
    if topology == '':
        criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)
    else:
        criteria_central = np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)
    # choose just one sequence of that topology to look at
    seq1 = variant_table[criteria_central]['junction_sequence'].iloc[0]
    criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence']) == seq1), axis=0)
    
    sub_table = variant_table[criteria_central]
    helix_context = np.unique(sub_table['helix_context'])
    total_lengths = np.array([8,9,10,11,12])
    delta_G_initial = variant_table.loc[0]['dG']
    delta_G_ub_initial = variant_table.loc[0]['dG_ub']-variant_table.loc[0]['dG']
    delta_G_lb_initial = variant_table.loc[0]['dG_lb']-variant_table.loc[0]['dG']   
    delta_deltaG = np.ones((len(helix_context), len(total_lengths)))*np.nan
    delta_deltaGerrub = np.ones((len(helix_context), len(total_lengths)))*np.nan
    delta_deltaGerrlb = np.ones((len(helix_context), len(total_lengths)))*np.nan
    for i, sequence in enumerate(helix_context):
        for j, length in enumerate(total_lengths):
            helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
            #variant_number = sub_table[np.all((np.array(sub_table['helix_context']==sequence),
            #                                   np.array(sub_table['total_length']==length),
            #                                   np.array(sub_table['helix_one_length']==helix_one_length)),axis=0)]['variant_number']
            variant_number = sub_table[np.all((np.array(sub_table['helix_context']==sequence),
                                               np.array(sub_table['total_length']==length),
                                               np.array(sub_table['helix_one_length']==helix_one_length)),axis=0)].index
            #calculate ddG, and errorbars ddG (sqrt of sum or squares)
            if len(variant_number)>0:
                delta_deltaG[i][j] = sub_table.loc[variant_number]['dG'] - delta_G_initial
                delta_deltaGerrub[i][j] = np.sqrt(pow(sub_table.loc[variant_number]['dG_ub']-sub_table.loc[variant_number]['dG'],2) + pow(delta_G_ub_initial,2))
                delta_deltaGerrlb[i][j] = np.sqrt(pow(sub_table.loc[variant_number]['dG_lb']-sub_table.loc[variant_number]['dG'],2) + pow(delta_G_lb_initial,2))
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    plotfun.plot_manylines(delta_deltaG, cmap='Paired', marker='o', labels=helix_context, alpha=0.5)
    values = range(len(delta_deltaG))
    cm = plt.cm.Paired
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    #put error bars on numbers
    for i, row in enumerate(delta_deltaG):
        ax.errorbar(total_lengths-8,row, yerr=[delta_deltaGerrub[i,:], delta_deltaGerrlb[i,:]], fmt='.', color=scalarMap.to_rgba(i),ecolor=scalarMap.to_rgba(i)) 
    ax.set_xticks(total_lengths-8)
    ax.set_xticklabels(total_lengths.astype(str))
    ax.legend_ = None
    ax.set_xlabel('total length')
    ax.set_ylabel('delta delta G')
    ax.set_ylim((-1, 5))
    ax.set_xlim((-0.5,4.5))
    plt.tight_layout()
    return
    
def convert_nomen(topologies):
    topos = {'':'0x0','B1':'1x0', 'B2':'0x1', 'B1_B1':'2x0', 'B2_B2':'0x2', 'B1_B1_B1':'3x0', 'B2_B2_B2':'0x3', 'M':'1x1','M_B1':'1x2', 'B2_M':'2x1', 'M_M':'2x2',
                                    'B2_B2_M':'2x3', 'M_B1_B1':'3x2', 'B2_M_M':'1x3', 'M_M_B1':'3x1', 'M_M_M':'3x3'}
    return [topos[t] for t in topologies]
def plot_changes_helices_allseqs(table, variant_table, topology, loop=None, receptor=None, offset=None):
    if loop is None:
        loop = 'goodLoop'
    if receptor is None:
        receptor='R1'
    if offset is None:
        offset = 0  # amount to change helix_one_length by from default
    couldPlot = True
    criteria_central = np.all((np.array(variant_table['receptor'] == receptor),
                           np.array(variant_table['loop']==loop)),
                          axis=0)
    if topology == '':
        criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)
    else:
        criteria_central = np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)
      
    
    sub_table = variant_table[criteria_central]
    helix_context = np.unique(sub_table['helix_context'])
    junctionseqs = np.unique(sub_table['junction_sequence'])
    
    #choose just one length = 10 
    length = 10
    delta_G_initial = variant_table.loc[0]['dG']
    delta_G_ub_initial = variant_table.loc[0]['dG_ub']-variant_table.loc[0]['dG']
    delta_G_lb_initial = variant_table.loc[0]['dG_lb']-variant_table.loc[0]['dG']
    
    delta_deltaG = np.ones((len(helix_context), len(junctionseqs)))*np.nan
    delta_deltaGerrub = np.ones((len(helix_context), len(junctionseqs)))*np.nan
    delta_deltaGerrlb = np.ones((len(helix_context), len(junctionseqs)))*np.nan
    for i, sequence in enumerate(helix_context):
        for j, jsequence in enumerate(junctionseqs):
            helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
            variant_number = sub_table[np.all((np.array(sub_table['helix_context']==sequence),
                                                np.array(sub_table['total_length']==length),
                                                np.array(sub_table['helix_one_length']==helix_one_length),
                                                np.array(sub_table['junction_sequence']==jsequence)),axis=0)].index
            #calculate ddG, and errorbars ddG (sqrt of sum or squares)
            if len(variant_number)>0:
                delta_deltaG[i][j],delta_deltaGerrub[i][j],delta_deltaGerrlb[i][j] = get_ddG_and_Errs(sub_table, variant_number, delta_G_initial,delta_G_ub_initial, delta_G_lb_initial)
    if len(junctionseqs)>4:   
        fig = plt.figure(figsize=(12,6))
    else:
        fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plotfun.plot_manylines(delta_deltaG, cmap='Paired', marker='o', labels=helix_context, alpha=0.5)
    values = range(len(delta_deltaG))
    cm = plt.cm.Paired
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    #put error bars on numbers
    for i, row in enumerate(delta_deltaG):
        ax.errorbar(np.arange(0,len(junctionseqs)),row, yerr=[delta_deltaGerrub[i,:], delta_deltaGerrlb[i,:]], fmt='.', color=scalarMap.to_rgba(i),ecolor=scalarMap.to_rgba(i)) 
    ax.set_xticks(np.arange(0,len(junctionseqs)))
    ax.set_xticklabels(junctionseqs)
    #ax.legend_ = None
    ax.set_xlabel('Junction')
    ax.set_ylabel('delta delta G')
    ax.set_ylim((-1, 5))
    ax.set_xlim((-0.5,len(junctionseqs)+1))
    plt.tight_layout()
    return
    
def get_ddG_and_Errs(table, variant_number, delta_G_initial,delta_G_ub_initial,delta_G_lb_initial):    
    delta_deltaG= table.loc[variant_number]['dG'] - delta_G_initial
    delta_deltaGerrub = np.sqrt(pow(table.loc[variant_number]['dG_ub']-table.loc[variant_number]['dG'],2) + pow(delta_G_ub_initial,2))
    delta_deltaGerrlb = np.sqrt(pow(table.loc[variant_number]['dG_lb']-table.loc[variant_number]['dG'],2) + pow(delta_G_lb_initial,2))
    return delta_deltaG, delta_deltaGerrub, delta_deltaGerrlb
    
    
def plot_helixvshelix_Corr(table, variant_table, topology, loop=None, receptor=None, offset=None):
    if loop is None:
        loop = 'goodLoop'
    if receptor is None:
        receptor='R1'
    if offset is None:
        offset = 0  # amount to change helix_one_length by from default
    couldPlot = True
    criteria_central = np.all((np.array(variant_table['receptor'] == receptor),
                           np.array(variant_table['loop']==loop)),
                          axis=0)
    if topology == '':
        criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)
    else:
        criteria_central = np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)
      
    
    sub_table = variant_table[criteria_central]
    helix_context = np.unique(sub_table['helix_context'])
    #do all but last two
    helix_context = helix_context[1:-2]
    junctionseqs = np.unique(sub_table['junction_sequence'])
    
    #choose just one length = 10 
    length = 10
    delta_G_initial = variant_table.loc[0]['dG']
    delta_G_ub_initial = variant_table.loc[0]['dG_ub']-variant_table.loc[0]['dG']
    delta_G_lb_initial = variant_table.loc[0]['dG_lb']-variant_table.loc[0]['dG']
    
    delta_deltaG = np.ones((len(helix_context), len(junctionseqs)))*np.nan
    delta_deltaGerrub = np.ones((len(helix_context), len(junctionseqs)))*np.nan
    delta_deltaGerrlb = np.ones((len(helix_context), len(junctionseqs)))*np.nan
    for i, sequence in enumerate(helix_context):
        for j, jsequence in enumerate(junctionseqs):
            helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
            variant_number = sub_table[np.all((np.array(sub_table['helix_context']==sequence),
                                                np.array(sub_table['total_length']==length),
                                                np.array(sub_table['helix_one_length']==helix_one_length),
                                                np.array(sub_table['junction_sequence']==jsequence)),axis=0)].index
            #calculate ddG, and errorbars ddG (sqrt of sum or squares)
            if len(variant_number)>0:
                delta_deltaG[i][j],delta_deltaGerrub[i][j],delta_deltaGerrlb[i][j] =get_ddG_and_Errs(sub_table, variant_number, delta_G_initial,delta_G_ub_initial, delta_G_lb_initial)
    #calculate spearman correlation (rank correlation)
    rho, p = st.spearmanr(delta_deltaG, axis = 1)
    fig = plt.figure(figsize=(10,10))
    plt.pcolor(rho, cmap='RdGy')
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(0,len(helix_context))+0.5)
    ax.set_yticks(np.arange(0,len(helix_context))+0.5)
    ax.set_xticklabels(helix_context)
    ax.set_yticklabels(helix_context)
    #ax.legend_ = None
    #calculate pearson correlation (linear correlation)
    
    return
    
def plot_juctionvsSeqrank_Corr(table, variant_table, topologies, loop=None, receptor=None, offset=None):
    #generate clustered heat map of ranking of each sequence variant vs topology
    #mastrix of ranked values vs helix
    rankedInds = np.empty((12,0), int)
    #junctions deleted because >2 nans
    numtodelete = list()
    #total number of available junctions
    totaljunctions = list()
    junctionskept = list()
    for m, topology in enumerate(topologies):
        if loop is None:
            loop = 'goodLoop'
        if receptor is None:
            receptor='R1'
        if offset is None:
            offset = 0  # amount to change helix_one_length by from default
        couldPlot = True
        criteria_central = np.all((np.array(variant_table['receptor'] == receptor),
                            np.array(variant_table['loop']==loop)),
                            axis=0)
        if topology == '':
            criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)
        else:
            criteria_central = np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)
        
        
        sub_table = variant_table[criteria_central]
        helix_context = np.unique(sub_table['helix_context'])
        junctionseqs = np.unique(sub_table['junction_sequence'])
        totaljunctions.append(len(junctionseqs))
        #choose just one length = 10 
        length = 10
        delta_G_initial = variant_table.loc[0]['dG']
        delta_G_ub_initial = variant_table.loc[0]['dG_ub']-variant_table.loc[0]['dG']
        delta_G_lb_initial = variant_table.loc[0]['dG_lb']-variant_table.loc[0]['dG']
        
        delta_deltaG = np.ones((len(helix_context), len(junctionseqs)))*np.nan
        delta_deltaGerrub = np.ones((len(helix_context), len(junctionseqs)))*np.nan
        delta_deltaGerrlb = np.ones((len(helix_context), len(junctionseqs)))*np.nan
        for i, sequence in enumerate(helix_context):  
            for j, jsequence in enumerate(junctionseqs):
                helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
                variant_number = sub_table[np.all((np.array(sub_table['helix_context']==sequence),
                                                    np.array(sub_table['total_length']==length),
                                                    np.array(sub_table['helix_one_length']==helix_one_length),
                                                    np.array(sub_table['junction_sequence']==jsequence)),axis=0)].index
                #calculate ddG, and errorbars ddG (sqrt of sum or squares)
                if len(variant_number)>0:
                    delta_deltaG[i][j],delta_deltaGerrub[i][j],delta_deltaGerrlb[i][j] = get_ddG_and_Errs(sub_table, variant_number, delta_G_initial,delta_G_ub_initial, delta_G_lb_initial)
        indsdelete = []
        for k, col in enumerate(delta_deltaG.T):                     
            if np.sum(np.isnan(col))>2:
                indsdelete.append(k)
        numtodelete.append(len(indsdelete))
        if len(indsdelete)>1:
            #remove entris in ddG that have greater than 2 NaNs
            delta_deltaG = np.delete(delta_deltaG, indsdelete, axis=1)
        #rank ddG matrix and output the indices of the helices rank
        rankedInds = np.append(rankedInds, delta_deltaG.argsort(0),axis = 1)
        junctionskept.append(delta_deltaG.shape[1])
    #generate rowlabels for dendogram
    rowlabels = list()
    for i, t in enumerate(topologies):
        name = convert_nomen([t])[0]
        rowlabels = rowlabels + [name]*(totaljunctions[i]-numtodelete[i])
    rowlabels = np.array(rowlabels)    
    heatmapfun.plotCoverageHeatMap(rankedInds,rowlabels=rowlabels )
    plt.xticks(np.arange(0,len(helix_context)), helix_context, size=11, rotation=0 )
    plt.title('Clustering of Junction Topology vs Helix Context rank')
    return

def plot_juctionvslength_Corr(table, variant_table, topologies,helix_context= None, loop=None, receptor=None, offset=None):
    #generate clustered heat map of ranking of each sequence variant vs topology
    #mastrix of ranked values vs helix
    total_lengths = np.array([8,9,10,11,12])
    ddGmat = np.empty((len(total_lengths),0))
    #junctions deleted because >2 nans
    numtodelete = list()
    #total number of available junctions
    totaljunctions = list()
    junctionskept = list()
    for m, topology in enumerate(topologies):
        if loop is None:
            loop = 'goodLoop'
        if receptor is None:
            receptor='R1'
        if offset is None:
            offset = 0  # amount to change helix_one_length by from default
        if helix_context is None:
            helix_context = 'rigid'  
        couldPlot = True
        criteria_central = np.all((np.array(variant_table['receptor'] == receptor),
                            np.array(variant_table['loop']==loop),
                            np.array(variant_table['helix_context']==helix_context)),
                            axis=0)
        if topology == '':
            criteria_central = np.all((criteria_central, np.array(variant_table['junction_sequence'] =='_')), axis=0)
        else:
            criteria_central = np.all((criteria_central, np.array(variant_table['topology']==topology)), axis=0)
        
        
        sub_table = variant_table[criteria_central]
        
        junctionseqs = np.unique(sub_table['junction_sequence'])
        totaljunctions.append(len(junctionseqs))
        
        delta_G_initial = variant_table.loc[0]['dG']
        delta_G_ub_initial = variant_table.loc[0]['dG_ub']-variant_table.loc[0]['dG']
        delta_G_lb_initial = variant_table.loc[0]['dG_lb']-variant_table.loc[0]['dG']
        
        delta_deltaG = np.ones((len(total_lengths), len(junctionseqs)))*np.nan
        delta_deltaGerrub = np.ones((len(total_lengths), len(junctionseqs)))*np.nan
        delta_deltaGerrlb = np.ones((len(total_lengths), len(junctionseqs)))*np.nan
        for i, length in enumerate(total_lengths):  
            for j, jsequence in enumerate(junctionseqs):
                helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
                variant_number = sub_table[np.all(( np.array(sub_table['total_length']==length),
                                                    np.array(sub_table['helix_one_length']==helix_one_length),
                                                    np.array(sub_table['junction_sequence']==jsequence)),axis=0)].index
                #calculate ddG, and errorbars ddG (sqrt of sum or squares)
                if len(variant_number)>0:
                    delta_deltaG[i][j],delta_deltaGerrub[i][j],delta_deltaGerrlb[i][j] = get_ddG_and_Errs(sub_table, variant_number, delta_G_initial,delta_G_ub_initial, delta_G_lb_initial)
        print delta_deltaG
        indsdelete = []
        for k, col in enumerate(delta_deltaG.T):                     
            if np.sum(np.isfinite(col))<2:
                indsdelete.append(k)
        numtodelete.append(len(indsdelete))
        if len(indsdelete)>1:
            #remove entris in ddG that have greater than 2 NaNs
            delta_deltaG = np.delete(delta_deltaG, indsdelete, axis=1)
        #rank ddG matrix and output the indices of the helices rank
        ddGmat = np.append(ddGmat, delta_deltaG,axis = 1)
        junctionskept.append(delta_deltaG.shape[1])
    #generate rowlabels for dendogram
    rowlabels = list()
    for i, t in enumerate(topologies):
        name = convert_nomen([t])[0]
        rowlabels = rowlabels + [name]*(totaljunctions[i]-numtodelete[i])
    rowlabels = np.array(rowlabels)    
    plot_heatmap(ddGmat,rowlabels=rowlabels)
    plt.title('Clustering of Junction Topology vs Length')
    return

def plot_heatmap(matrix,rowlabels=None, columnlabels=None, rowIndx=None, fontSize=None, columnIndx=None, cmap=None, vmin=None, vmax=None, colorbar = None):
    if fontSize==None:
        fontSize='6'
    else: fontsize = str(fontSize)
    
    numSamples = np.size(matrix, axis=1)
    numMotifs = np.size(matrix, axis=0)
    metric = 'euclidean'
    if rowlabels is None:
        rowlabels = np.arange(numSamples)
    if columnlabels is None:
        columnlabels = np.arange(numMotifs)
    if colorbar is None:
        colorbar = True
    
    
    if rowIndx is None:
        rowIndx    = sch.leaves_list(sch.linkage(np.transpose(matrix), method='weighted', metric=metric)).astype(int)
    if columnIndx is None:
        columnIndx = sch.leaves_list(sch.linkage(matrix[:,rowIndx], method='weighted', metric=metric)).astype(int)
    #rowIndx = np.arange(len(rowlabels))
    
    if cmap is None:
        cm = plt.get_cmap('seismic')
    else:
        cm = plt.get_cmap(cmap)
    
    matrixvals = matrix[np.isfinite(matrix)]
    cmap_max = np.max([np.abs(np.min(matrixvals)), np.max(matrixvals)])
    if vmin is None:
        vmin = np.min(matrixvals)
    if vmax is None:
        vmax = np.max(matrixvals)
    fig = plt.figure(figsize=(11,13))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.transpose(matrix)[rowIndx,:],extent=[0, 16, 0, 39] )
    plt.xticks(np.arange(0,17,4), columnlabels, size=15, rotation=90)
    plt.yticks(np.arange(0,39,0.3333), rowlabels[rowIndx[0::len(rowIndx)/(39*3)]], size=9)
    ax.set_xlabel('Helix Length')
    ax.set_ylabel('Junction Type')
    cb = plt.colorbar(im)

    return
