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
from scikits.bootstrap import bootstrap

class Parameters():
    def __init__(self):
    
        # save the units of concentration given in the binding series
        self.RT = 0.582  # kcal/mol at 20 degrees celsius
        self.min_deltaG = -12
        self.max_deltaG = -3
        
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
            newtable.iloc[i]['dG_lb'], newtable.iloc[i]['dG_ub'] = bootstrap.ci(sub_table_filtered['dG'], np.median)
    return newtable
        

def filterFitParameters(sub_table):
    binding_curves = np.array([np.array(sub_table[i]) for i in range(8)])
    num_curves = binding_curves.shape[1]
    #indices = np.arange(num_curves)[np.all((np.array(sub_table['rsq'] > 0.5), np.sum(np.isnan(binding_curves), 0) < 2), axis=0)]
    indices = np.arange(num_curves)[np.array(sub_table['rsq'] > 0.5)]
    sub_table = sub_table.iloc[indices]
    return sub_table

def bindingCurve(concentrations, dG, fmax=None, fmin=None):
    if fmax is None:
        fmax = 1
    if fmin is None:
        fmin = 0
    parameters = Parameters()
    return fmax*concentrations/(concentrations + np.exp(dG/parameters.RT)/1E-9) + fmin

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

def findVariantNumbers(table, criteria_dict):
    for name, value in criteria_dict.items():
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

def plot_dG_errorbars_vs_length(series):
    ax = plt.gca()
    fmt, wiggle, color = getMarker(series)
    ax.errorbar(series['total_length']+wiggle, series['dG'],
                    yerr=[[series['dG'] - series['dG_lb']], [series['dG_ub'] - series['dG']]], fmt=fmt, color=color, ecolor='k')
    return

def plot_parameter_vs_length(series, p1, p2):
    ax = plt.gca()
    fmt, wiggle, color = getMarker(series)
    ax.plot(series[p1]+wiggle, series[p2], fmt, color=color)
    return ax

def plot_over_coordinate(per_variant):
    per_variant.sort('total_length', inplace=True)
    total_lengths = np.unique(per_variant['total_length']).astype(int)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    # plot gray boundaries
    ax.fill_between(total_lengths, [np.mean(per_variant[per_variant['total_length']==length]['dG_lb']) for length in total_lengths],
                    [np.mean(per_variant[per_variant['total_length']==length]['dG_ub']) for length in total_lengths],
                    facecolor='0.5', alpha=0.05, linewidth=0)
    #ax.fill_between(total_lengths, [np.min(per_variant[per_variant['total_length']==length]['dG']) for length in total_lengths],
    #                [np.max(per_variant[per_variant['total_length']==length]['dG']) for length in total_lengths],
    #                facecolor='0.5', alpha=0.05, linewidth=0)
    
    # plot points with errorbars
    for i in range(len(per_variant)):
        series = per_variant.iloc[i]
        plot_dG_errorbars_vs_length(series)
    
    # plot ticks and labels
    ax.set_xticks(total_lengths)
    ax.legend_ = None
    ax.set_xlabel('total length')
    ax.set_ylabel('dG (kcal/mol)')
    ax.set_ylim((-12, -6))
    ax.set_xlim((total_lengths[0]-1, total_lengths[-1]+1))
    plt.subplots_adjust(bottom=0.15, left=0.2)
    ax.grid(linestyle=':', alpha=0.5)
    
    xvalues = np.arange(len(per_variant))
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.bar(xvalues, per_variant['numTests'], color='0.75', linewidth=0)
    ax.set_xticks(xvalues+0.4)
    ax.set_xticklabels(np.array(per_variant['total_length'], dtype=int), rotation=90)
    ax.set_ylabel('number of tests')
    ax.set_xlabel('total length')
    for i in xvalues:
        series = per_variant.iloc[i]
        fmt, wiggle, color = getMarker(series)
        ax.plot(i+0.4, series['numTests']+1, fmt, color=color)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    return
    

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
        delta_deltaG = np.ones((len(junction_sequences), len(total_lengths)))*np.nan
        for i, sequence in enumerate(junction_sequences):
            for j, length in enumerate(total_lengths):
                helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
                variant_number = sub_table[np.all((np.array(sub_table['junction_sequence']==sequence),
                                                   np.array(sub_table['total_length']==length),
                                                   np.array(sub_table['helix_one_length']==helix_one_length)),axis=0)]['variant_number']
                if len(variant_number)>0:
                    delta_deltaG[i][j] = sub_table.loc[variant_number]['dG'] - delta_G_initial
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        plotfun.plot_manylines(delta_deltaG, cmap='Paired', marker='o', labels=junction_sequences, alpha=0.5)
        ax.set_xticks(total_lengths-8)
        ax.set_xticklabels(total_lengths.astype(str))
        ax.legend_ = None
        ax.set_xlabel('total length')
        ax.set_ylabel('delta delta G')
        ax.set_ylim((-1, 5))
        plt.tight_layout()
    else: couldPlot = False
    return couldPlot


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
    delta_deltaG = np.ones((len(helix_context), len(total_lengths)))*np.nan
    for i, sequence in enumerate(helix_context):
        for j, length in enumerate(total_lengths):
            helix_one_length = np.floor((length - sub_table['junction_length'].iloc[0])*0.5) + offset
            variant_number = sub_table[np.all((np.array(sub_table['helix_context']==sequence),
                                               np.array(sub_table['total_length']==length),
                                               np.array(sub_table['helix_one_length']==helix_one_length)),axis=0)]['variant_number']
            if len(variant_number)>0:
                delta_deltaG[i][j] = sub_table.loc[variant_number]['dG'] - delta_G_initial
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    plotfun.plot_manylines(delta_deltaG, cmap='Paired', marker='o', labels=helix_context, alpha=0.5)
    ax.set_xticks(total_lengths-8)
    ax.set_xticklabels(total_lengths.astype(str))
    ax.legend_ = None
    ax.set_xlabel('total length')
    ax.set_ylabel('delta delta G')
    ax.set_ylim((-1, 5))
    plt.tight_layout()
    return