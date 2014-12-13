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
import CPlibs

class Parameters():
    def __init__(self):
    
        # save the units of concentration given in the binding series
        self.RT = 0.582  # kcal/mol at 20 degrees celsius
        self.min_deltaG = -12
        self.max_deltaG = -3
        
def perVariantInfo(table):
    variants = np.arange(0, np.max(table['variant_number']))
    columns = [name for name in table][:12] + ['numTests', 'numRejects', 'kd', 'dG', 'fmax', 'fmin']
    newtable = pd.DataFrame(columns=columns, index=np.arange(len(variants)))
    for i, variant in enumerate(variants):
        if i%1000==0: print 'computing iteration %d'%i
        sub_table = table[table['variant_number']==variant]
        if len(sub_table) > 0:
            sub_table_filtered = filterFitParameters(sub_table)[['kd', 'dG', 'fmax', 'fmin']]
            newtable.iloc[i]['numTests'] = len(sub_table_filtered)
            newtable.iloc[i]['numRejects'] = len(sub_table) - len(sub_table_filtered)
            newtable.iloc[i]['kd':] = np.median(sub_table_filtered, 0)
            newtable.iloc[i][:'total_length'] = sub_table.iloc[0][:'total_length']
    return newtable
        

def filterFitParameters(sub_table):
    sub_table = sub_table[sub_table['fit_success']==1]
    sub_table = sub_table[sub_table['rsq']>0.5]
    sub_table = sub_table[sub_table['fraction_consensus']>=67] # 2/3 majority
    return sub_table

def bindingCurve(concentrations, kd, fmax=None, fmin=None):
    if fmax is None:
        fmax = 1
    if fmin is None:
        fmin = 0
    return fmax*concentrations/(concentrations + kd) + fmin

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
    frac_bound = np.median(binding_curves, axis=1)
    [percentile25, percentile75] = np.percentile(binding_curves, [25,75],axis=1)
    sub_table['kd'] = np.exp(sub_table['dG']/parameters.RT)/1E-9
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.errorbar(concentrations, frac_bound, yerr=[frac_bound-percentile25, percentile75-frac_bound], fmt='o')
    ax.plot(np.logspace(-1, 4, 50), bindingCurve(np.logspace(-1, 4, 50), np.median(sub_table['kd']), np.median(sub_table['fmax']), np.median(sub_table['fmin'])),'k')
    ax.set_xscale('log')
    ax.set_xlabel('concentration (nM)')
    ax.set_ylabel('normalized fluorescence')
    ax.set_ylim((0, 1.5))
    plt.title(name)
    plt.tight_layout()
    
    # make histogram
    parameters = Parameters()
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    xbins = np.arange(parameters.min_deltaG, parameters.max_deltaG)-0.5
    histogram.compare([parameters.RT*np.log(sub_table['kd']*1E-9)], bar=True, xbins=xbins, normalize=False )
    ax.set_xlim((parameters.min_deltaG-1, parameters.max_deltaG+1))
    ax.set_xlabel('dG (kcal/mol)')
    ax.legend_ = None
    ax.set_ylabel('number')
    plt.title(name)
    plt.tight_layout()
    return

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