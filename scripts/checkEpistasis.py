#!/usr/bin/env python

# Make variant table etc into something to fit into linear model
# ---------------------------------------------
#
#
# Sarah Denny
# December 2014

import sys
import os
import time
import re
import argparse
import subprocess
import multiprocessing
import shutil
import uuid
import numpy as np
import scipy.io as sio
import pandas as pd
import datetime
import functools
import variantFun
import IMlibs
import fitFun
import seaborn
parameters = variantFun.Parameters()


flowpiece = 'wc'
dirname = '141111_miseq_run_tecto_TAL_VR/with_all_clusters/'
name = dirname + 'binding_curves'
fittedBindingFilename = '%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPfitted'%(name, flowpiece) 
variantFittedFilename = '%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'%(name, flowpiece)
variant_table = pd.read_table(variantFittedFilename)

def getDifference(variant_table, variant1, variant2, parameter=None, name=None):
    if name is None: name = '%d_%d'%(variant2, variant1)
    if parameter is None: parameter='dG'
    ddG = pd.DataFrame(columns=[parameter, 'eminus', 'eplus'], index=[name])
    ddG.loc[name, parameter] = variant_table.loc[variant1, parameter] - variant_table.loc[variant2, parameter]
    ddG.loc[name, 'eminus'] = np.sqrt(np.power(variant_table.loc[variant1, parameter]-variant_table.loc[variant1, parameter+'_lb'], 2) +
                                      np.power(variant_table.loc[variant2, parameter]-variant_table.loc[variant2, parameter+'_lb'], 2))
    ddG.loc[name, 'eplus'] = np.sqrt(np.power(variant_table.loc[variant1, parameter+'_ub']-variant_table.loc[variant1, parameter], 2) +
                                     np.power(variant_table.loc[variant2, parameter+'_ub']-variant_table.loc[variant2, parameter], 2))
    return ddG

def getDifferenceSimple(ddG1, ddG2, name=None):
    if name is None: name='%s|%s'%(ddG2.index[0], ddG1.index[0])
    parameter = 'dG'
    ddG = pd.DataFrame(columns=[parameter, 'eminus', 'eplus'], index=[name])
    ddG1 = ddG1.iloc[0]
    ddG2 = ddG2.iloc[0]
    ddG.loc[name, parameter] = ddG1.loc[parameter] - ddG2.loc[parameter]
    for e in ['eminus', 'eplus']:
        ddG.loc[name, e] = np.sqrt(np.power(ddG1.loc[e], 2) + np.power(ddG2.loc[e], 2))
    return ddG

def getSimple(variant_table, variant1, parameter=None, name=None):
    if parameter is None:
        parameter = 'dG'
    if name is None: name='%d'%(variant1)
    ddG = pd.DataFrame(columns=[parameter, 'eminus', 'eplus'], index=[name])
    ddG.loc[name, parameter] = variant_table.loc[variant1, parameter]
    ddG.loc[name, 'eminus'] = variant_table.loc[variant1, parameter]-variant_table.loc[variant1, parameter+'_lb']
    ddG.loc[name, 'eplus']  = variant_table.loc[variant1, parameter+'_ub']-variant_table.loc[variant1, parameter]
    ddG.loc[name, 'variant'] = variant1
    return ddG

# find single base difference
ddG = getDifference(variant_table, 0, 1, parameter=None, name='wc_to_rigid')

# find all variants with motif in wc context. What is ddG between this context and another
total_length = 10
names_to_compare = [ 'topology', 'loop', 'receptor',  'junction_sequence',  'helix_one_length', 'total_length']
variants = variantFun.findVariantNumbers(variant_table, {'loop':'goodLoop', 'helix_context':'wc', 'total_length':total_length,
                                                         'receptor':'R1'})


ddG_context = {'none':pd.DataFrame(columns=['dG',  'eminus', 'eplus']), 'motif':pd.DataFrame(columns=['dG',  'eminus', 'eplus'])}
ddG_motif = {'wc':pd.DataFrame(columns=['dG',  'eminus', 'eplus']), 'rigid':pd.DataFrame(columns=['dG',  'eminus', 'eplus'])}
ddG_context['none'] = getDifference(variant_table, 0, 1)

all_info = {}
variant_subtable = variant_table.loc[variants]
all_junction_motifs = np.unique(variant_table.loc[variants, 'topology'].dropna())

for junction_motif in all_junction_motifs[:-1]:
    print junction_motif
    index = variant_subtable.loc[variant_subtable.loc[:, 'topology'] == junction_motif].index
    all_info[junction_motif] = {}
    for helix_one_length in np.unique(variant_subtable.loc[index, 'helix_one_length']):
        index2 = variant_subtable.loc[index].loc[variant_subtable.loc[index, 'helix_one_length'] == helix_one_length].index
        offset = variant_subtable.loc[index2[0], 'helix_one_length'] + variant_subtable.loc[index2[0], 'junction_length']*0.5
        all_info[junction_motif][offset] = {}
        for ind in index2:
            ddG_motif['wc'] = pd.concat([ddG_motif['wc'], getDifference(variant_table, ind, 1, parameter=None)])
            other_inds = variant_table.loc[(variant_table.loc[:, names_to_compare] == variant_table.loc[ind, names_to_compare]).all(axis=1)].index
            for ind2 in other_inds:
                if (variant_table.loc[ind2, 'helix_context'] == 'rigid'):
                    #ddG_context['motif'] = pd.concat([ddG_context['motif'], getDifference(variant_table, ind2, ind, parameter=None)])
                    #ddG_motif['rigid'] = pd.concat([ddG_motif['rigid'], getDifference(variant_table, ind2, 0, parameter=None)])
                    seq = variant_table.loc[ind2, 'junction_sequence']
                    all_info[junction_motif][offset][seq] = {}
                    #all_info[junction_motif][helix_one_length][seq]['motif'] = getDifferenceSimple(
                    #    getDifference(variant_table, ind2, 0),  getDifference(variant_table, ind, 1))
                    all_info[junction_motif][offset][seq] = pd.concat([getSimple(variant_table, ind, name='wc'), getSimple(variant_table, ind2, name='rigid')])
                    #all_info[junction_motif][helix_one_length][seq][1] = getSimple(variant_table, ind2, parameter=None)
                    #all_info[junction_motif][helix_one_length][seq] = pd.concat(all_info[junction_motif][helix_one_length][seq], axis=1)
        if all_info[junction_motif][offset]:
            all_info[junction_motif][offset] = pd.concat(all_info[junction_motif][offset])
        else:
            all_info[junction_motif].pop(offset)
    all_info[junction_motif] =  pd.concat(all_info[junction_motif])  
        
all_info = pd.concat(all_info)
all_info_old = all_info.copy()
all_info = all_info_old.loc[:, 'motif'].dropna()

# plot


junction_motif = 'B1_B1'
for junction_motif in ['B1', 'B1_B1', 'B2', 'B2_B2', 'B2_M', 'M', 'M_B1',  'M_M']:
    plt.figure(figsize = (5,4))
    helix_one_lengths = np.unique([index[0] for index in all_info.loc[junction_motif].index.tolist()])
    start_loc = 0
    xticks = []
    for length in helix_one_lengths:
        subinfo = all_info.loc[(junction_motif, length)]
        xvalues = np.arange(start_loc, start_loc+len(subinfo))
        yvalues = subinfo.loc[:, 'dG'].values
        yerr = [subinfo.loc[:, 'eminus'].values, subinfo.loc[:, 'eplus'].values]   
        plt.errorbar(xvalues, yvalues, yerr, fmt='o', label='%d'%(length))
        start_loc = xvalues[-1] + 1
        xticks = xticks + [index[0] for index in subinfo.index.tolist()]
    plt.legend()
    plt.xlim(-1, start_loc)
    plt.plot([-1, start_loc], [0,0], '0.5')
    plt.ylim(-2, 2)
    plt.xticks(np.arange(0, start_loc), xticks, rotation=90)
    plt.ylabel('dddG')
    plt.title(junction_motif)
    plt.tight_layout()
    plt.savefig('%s/flow_%s/figs_2015-03-17/wc_to_rigid_epistatics.length_%d.%s.pdf'%(dirname, flowpiece, total_length, junction_motif))
    
# plot histogram
junction_motifs = ['B1',  'B2', 'B1_B1',  'B2_B2', 'M', 'B2_M', 'M_B1',  'M_M']
heights = {'B1':4, 'B2':4, 'B1_B1':8,  'B2_B2':8, 'B2_M':20, 'M':8, 'M_B1':20,  'M_M':20}
lengths = [-2, -1, 0, 1, 2]
fig = plt.figure(figsize=(7.5,10))
gs = gridspec.GridSpec(len(junction_motifs), len(lengths), height_ratios=[float(heights[key])/np.max(heights.values()) for key in junction_motifs])
colors = sns.color_palette('Set2', len(junction_motifs))
for j, length in enumerate(lengths):   
    for i, junction_motif in enumerate(junction_motifs):
        height = heights[junction_motif]
        
        ax = fig.add_subplot(gs[i,j])
        if length in [index[0] for index in all_info.loc[junction_motif].index.tolist()]:
            subinfo = all_info.loc[(junction_motif, length)]
            ax.hist(subinfo.loc[:, 'dG'], bins = np.arange(-2, 2, 0.125), normed=False, alpha=.5, color=colors[i], label=junction_motif)
        else:
            ax.hist([np.nan], bins = np.arange(-2, 2, 0.125), normed=False, alpha=.5, color=colors[i], label=junction_motif)
        ax.set_ylim([0, height])
        ax.set_yticks(np.arange(0, height, 2))
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels([-2, -1, 0, 1, 2])
        ax.plot([0,0], [0,height], '0.75')
        if i == 0:
            ax.set_title(length)
        if i != len(junction_motifs) - 1:
            ax.set_xticklabels([])
        if j != 0:
            ax.set_yticklabels([])
        ax.annotate(junction_motif, xy=(0, 1) ,xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontsize=9)
plt.savefig('%s/flow_%s/figs_2015-03-17/wc_to_rigid_epistatics.length_%d.histogram.pdf'%(dirname, flowpiece, total_length))            

# plot scatterplot
junction_motifs = ['M_M','B2_M', 'M_B1', 'B1_B1',  'B2_B2', 'M',  'B1',  'B2',  ]
lengths = np.arange(4, 6.5, 0.5)
colors = sns.color_palette('Set2', len(junction_motifs))
for j, length in enumerate(lengths):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    colors = sns.color_palette('Set2', len(junction_motifs))
    xvalues = getSimple(variant_table, 1, parameter=None, name='wc')
    yvalues = getSimple(variant_table, 0, parameter=None, name='rigid')
    plt.plot([xvalues.dG-5, xvalues.dG+5], [yvalues.dG-5, yvalues.dG+5], '0.5')
    
    plt.errorbar(xvalues.dG, yvalues.dG, yerr=[yvalues.eminus, yvalues.eplus],
                 xerr=[xvalues.eminus, xvalues.eplus], fmt='o', color='k', alpha=0.5, label='None')
    for i, junction_motif in enumerate(junction_motifs):

        if length in [index[0] for index in all_info.loc[junction_motif].index.tolist()]:
            subinfo = all_info.loc[(junction_motif, length)]
            xvalues = subinfo.loc[[name for name in subinfo.index.tolist() if name[1] == 'wc' ]]
            yvalues = subinfo.loc[[name for name in subinfo.index.tolist() if name[1] == 'rigid' ]]

            plt.errorbar(xvalues.dG, yvalues.dG, yerr=[yvalues.eminus, yvalues.eplus],
                         xerr=[xvalues.eminus, xvalues.eplus], fmt='o', color=colors[i], alpha=0.5, label=junction_motif)
    plt.legend(loc='upper left')
    lims = [-11.5, -8]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('dG -2CG;UA')
    plt.ylabel('dG -2GU;UG')
    plt.title('helix one length: %4.1f'%length)
    plt.savefig('%s/variation_with_GUwobble/wc_to_rigid_epistatics.flow_%s.length_%d.helix_one_length_%4.1f.scatterplot.pdf'%(dirname, flowpiece, total_length, length)) 