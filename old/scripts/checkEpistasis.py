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
import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
parameters = variantFun.Parameters()



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



def plotOffsetScatterplot(variant_table, all_info, offset, junction_motifs=None, colors=None):
    if junction_motifs is None:
        junction_motifs = np.unique([index[0] for index in all_info.index.tolist()])
    if colors is None:
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(junction_motifs)-1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='Paired')
        colors = [scalarMap.to_rgba(i) for i in range(len(junction_motifs))]
        
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    xvalues = getSimple(variant_table, 1, parameter=None, name='wc')
    yvalues = getSimple(variant_table, 0, parameter=None, name='rigid')
    plt.plot([xvalues.dG-5, xvalues.dG+5], [yvalues.dG-5, yvalues.dG+5], '0.5')
    
    plt.errorbar(xvalues.dG, yvalues.dG, yerr=[yvalues.eminus, yvalues.eplus],
                 xerr=[xvalues.eminus, xvalues.eplus], fmt='o', color='k', alpha=0.5, label='None')
    
    for i, junction_motif in enumerate(junction_motifs):

        if offset in [index[0] for index in all_info.loc[junction_motif].index.tolist()]:
            subinfo = all_info.loc[(junction_motif, offset)]
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
    plt.title('helix one length: %4.1f'%offset)
    return

def plotScatterplots(variant_table, all_info, offsets=None, junction_motifs=None, colors=None):
    if junction_motifs is None:
        junction_motifs = np.unique([index[0] for index in all_info.index.tolist()])
    if offsets is None:
        offsets = np.unique([index[1] for index in all_info.index.tolist()])
    for offset in offsets:
        if offset in [name[1] for name in all_info.loc[junction_motifs].index.tolist()]:
            plotOffsetScatterplot(variant_table, all_info, offset, junction_motifs, colors)
    return

if __name__ == '__main__':
    # find all variants with motif in wc context. What is ddG between this context and another
    
    flowpiece = 'rigid'
    dirname = '141111_miseq_run_tecto_TAL_VR/with_all_clusters/'
    name = dirname + 'binding_curves'
    variantFittedFilename = '%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'%(name, flowpiece)
    variant_table = pd.read_table(variantFittedFilename)
    
    total_length = 10
    names_to_compare = [ 'topology', 'loop', 'receptor',  'junction_sequence',  'helix_one_length', 'total_length']
    variants = variantFun.findVariantNumbers(table, {'loop':'goodLoop', 'helix_context':'wc', 'total_length':total_length,
                                                             'receptor':'R1'})
    all_junction_motifs = np.unique(variant_table.loc[variants, 'topology'].dropna())
    
    variant_subtable = variant_table.loc[variants]
    
    all_info = {}
    for junction_motif in all_junction_motifs[:-1]:
        print junction_motif
        
        # find indices of those variants with matching junction motifs
        index = variant_subtable.loc[variant_subtable.loc[:, 'topology'] == junction_motif].index
        all_info[junction_motif] = {}
        
        for helix_one_length in np.unique(variant_subtable.loc[index, 'helix_one_length']):
            
            # find indices of those variants with same helix one length
            index2 = variant_subtable.loc[index].loc[variant_subtable.loc[index, 'helix_one_length'] == helix_one_length].index
            
            # but rather than saving helix one length, save that plus the junction length / 2 for easier comparisons between motifs later
            offset = variant_subtable.loc[index2[0], 'helix_one_length'] + variant_subtable.loc[index2[0], 'junction_length']*0.5
            all_info[junction_motif][offset] = {}
            for ind in index2:
                
                # now check if that motif and helix length also exists in rigid context
                other_inds = variant_table.loc[(variant_table.loc[:, names_to_compare] == variant_table.loc[ind, names_to_compare]).all(axis=1)].index
                for ind2 in other_inds:
                    
                    # assuming there is only one partner with same "names_to_compare" and length rigid
                    if (variant_table.loc[ind2, 'helix_context'] == 'rigid'):
    
                        seq = variant_table.loc[ind2, 'junction_sequence']
                        all_info[junction_motif][offset][seq] = pd.concat([getSimple(variant_table, ind, name='wc'), getSimple(variant_table, ind2, name='rigid')])
            
            # concatenate into data structure
            if all_info[junction_motif][offset]:
                all_info[junction_motif][offset] = pd.concat(all_info[junction_motif][offset])
            else:
                all_info[junction_motif].pop(offset)
        all_info[junction_motif] =  pd.concat(all_info[junction_motif])  
            
    all_info = pd.concat(all_info)
    all_info.to_csv('141111_miseq_run_tecto_TAL_VR/with_all_clusters/variation_with_GUwobble/all_info.length_%d.flow_%s.dat'%(total_length, flowpiece), sep='\t')
    sys.exit()
    all_info.to_csv('variation_with_GUwobble/all_info.length_%d.flow_%s.dat'%(total_length, flowpiece), sep='\t')
   
    
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
    
    junction_motifs = ['B2_M_M','M_M_B1', 'M_M', 'M_B1_B1','B2_B2_M','B1_B1_B1', 'B2_B2_B2', 'M_B1', 'B2_M', 'B1_B1',  'B2_B2', 'M',  'B1',  'B2',  ]
    junction_motifs = ['B2_M_M', 'M_M','B2_B2_M','B2_B2_B2', 'B2_M', 'B2_B2',  'B2', 'M_M_B1',  'M_B1_B1','B1_B1_B1', 'M_B1', 'B1_B1',  'M',  'B1',   ]
    
    lengths = np.arange(3.5, 7, 0.5)

    
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(junction_motifs)-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='Paired')
    colors = np.array([scalarMap.to_rgba(i) for i in range(len(junction_motifs))])
    
    junction_motifs_test = ['B2_M_M', 'M_M_B1']
    junction_motifs_test = ['M_M', 'M_B1', 'B2_M', 'B1_B1',  'B2_B2', 'M',  'B1',  'B2',  ]
    lengths = np.arange(2.5, 7, 0.5)
    plotScatterplots(variant_table, all_info, lengths, junction_motifs_test, colors=colors[np.in1d(junction_motifs,junction_motifs_test)])    
    
    
    indices = [name for name in all_info.index.tolist() if (name[1] == 4 or name[1] == 4.5 or name[1] == 5 or name[1] == 5.5)]
    x = all_info.loc[[name for name in indices if name[3] == 'wc'], 'dG'].astype(float)
    
    y = all_info.loc[[name for name in indices if name[3] == 'rigid'], 'dG'].astype(float)
    junctions = ['B1', 'B1_B1', 'B1_B1_B1', 'B2_B2', 'B2_B2_B2', 'B2_B2_M', 'B2_M',
       'B2_M_M', 'M', 'M_B1', 'M_B1_B1', 'M_M', 'M_M_B1']
    
    data = pd.DataFrame(columns=['wc', 'rigid'])
    data.loc[:, 'wc'] = x.values
    data.loc[:, 'rigid'] = y.values
    data.dropna(inplace=True)
    indx = np.isfinite(x)*np.isfinite(y)
    p = np.polyfit(x[indx], y[indx], 1)

    plt.figure(figsize=(4,4));
    plt.scatter(x, y, facecolors='0.5', edgecolors='k', alpha=0.5)
    plt.plot(np.arange(-12, 0), p[0]*np.arange(-12, 0) + p[1], 'r')
    plt.xlim([-12, -7])
    plt.ylim([-12, -7])
    plt.xlabel('dG -2CG;UA')
    plt.ylabel('dG -2GU;UG')
    plt.tight_layout()
    
    xlim =  [-11, -7]
    ylim = [-11, -7]
    color = 'vermillion'
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    sns.kdeplot(data.loc[:, 'wc'], data.loc[:, 'rigid'], cmap='Greys' ,n_levels=40, shade=True, gridsize=50, ax=ax)
    ax.plot(xlim, ylim)
    a_wc = tectoData.loadTectoData(table, variant_table, 1)
    a_rigid= tectoData.loadTectoData(table, variant_table, 0)
    ax.errorbar(a_wc.affinity_params.loc['dG'], a_rigid.affinity_params.loc['dG'],
                yerr=[[a_rigid.affinity_params.loc['eminus']], [a_rigid.affinity_params.loc['eplus']]],
                xerr=[[a_wc.affinity_params.loc['eminus']], [a_wc.affinity_params.loc['eplus']]],
                fmt='o', elinewidth=1, capsize=2, capthick=1, alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
    
    
    name1 = ('good', 'dG'); name2 = ('bad', 'dG')
    g = sns.JointGrid(x, y, size=3.75, ratio=7, space=0, dropna=True, xlim=xlim, ylim=xlim)
    g.plot_marginals(sns.kdeplot, color=sns.xkcd_rgb[color], shade=True)
    #g.plot_joint(plt.hexbin, cmap='Greys', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], gridsize=40)
    g.plot_joint(sns.kdeplot, cmap='Greys',n_levels=40, shade=True, gridsize=50, clip=[(xlim[0], xlim[1]), (ylim[0], ylim[1])])
    #g.plot_joint(plt.scatter, alpha=0.5, edgecolors=sns.xkcd_rgb['charcoal'], facecolors='none')
    g.set_axis_labels('GGAA loop dG (kcal/mol)', 'GAAA loop dG (kcal/mol)')
    #.annotate(st.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");
    
    
    