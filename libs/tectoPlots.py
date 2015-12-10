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
import statsmodels.nonparametric.api as smnp
from sklearn.decomposition import PCA as sklearnPCA
import scipy.cluster.hierarchy as sch
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1})
import seqfun
import hjh.junction
from tectoThreeWay import returnFracKey, returnBindingKey, returnLoopFracKey


    

def plotChevronPlot(dGs, index=None, flow_id=None):
    #cluster = 2
    #index = labels.loc[labels==cluster].index
    if index is None: index = dGs.index
    if flow_id is None: flow_id = 'wc10'
    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(2, 2, top=0.975, right=0.975, hspace=0.30, bottom=0.1, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, :])
    x = [8, 9, 10, 11, 12]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['8_1', '9_1','10_2','11_2','12_3',]]].transpose()
    for ind in y:
        ax1.scatter(x, y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax1.plot(x, y, 'k-', alpha=0.1)
    ax1.set_xlim(7.5, 12.5)
    ax1.set_xticks(x)
    ax1.set_ylim(-12, -5)
    ax1.set_xlabel('length (bp)')
    
    ax2 = fig.add_subplot(gs[1, 0])
    x = [0, 1, 2]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['9_0', '9_1','9_2']]].transpose()
    for ind in y:
        ax2.scatter(x, y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax2.plot(x, y, 'k-', alpha=0.1)
    ax2.set_xlim(-.5, 2.5)
    ax2.set_xticks(x)
    ax2.set_ylim(-12, -8)
    ax2.set_yticks([-12, -11, -10, -9, -8])
    
    ax3 = fig.add_subplot(gs[1,1])
    x = [1, 2, 3]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['10_1','10_2','10_3']]].transpose()
    for ind in y:
        ax3.scatter(x, y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax3.plot(x, y, 'k-', alpha=0.1)
    ax3.set_xlim(.5, 3.5)
    ax3.set_xticks(x)
    ax3.set_ylim(-12, -8)
    ax3.set_yticklabels([])
    ax3.set_yticks([-12, -11, -10, -9, -8])
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(top='off', right='off')
        
    plt.annotate('length to receptor (bp)', xy=(0.5, 0.01),
                     xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='bottom',
                     fontsize=12)
    
    
def plotChevronPlot2(dGs, index=None, flow_id=None):
    #cluster = 2
    #index = labels.loc[labels==cluster].index
    if index is None: index = dGs.index
    if flow_id is None: flow_id = 'wc10'
    
    # settings for plot
    jitter_scale = 0.05
    jitter_mean = 0
    num_points = len(dGs.loc[index])
    jitter = st.norm.rvs(jitter_mean, jitter_scale, num_points)
    
    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(2, 2, top=0.975, right=0.975, hspace=0.30, bottom=0.1, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, :])
    x = [8, 9, 10, 11, 12]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['8_1', '9_1','10_2','11_2','12_3',]]].transpose()
    for i, ind in enumerate(y):
        ax1.scatter(x+jitter[i], y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax1.errorbar(x, y.mean(axis=1),  yerr=y.std(axis=1), alpha=0.5, fmt='-', elinewidth=2,
                 capsize=4, capthick=2, color='r', linewidth=2)
    ax1.set_xlim(7.5, 12.5)
    ax1.set_xticks(x)
    ax1.set_ylim(-12, -5)
    ax1.set_xlabel('length (bp)')
    
    ax2 = fig.add_subplot(gs[1, 0])
    x = [0, 1, 2]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['9_0', '9_1','9_2']]].transpose()
    for i, ind in enumerate(y):
        ax2.scatter(x+jitter[i], y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax2.errorbar(x, y.mean(axis=1), yerr=y.std(axis=1), alpha=0.5, fmt='-', elinewidth=2,
                 capsize=2, capthick=2, color='r', linewidth=2)
    ax2.set_xlim(-.5, 2.5)
    ax2.set_xticks(x)
    ax2.set_ylim(-12, -8)
    ax2.set_yticks([-12, -11, -10, -9, -8])
    
    ax3 = fig.add_subplot(gs[1,1])
    x = [1, 2, 3]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['10_1','10_2','10_3']]].transpose()
    for i, ind in enumerate(y):
        ax3.scatter(x+jitter[i], y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax3.errorbar(x, y.mean(axis=1),  yerr=y.std(axis=1), alpha=0.5, fmt='-', elinewidth=2,
                 capsize=2, capthick=2, color='r', linewidth=2)
    ax3.set_xlim(.5, 3.5)
    ax3.set_xticks(x)
    ax3.set_ylim(-12, -8)
    ax3.set_yticklabels([])
    ax3.set_yticks([-12, -11, -10, -9, -8])
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(top='off', right='off')
        
    plt.annotate('length to receptor (bp)', xy=(0.5, 0.01),
                     xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='bottom',
                     fontsize=12)
    
def plotChevronPlot3(dGs, index=None, flow_id=None, dG_ref=None, cutoff=None):
    #cluster = 2
    #index = labels.loc[labels==cluster].index
    if index is None: index = dGs.index
    if flow_id is None: flow_id = 'wc10'
    
    # settings for plot
    jitter_scale = 0.05
    jitter_mean = 0
    num_points = len(dGs.loc[index])
    jitter = st.norm.rvs(jitter_mean, jitter_scale, num_points)
    
    # limits
    xlim1 = [7.5, 12.5]
    xlim2 = [-.5, 2.5]
    xlim3 = [.5, 3.5]
    if flow_id=="wc10":
        ylim1 = [-12, -6]
        ylim2 = [-12, -8]
    elif flow_id=="wc11":
        ylim1 = [-10, -6]
        ylim2 = [-9, -6]        
    fig = plt.figure(figsize=(5,3.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=(2,1), top=0.975, right=0.975, hspace=0.30, bottom=0.2, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[:, 0])
    x = [8, 9, 10, 11, 12]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['8_1', '9_1','10_2','11_2','12_3',]]].transpose()
    for i, ind in enumerate(y):
        ax1.scatter(x+jitter[i], y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax1.errorbar(x, y.mean(axis=1),  yerr=y.std(axis=1), alpha=0.5, fmt='-', elinewidth=2,
                 capsize=4, capthick=2, color='r', linewidth=2)
    ax1.set_xlim(xlim1)
    ax1.set_xticks(x)
    ax1.set_ylim(ylim1)
    ax1.set_yticks(np.arange(*ylim1))
    ax1.set_xlabel('length (bp)')
    if dG_ref is not None:
        ax1.plot(x,dG_ref.loc[['%s_%s'%(flow_id, i) for i in ['8_1', '9_1','10_2','11_2','12_3',]]],
                 color='0.5', alpha=0.5, linestyle='--', marker='.')
    if cutoff is not None:
        if ylim1[-1] > cutoff:
            ax1.fill_between(xlim1, [cutoff]*2, [ylim1[-1]]*2, alpha=0.5, color='0.5')
    
    ax2 = fig.add_subplot(gs[0, 1])
    x = [0, 1, 2]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['9_0', '9_1','9_2']]].transpose()
    for i, ind in enumerate(y):
        ax2.scatter(x+jitter[i], y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax2.errorbar(x, y.mean(axis=1), yerr=y.std(axis=1), alpha=0.5, fmt='-', elinewidth=2,
                 capsize=2, capthick=2, color='r', linewidth=2)
    if dG_ref is not None:
        ax2.plot(x, dG_ref.loc[['%s_%s'%(flow_id, i) for i in ['9_0', '9_1','9_2']]],
                 color='0.5', alpha=0.5, linestyle='--',marker='.')
    ax2.set_xlim(xlim2)
    ax2.set_xticks(x)
    ax2.set_ylim(ylim2)
    ax2.set_yticks(np.arange(*ylim2))
    ax2.annotate('9bp', xy=(0.5, 0.95),
                     xycoords='axes fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=12)
    if cutoff is not None:
        if ylim2[-1] > cutoff:
            ax2.fill_between(xlim2, [cutoff]*2, [ylim2[-1]]*2, alpha=0.5, color='0.5')

    ax3 = fig.add_subplot(gs[1,1])
    x = [1, 2, 3]
    y = dGs.loc[index, ['%s_%s'%(flow_id, i) for i in ['10_1','10_2','10_3']]].transpose()
    for i, ind in enumerate(y):
        ax3.scatter(x+jitter[i], y.loc[:, ind], facecolors='none', edgecolors='k', alpha=0.5, s=2)
    ax3.errorbar(x, y.mean(axis=1),  yerr=y.std(axis=1), alpha=0.5, fmt='-', elinewidth=2,
                 capsize=2, capthick=2, color='r', linewidth=2)
    if dG_ref is not None:
        ax3.plot(x, dG_ref.loc[['%s_%s'%(flow_id, i) for i in ['10_1','10_2','10_3']]],
                 color='0.5', alpha=0.5, linestyle='--', marker='.')
    ax3.set_xlim(xlim3)
    ax3.set_xticks(x)
    ax3.set_ylim(ylim2)
    ax3.set_yticks(np.arange(*ylim2))
    ax3.set_xlabel('length to receptor (bp)')
    ax3.annotate('10bp', xy=(0.5, 0.95),
                     xycoords='axes fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=12)
    if cutoff is not None:
        if ylim2[-1] > cutoff:
            ax3.fill_between(xlim3, [cutoff]*2, [ylim2[-1]]*2, alpha=0.5, color='0.5')
    
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(top='off', right='off')
    
    plt.annotate('$\Delta G$ (kcal/mol)', xy=(0.01, 0.6),
                     xycoords='figure fraction',
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=12, rotation=90)
    
    
def plotClusterStats(fractions, pvalue, counts=None, pvalue_threshold=None):
    if pvalue_threshold is None:
        pvalue_threshold = 0.05/15
        
    pvalue_adder = 1E-32
    
    # which junctions are significant?    
    junctions = pvalue.loc[pvalue < pvalue_threshold].index.tolist()
    x = np.arange(len(fractions))
    
    # pot fraction and pvalues
    fig = plt.figure(figsize=(4, 2.5))
    gs = gridspec.GridSpec(1, 2, bottom=0.35, left=0.25)

    x1 = x[np.vstack([(fractions.index==ss) for ss in junctions]).any(axis=0)]
    x2 = x[np.vstack([(fractions.index!=ss) for ss in junctions]).all(axis=0)]
    colors = pd.DataFrame(index=fractions.index, columns=np.arange(3))
    colors.iloc[x1] = sns.color_palette('Reds_d',  max(2*len(x2), 8))[-len(x1):][::-1]
    colors.iloc[x2] = sns.color_palette('Greys_d', max(2*len(x2), 8))[-len(x2):][::-1]
    #colors = (sns.color_palette('Greys_d', max(2*len(x2), 8))[-len(x2):][::-1] +
    #          sns.color_palette('Reds_d',  max(2*len(x2), 8))[-len(x1):][::-1])

    ax1 = fig.add_subplot(gs[0,0])
    pd.Series.plot(fractions, kind='barh', color=colors.values, width=0.8, ax=ax1)

    ax1.set_xlabel('fraction of \ntopology in cluster')
    xlim = ax1.get_xlim()
    xticks = np.linspace(xlim[0], xlim[1], 5)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([('%4.2f'%i)[1:] if i < 1 else ('%3.1f'%i) for i in xticks]) 
    
    y1 = -np.log10(pvalue.loc[fractions.iloc[x1].index])
    y2 = -np.log10(pvalue.loc[fractions.iloc[x2].index])
    
    ax2 = fig.add_subplot(gs[0,1])
    pd.Series.plot(-np.log10(pvalue.loc[fractions.index]+pvalue_adder),
                   kind='barh', color=colors.values, width=0.8, ax=ax2)
    ax2.set_yticklabels([])
    ax2.set_xlabel('-log10(pvalue)')
    
    for ax in [ax1, ax2]:
        ax.tick_params(top='off', right='off')
        
    # plot pie graph of cluster
    if counts is not None:

        plt.figure(figsize=(1.5, 2.5))
        bottom = 0
        for i, junction in enumerate(fractions.index):
            plt.bar([1], counts.loc[junction], bottom=bottom, color=colors.iloc[i],
                edgecolor='w', linewidth=1)
            bottom += counts.loc[junction]
        plt.xlim([0.8, 2])
        plt.xticks([])
        ax = plt.gca()
        ax.tick_params(top='off', right='off')
        
        plt.subplots_adjust(left=0.35, bottom=0.35)
        
def plotBarPlotdG_bind(results, data):
    fixed_inds = data.index[0][:3]
    loop = fixed_inds[0]
    topology = fixed_inds[2]
    lengths = data.index.levels[1]
    
    colors = ["#3498db", "#95a5a6","#e74c3c", "#34495e"]
    ylim = [5.5, 8.5]
    plt.figure(figsize=(3,3));
    x = lengths
    plt.bar(x,
            -results.loc[[returnBindingKey(loop, length, topology) for length in lengths]],
            yerr=results.loc[['%s_stde'%returnBindingKey(loop, length, topology) for length in lengths]],
            color=colors[3],
            error_kw={'ecolor': '0.3'})
    plt.ylim(ylim)
    plt.yticks(np.arange(ylim[0], ylim[1], 0.5), -np.arange(5.5, 10.5, 0.5))
    plt.ylabel('$\Delta G_{bind}$ (kcal/mol)')
    plt.xticks(lengths+0.4, lengths)
    plt.xlim(lengths[0]-0.2, lengths[-1]+1)
    plt.xlabel('positions')
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    plt.tight_layout() 

def plotBarPlotFrac(results, data, error=None):
    if error is None:
        error = True
    fixed_inds = data.index[0][:3]
    loop = fixed_inds[0]
    print loop
    topology = fixed_inds[-1]
    #to find seq order: 
    #binned_fractions = fractions.loc[:, [0,1,2]].copy()
    #for i in [0,1,2]: binned_fractions.loc[:, i] = np.digitize(fractions.loc[:, i].astype(float), bins=np.linspace(0, 1, 10))
    #seqs = binned_fractions.sort([0, 1, 2]).index.tolist()
    seqs = ['AUU','AGG','GCC', 'AAG', 'AUG', 'ACC',
            'AGU', 'ACU','AAC','UUC','AUC','UGG','AGC','GGC',
            'UCC','UCG','ACG','AAU','UGC','UUG']
    fractions = pd.DataFrame(index=seqs, columns=[0, 1, 2] + ['stde_%d'%i for i in [0,1,2]])
    for seq in seqs:
        for permute in [0,1,2]:
            if loop == 'GGAA_UUCG':
                fractions.loc[seq, permute] = results.loc[returnFracKey(topology, seq, permute)]
                fractions.loc[seq, 'stde_%d'%permute] = results.loc[returnFracKey(topology, seq, permute)+'_stde']
            else:
                fractions.loc[seq, permute] = results.loc[returnLoopFracKey(topology, seq, permute)]
                fractions.loc[seq, 'stde_%d'%permute] = results.loc[returnLoopFracKey(topology, seq, permute)+'_stde']                
    
    plt.figure(figsize=(4,3));
    colors = ["#3498db", "#95a5a6","#e74c3c", "#34495e"]
    x = np.arange(len(seqs))
    for permute in [0,1,2]:
        if permute==0:
            bottom = 0
        elif permute==1:
            bottom = fractions.loc[:, 0]
        else:
            bottom = fractions.loc[:, [0, 1]].sum(axis=1)
        if error:
            yerr = fractions.loc[:, 'stde_%d'%permute]
        else:
            yerr = None
        plt.bar(x,
                fractions.loc[:, permute],
                bottom=bottom,
                yerr=yerr,
                color=colors[permute],
                error_kw={'ecolor': '0.3'})
    plt.xticks(x+0.5, fractions.index, rotation=90)
    #plt.ylim(0, 1.2)
    #plt.yticks(np.arange(0, 1.2, 0.2))
    plt.ylabel('fraction')
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    plt.tight_layout()
    
def plotScatterplotLoopChange(data):
    colors = {3:"#3498db", 4:"#95a5a6", 5:"#e74c3c", 6:"#34495e"}
    lengths = data.index.levels[1]
    plt.figure(figsize=(3,3));
    for length in lengths:
        key = 'length'
        index_length = (data.reset_index(level=key).loc[:, key]==length).values
        plt.scatter(data.not_fit.loc[index_length],
                    data.dG_conf_loop.loc[index_length],
                    c=colors[length]);
        r, pvalue = st.pearsonr(data.loc[index_length].dropna(subset=['not_fit', 'dG_conf_loop']).not_fit,
                                data.loc[index_length].dropna(subset=['not_fit', 'dG_conf_loop']).dG_conf_loop)
        print length, r**2
    
    plt.xlabel('$\Delta \Delta G$ (kcal/mol)'); plt.ylabel('$\Delta \Delta G$ predicted (kcal/mol)'); 
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    plt.tight_layout()
    
def plotScatterplotTestSet(data, leave_out_lengths):
    # set plotting parameters
    xlim = np.array([-11, -6.5])
    ylim = np.array([-3, 0])
    colors = {3:"#3498db", 4:"#95a5a6", 5:"#e74c3c", 6:"#34495e"}
    
    index = data.loc[data.isnull().dG_bind].dropna(subset=['fit', 'dG_conf']).index
    plt.figure(figsize=(3,3));
    annotateString = ''
    key = 'length'
    for length in leave_out_lengths:
        index_length = (data.reset_index(level=key).loc[:, key]==length).values
        plt.scatter(data.loc[index_length].dG,
                    data.loc[index_length].dG_conf,
                    c=colors[length]);
        r, pvalue = st.pearsonr(data.loc[index_length].dropna(subset=['dG', 'dG_conf']).dG,
                                data.loc[index_length].dropna(subset=['dG', 'dG_conf']).dG_conf)
        annotateString = '$R^2$=%4.2f for length=%d'%(r**2, length)
    plt.xlim(xlim); plt.ylim(ylim); plt.xticks(np.arange(*xlim))
    plt.xlabel('$\Delta G$ (kcal/mol)'); plt.ylabel('$\Delta G$ predicated (kcal/mol)'); 
    plt.annotate(annotateString,
                     xy=(0.95, 0.05),
                     xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='bottom',
                     fontsize=12)
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    plt.tight_layout()
    
def plotScatterPlotTrainingSet(data):

    # set plotting parameters
    xlim = np.array([-11, -6])
    ylim = np.array([-2.5, 0])
    colors = {3:"#3498db", 4:"#95a5a6", 5:"#e74c3c", 6:"#34495e"}
    
    # plot the training set
    index = data.dropna(subset=['dG', 'dG_fit']).index
    r, pvalue = st.pearsonr(data.loc[index].dG, data.dG_fit.loc[index])
    plt.figure(figsize=(3,3));
    key = 'length'
    other_lengths = np.unique(data.reset_index(level=key).loc[:, key])
    for length in other_lengths:
        index_length = (data.reset_index(level=key).loc[:, key]==length).values
        plt.scatter(data.loc[index_length].dG,
                    data.loc[index_length].dG_fit,
                    c=colors[length]);
    plt.xlim(xlim); plt.ylim(xlim); plt.xticks(np.arange(*xlim))
    plt.xlabel('$\Delta G$ (kcal/mol)'); plt.ylabel('$\Delta G$ predicated (kcal/mol)'); 
    plt.annotate('$R^2$ = %4.2f\nlengths=%s'%(r**2, ','.join(other_lengths.astype(int).astype(str))),
                     xy=(0.95, 0.05),
                     xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='bottom',
                     fontsize=12)
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    plt.tight_layout()

def plotScatterPlotNearestNeighbor(data):
    # plot the nn versus dG conf
    colors = {3:"#3498db", 4:"#95a5a6", 5:"#e74c3c", 6:"#34495e"}
    plt.figure(figsize=(3,3));
    
    plt.scatter(data.dG_conf,
                data.nn_pred,
                c=colors[6]);
    index = data.dropna(subset=['dG_conf',]).index
    r, pvalue = st.pearsonr((data.dG_conf).loc[index], data.loc[index].nn_pred)
    plt.annotate('$R^2$ = %4.2f'%r**2, 
                     xy=(0.05, 0.95),
                     xycoords='axes fraction',
                     horizontalalignment='left', verticalalignment='top',
                     fontsize=12)
    plt.xlabel('$\Delta G_{conf}$ (kcal/mol)');
    plt.ylabel('$\Delta G$ nearest neighbor (kcal/mol)'); 
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    xlim = ax.get_xlim()
    plt.xticks(np.arange(*xlim))
    plt.tight_layout()

def plotAllClusterGrams(mat, mean_vector=None, n_clusters=None,
                        fillna=None, distance=None, mask=None, n_components=None,
                        heatmap_scale=None, pc_heatmap_scale=None, plotClusters=False, plotMM=False, plotFlank=False):
    if n_clusters is None:
        n_clusters = 10
    if fillna is None:
        fillna = False
    if mask is None:
        mask = False
    if n_components is None:
        n_components = 6
    if heatmap_scale is None:
        heatmap_scale = 2.5
    if pc_heatmap_scale is None:
        pc_heatmap_scale = 3
        
    mat = mat.astype(float)
    if fillna:
        masked = mat.isnull()
        submat = mat.copy()
        for col in mat:
            index =  mat.loc[:, col].isnull()
            submat.loc[index, col] = mat.loc[:, col].mean()
    else:
        submat = mat.dropna().copy()
        masked = pd.DataFrame(index=submat.index, columns=submat.columns, data=0).astype(bool)
    
    toplot = submat.copy()
    if mask:
        toplot[masked] = np.nan
        
    seqMat = processStringMat(submat.index.tolist()).fillna(4)
    structMat = processStructureMat(submat)
    labelMat = getLabels(submat.index)
    
    
    # PCA data
    sklearn_pca = sklearnPCA(n_components=n_components)
    sklearn_transf = sklearn_pca.fit_transform(submat)
    
    # cluster PCA
    z = sch.linkage(sklearn_transf, method='average', metric='euclidean')
    order = sch.leaves_list(z)
    if distance is None:
        clusters = pd.Series(sch.fcluster(z, t=n_clusters, criterion='maxclust'), index=submat.index)
    else:
        clusters = pd.Series(sch.fcluster(z, t=distance, criterion='distance'), index=submat.index)
    if mean_vector is None:
        mean_vector = submat.mean()

    # plot
    with sns.axes_style('white', {'lines.linewidth': 0.5}):
        fig = plt.figure(figsize=(7,8))
        gs = gridspec.GridSpec(4, 7, width_ratios=[0.5, 0.75, 4, 1, 0.75, 0.2, 0.2], height_ratios=[1,6,1,1],
                               left=0.01, right=0.99, bottom=0.15, top=0.95,
                               hspace=0.05, wspace=0.05)
        
        dendrogramAx = fig.add_subplot(gs[1,0])
        sch.dendrogram(z, orientation='right', ax=dendrogramAx, no_labels=True, link_color_func=lambda k: 'k')
        dendrogramAx.set_xticks([])
        
        pcaAx = fig.add_subplot(gs[1,1])
        pcaAx.imshow(sklearn_transf[order], cmap='RdBu_r', aspect='auto', interpolation='nearest', origin='lower',
                     vmin=-pc_heatmap_scale, vmax=pc_heatmap_scale)
        pcaAx.set_yticks([])
        pcaAx.set_xticks([])

        heatmapAx = fig.add_subplot(gs[1,2])
        heatmapAx.imshow(toplot.iloc[order] - mean_vector.loc[toplot.columns], cmap='coolwarm', aspect='auto',
                     interpolation='nearest', origin='lower', vmin=-heatmap_scale, vmax=heatmap_scale)
        heatmapAx.set_yticks([])
        heatmapAx.set_xticks([])
        #heatmapAx.set_xticks(np.arange(toplot.shape[1]))
        #heatmapAx.set_xticklabels(toplot.columns.tolist(), rotation=90)
        
        # plot mean plot
        meanPlotAx = fig.add_subplot(gs[0,2])
        meanPlotAx.plot(np.arange(submat.shape[1])+0.5,
                        mean_vector.loc[submat.columns], 's', markersize=3, color=sns.xkcd_rgb['dark red'])
        plotMeanPlot(submat, ax=meanPlotAx)
        
        # plot contexts
        contextAx = fig.add_subplot(gs[3,2])
        if plotFlank:
            plotLengthMatFlank(contextAx, labels=toplot.columns.tolist())
        else:
            plotLengthMat(contextAx, labels=toplot.columns.tolist())
        
        # plot sequenceseqMat
        seqAx = fig.add_subplot(gs[1,3])
        #seqCbarAx = fig.add_subplot(gs[0,3])
        plotSeqMat(seqMat.iloc[order], heatmapAx=seqAx, cbarAx=None)
        
        # plot structure
        structAx = fig.add_subplot(gs[1,4])
        #structCbarAx = fig.add_subplot(gs[0,4])
        plotStructMat(structMat.iloc[order, :-1], heatmapAx=structAx)

        # plot bulge
        bulgeAx = fig.add_subplot(gs[1,5])
        #bulgeCbarAx = fig.add_subplot(gs[0,5])
        plotBulgeMat(structMat.iloc[order].loc[:, ['bulge']], heatmapAx=bulgeAx, cbarAx=None)
        
        # plot labels
        if plotClusters:
            ## plot clusters
            clusterAx = fig.add_subplot(gs[1, 6])
            clusterCbarAx = fig.add_subplot(gs[0, 6])
            plotClusterMat(clusters.iloc[order], clusterAx=clusterAx, clusterCbarAx=clusterCbarAx,
                           maxNumClusters=len(np.unique(clusters)))
        else:
            clusterAx = fig.add_subplot(gs[1, 6])
            #clusterCbarAx = fig.add_subplot(gs[0, 6])
            plotClusterMat(labelMat.iloc[order], clusterAx=clusterAx, clusterCbarAx=None)
            
        if plotMM:
            mmMat = processMMMat(submat)
            clusterAx = fig.add_subplot(gs[1, 6])
            #clusterCbarAx = fig.add_subplot(gs[0, 6])
            plotMMMat(mmMat.iloc[order], heatmapAx=clusterAx, cbarAx=None)    

        # plot pcs
        pcAx = fig.add_subplot(gs[2, 2])
        plotContextPCAs(submat, sklearn_transf=sklearn_transf, n_components=n_components, ax=pcAx)

        varAx = fig.add_subplot(gs[2, 3])
        varAx.barh(np.arange(n_components)[::-1], sklearn_pca.explained_variance_ratio_)
        varAx.set_yticks([])
        varAx.set_xlim([0, 0.5])
        for y in [0.1, 0.2, 0.3, 0.4]:
            varAx.axvline(y, color='k', alpha=0.5, linestyle=':', linewidth=0.5)
        varAx.set_xticks([])


    return submat, order, clusters

def plotMeanPlot(submat, ax=None, c=None, ecolor=None, offset=None, plot_line=None):
    if c is None:
        c = '0.5'
    if ecolor is None:
        ecolor='k'
    if offset is None:
        offset=0
    if ax is None:
        fig = plt.figure(figsize=(4,2));
        ax = fig.add_subplot(111)
    if plot_line is None:
        plot_line = False
    if plot_line:
        fmt = '.-'
    else:
        fmt = '.'

    ax.errorbar(np.arange(submat.shape[1])+0.5+offset,
                        submat.mean(),
                        yerr=submat.std(),
                        fmt=fmt, capsize=0, capthick=0, ecolor=ecolor, linewidth=1,
                        color=c)
    ax.set_ylim([-12, -7])
    ax.set_xlim(0, submat.shape[1])
    ax.set_xticks(np.arange(submat.shape[1])+0.5)
    ax.set_xticklabels([])
    for y in np.arange(-12, -7):
        ax.axhline(y, color='k', alpha=0.5, linestyle=':', linewidth=0.5)
    ax.tick_params(top='off', right='off')
    return ax

def plotSeqMat(seqMat, heatmapAx=None, cbarAx=None):
    # plot seq
    # cmap is linear
    cmap = mpl.colors.ListedColormap(['#921927', '#F1A07E', '#FFFFFF', '#989798', '#000000'])
    heatmapAx.imshow(seqMat.astype(float), cmap=cmap, vmin=0, vmax=4, aspect='auto',
                 interpolation='nearest', origin='lower')
    heatmapAx.set_yticks([])
    heatmapAx.set_xticks([])
    
    # plot cbar
    if cbarAx is not None:
        cbarAx.imshow(np.vstack(np.arange(4)).T, cmap=cmap, vmin=0, vmax=4,
                         interpolation='nearest', origin='lower');
        cbarAx.set_xticks(np.arange(4))
        cbarAx.set_xticklabels(['A', 'G', 'C', 'U'])
        cbarAx.set_yticks([])
    
def plotStructMat(structMat, heatmapAx=None, cbarAx=None):
    cmap = mpl.colors.ListedColormap(['#FFFFFF', '#FFF100', '#00A79D', '#006738', '#2E3092'])
    # plot structure
    heatmapAx.imshow(structMat.astype(float), cmap=cmap, aspect='auto',
                 interpolation='nearest', origin='lower', vmin=0, vmax=4)
    heatmapAx.set_yticks([])
    #structAx.set_xticks(np.arange(structMat.shape[1]))
    #structAx.set_xticklabels(structMat.columns.tolist()[:-1], rotation=90)
    heatmapAx.set_xticks([])
    
    # plot structure
    if cbarAx is not None:
        cbarAx.imshow(np.vstack([0, 1, 2, 3, 4]), cmap=cmap,
                     interpolation='nearest', origin='lower', vmin=0, vmax=4)
        cbarAx.set_xticks([])
        cbarAx.set_yticks(np.array([0, 1, 2, 3, 4])+0.5)
        cbarAx.set_yticklabels(['WC', 'GU', 'other', 'Py-Py', 'Pu-Pu', ])
        
def plotMMMat(mmMat, heatmapAx=None, cbarAx=None):

        
    cmap = mpl.colors.ListedColormap(['#FFFFFF', '#FFF100', '#00A79D', '#006738', '#2E3092'])
    # plot structure
    heatmapAx.imshow(mmMat.astype(float), cmap=cmap, aspect='auto',
                 interpolation='nearest', origin='lower', vmin=0, vmax=4)
    heatmapAx.set_yticks([])
    #structAx.set_xticks(np.arange(structMat.shape[1]))
    #structAx.set_xticklabels(structMat.columns.tolist()[:-1], rotation=90)
    heatmapAx.set_xticks([])
    
    # plot structure
    if cbarAx is not None:
        cbarAx.imshow(np.vstack([0, 1, 2, 3, 4]), cmap=cmap,
                     interpolation='nearest', origin='lower', vmin=0, vmax=4)
        cbarAx.set_xticks([])
        cbarAx.set_yticks(np.array([0, 1, 2, 3, 4])+0.5)
        cbarAx.set_yticklabels(['WC', 'GU', 'other', 'Py-Py', 'Pu-Pu', ])

def plotBulgeMat(bulges, heatmapAx=None, cbarAx=None):
    cmap = mpl.colors.ListedColormap(['#EF4036', '#F05A28', '#F7931D', '#F4F5F6', '#27A9E1', '#1B75BB', '#2E3092'][::-1])
    # plot bulge
    heatmapAx.imshow(bulges.astype(float), cmap=cmap, aspect='auto', vmin=-3, vmax=3,
                 interpolation='nearest', origin='lower')
    heatmapAx.set_yticks([])
    heatmapAx.set_xticks([1])
    heatmapAx.set_xticklabels(['bulge'], rotation=90)
    
    if cbarAx is not None:
        cbarAx.imshow(np.vstack(np.arange(-3, 4)), cmap=cmap, vmin=-3, vmax=3,
                     interpolation='nearest', origin='lower')
        cbarAx.set_xticks([])
        cbarAx.set_yticks([])

def plotClusterMat(clusters, clusterAx=None, clusterCbarAx=None, maxNumClusters=None):
    if maxNumClusters is None:
        maxNumClusters = 9
    # plot clusters
    clusterAx.imshow(np.vstack(clusters), cmap='Paired',
                     interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=maxNumClusters);
    clusterAx.set_yticks([])
    clusterAx.set_xticks([])
    
    if clusterCbarAx is not None:
        clusterCbarAx.imshow(np.vstack(np.unique(clusters.dropna())), cmap='Paired',
                         interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=maxNumClusters);
        clusterCbarAx.set_yticks(np.arange(len(np.unique(clusters.dropna()))))
        clusterCbarAx.set_yticklabels(np.unique(clusters))
        clusterCbarAx.yaxis.tick_right()
        clusterCbarAx.set_xticks([])
    
def plotLengthMat(ax, labels=None):
    mat = [[9]*9+[10]*9+[11]*9, ([8]+[9]*3+[10]*3+[11]+[12])*3, ([0]+[-1,0,1]+[-1,0,1]+[0]+[0])*3]  
    
    cmap_range = pd.Series('w', index=np.arange(-1, 13))
    cmap_range.loc[[-1, 0, 1]] = [[1-i]*3 for i in [0.05, 0.4, 0.8]]
    cmap_range.loc[[8,9,10,11,12]] = [[1-i]*3 for i in [0.05,0.25,.5, 0.8,1]]
    cmap = mpl.colors.ListedColormap(cmap_range.tolist())
    
    ax.imshow(mat, interpolation='nearest',  aspect='auto', cmap=cmap, vmin=-1, vmax=12)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(['flow length (bp):', 'chip length (bp):', 'motif offset:'])
    ax.set_xticks(np.arange(len(mat[0])))
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90)

def plotLengthMatFlank(ax, labels=None):
    mat = [[9]*2+[10]*2+[11]*2, [9, 10]*3]
    
    cmap_range = pd.Series('w', index=np.arange(-1, 13))
    cmap_range.loc[[-1, 0, 1]] = [[1-i]*3 for i in [0.05, 0.4, 0.8]]
    cmap_range.loc[[8,9,10,11,12]] = [[1-i]*3 for i in [0.05,0.25,.5, 0.8,1]]
    cmap = mpl.colors.ListedColormap(cmap_range.tolist())
    
    ax.imshow(mat, interpolation='nearest', aspect='auto', cmap=cmap, vmin=-1, vmax=12)
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(['flow length (bp):', 'chip length (bp):'])
    ax.set_xticks(np.arange(len(mat[0])))
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90)    

def processStringMat(seqs):
    
    stringMat = pd.concat(list(pd.Series(seqs).str), axis=1)
    
    # find those columsn that aren't unique
    cols_to_keep = stringMat.columns
    #for col in stringMat:
    #    if len(stringMat.loc[:, col].value_counts()) > 1:
    #        cols_to_keep.append(col)
            
    seqMat = pd.DataFrame(index=stringMat.index, columns=cols_to_keep)
    for i, base in enumerate(['A', 'G', 'C', 'U', '_']):
        for col in cols_to_keep:
            index = stringMat.loc[:, col] == base
            seqMat.loc[index, col] = i
    seqMat.index = seqs
    return seqMat

def makeLogo(mat):
    stringMat = pd.concat(list(pd.Series(mat.index.tolist()).str), axis=1)
    
    fractionEach = pd.DataFrame(columns=stringMat.columns, index=['A', 'G', 'C', 'U' ])
    
    for col in stringMat:
        fractionEach.loc[:, col] = stringMat.loc[:, col].value_counts()/float(len(stringMat))

    #cols_to_keep = fractionEach.max() != 1
    cols_to_keep = fractionEach.columns
    colors = {'A': sns.xkcd_rgb['dark red'], 'G':'w', 'C':sns.xkcd_rgb['salmon'], 'U':'0.5'}
    colors = [sns.xkcd_rgb['dark red'], 'w', sns.xkcd_rgb['salmon'], '0.5']
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    fractionEach.transpose().loc[cols_to_keep].plot(kind='bar', stacked=True, colors=colors,
                                                    ax=ax, legend=False)
    ax.tick_params(top='off', right='off')
    plt.ylabel('fraction')
    plt.tight_layout()
    
def processStructureMat(mat):
    columns = processStructureString(mat.index[0]).index
    structMat = pd.DataFrame(index=mat.index, columns=columns)
    for index in structMat.index:
        structMat.loc[index] = processStructureString(index)
    return structMat

def processStructureString(string):
    stringVec = pd.Series(list(string))
    
    # inside, then outside
    separator = (stringVec == '_').idxmax()
    
    # num locs
    side1 = len(stringVec.loc[:separator])-1
    side2 = len(stringVec.loc[separator:])-1
    
    actual_length = min(side1, side2)
    max_num = max(side1, side2)
    first = stringVec.index[0]
    last = stringVec.index[-1]
    
    ind1 = [val for pair in zip(np.arange(first, separator),
                                np.arange(first+1, separator)[::-1])
            for val in pair][:actual_length]
    
    ind2 = [val for pair in zip(np.arange(last, separator, -1),
                                np.arange(last-1, separator, -1)[::-1])
            for val in pair][:actual_length]

    order = [val for pair in zip(np.arange(first, actual_length),
                                np.arange(first+1, actual_length)[::-1])
             for val in pair][:actual_length]
    
    annotations = pd.Series(index=np.arange(actual_length))
    matchDict = {'A':'U', 'U':'A', 'C':'G', 'G':'C'}
    pupuDict = {'A':set(['A','G']), 'G':set(['A', 'G']), 'C':set(), 'U':set()}
    pypyDict = {'C':set(['U','C']), 'U':set(['U', 'C']), 'G':set(), 'A':set()}
    guDict = {'G':'U', 'U':'G', 'A':None, 'C':None}
    for i, j, loc in itertools.izip(ind1, ind2, order):
        if stringVec.loc[i] == matchDict[stringVec.loc[j]]:
            annotations.loc[loc] = 0
        elif stringVec.loc[i] in pupuDict[stringVec.loc[j]]:
            annotations.loc[loc] = 4          
        elif stringVec.loc[i] in pypyDict[stringVec.loc[j]]:
            annotations.loc[loc] = 3
        elif stringVec.loc[i] == guDict[stringVec.loc[j]]:
            annotations.loc[loc] = 1
        else:
            annotations.loc[loc] = 2
    
    
    # bulge is positive if on side1, negative on other
    annotations.loc['bulge'] =  side1-side2
    return annotations

def processMMMat(mat):
    mms = ['AA', 'AG', 'GA', 'GG', 'AC', 'CA', 'UG', 'GU', 'UU', 'UC', 'CU', 'CC']
    structInds = [4]*4 + [2]*2 + [1]*2 + [3]*4
    positions = [[1, -2], [2, -3]]
    
    mmMat = pd.DataFrame(0, index=mat.index, columns=mms)
    for i, mm in enumerate(mms):
        c = np.array([(s[positions[0][0]]==mm[0] and s[positions[0][1]]==mm[1]) or
                      (s[positions[1][0]]==mm[0] and s[positions[1][1]]==mm[1])
                      for s in mat.index])
        mmMat.loc[c, mm] = structInds[i]
    return mmMat

def getLabels(junctionSeqs):
    labelMat = pd.Series(index=junctionSeqs)
    
    motifSets = {'WC':['W'],
                '1x0 or 0x1': ['B1', 'B2'],
                '1x1':        ['M,W', 'W,M'],
                '2x0 or 0x2': ['B1,B1','B2,B2'],
                '3x0 or 0x3': ['B1,B1,B1', 'B2,B2,B2'],
                '2x2':        ['M,M'],
                '3x3':        ['W,M,M,M', 'M,M,M,W'],
                '2x1 or 1x2': ['W,B1,M','M,B1,W','W,B2,M','M,B2,W'],
                '3x1 or 1x3': ['W,B1,B1,M','M,B1,B1,W','W,B2,B2,M','M,B2,B2,W']}
    count = 0
    for name, motifs in motifSets.items():
        index = getJunctionSeqs(motifs, junctionSeqs)
        labelMat.loc[index] = count
        count+=1
    return labelMat

def getJunctionSeqs(motifs, junctionSeqs, add_flank=None):
    if add_flank is None:
        add_flank = False
    associatedMotifs = {'W':('W', 'W', 'W', 'W'),
        'B1':('W', 'W', 'B1', 'W', 'W'),
                        'B2':('W', 'W', 'B2', 'W', 'W',),
                        'B1,B1':('W', 'W', 'B1', 'B1', 'W', 'W',),
                        'B2,B2':('W', 'W', 'B2', 'B2', 'W', 'W',),
                        'B1,B1,B1':('W', 'W', 'B1', 'B1', 'B1','W', 'W',),
                        'B2,B2,B2':('W', 'W', 'B2', 'B2', 'B2','W', 'W',),
                        'W,B1,M':('W', 'W', 'B1', 'M', 'W'),
                        'M,B1,W':('W', 'M', 'B1', 'W', 'W'),
                        'W,B2,M':('W', 'W', 'B2', 'M', 'W'),
                        'M,B2,W':('W', 'M', 'B2', 'W', 'W'),
                        'W,B1,B1,M':('W', 'W', 'B1', 'B1', 'M', 'W'),
                        'M,B1,B1,W':('W', 'M', 'B1', 'B1', 'W', 'W'),
                        'W,B2,B2,M':('W', 'W', 'B2', 'B2', 'M', 'W'),
                        'M,B2,B2,W':('W', 'M', 'B2', 'B2', 'W', 'W'),
                        'M,W':('W', 'M', 'W', 'W'),
                        'W,M':('W', 'W', 'M', 'W'),
                        'M,M':('W', 'M', 'M', 'W'),
                        'W,M,M,M':('W', 'M', 'M', 'M'),
                        'M,M,M,W':('M', 'M', 'M', 'W')}
    a = []
    for motif in motifs:
        seqs = hjh.junction.Junction(associatedMotifs[motif]).sequences
        # with U,A flanker
        if add_flank:
            newseqs = 'U' + seqs.side1 + 'U_A' + seqs.side2 + 'A'
        else:
            newseqs =  seqs.side1 + '_' + seqs.side2
        a.append(newseqs)
    
    a = pd.concat(a)
    return pd.Series(np.in1d(junctionSeqs, a), index=junctionSeqs)

def plotContextPCAs(submat, sklearn_transf=None, n_components=None, ax=None):
    
    if sklearn_transf is None:
        sklearn_pca = sklearnPCA(n_components=None)
        sklearn_transf = sklearn_pca.fit_transform(submat)
    
    pca_projections = np.dot(np.linalg.inv(np.dot(sklearn_transf.T, sklearn_transf)),
                                 np.dot(sklearn_transf.T, submat))
    
    if ax is None:
        newplot=True
        with sns.axes_style('white'):
            fig = plt.figure(figsize=(5,2.5))
            ax = fig.add_subplot(111)
    else:
        newplot=False

    im = ax.imshow(pca_projections[:n_components],
               interpolation='nearest', aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    #ax.set_ylim(n_components-1, -1)
    ax.set_yticks(np.arange(n_components)),
    ax.set_yticklabels(['PC %d'%i
                for i in np.arange(n_components)])
    ax.set_xticks([])
    
    if newplot:
        ax.set_xticks(np.arange(submat.shape[1]))
        ax.set_xticklabels(submat.columns.tolist(), rotation=90)
        plt.tight_layout()

def doPCA(submat, whiten=None):
    sklearn_pca = sklearnPCA(n_components=None, whiten=whiten)
    sklearn_transf = sklearn_pca.fit_transform(submat)
    pca_projections = np.dot(np.linalg.inv(np.dot(sklearn_transf.T, sklearn_transf)),
                                 np.dot(sklearn_transf.T, submat))
    
    
    return (sklearn_pca, pd.DataFrame(sklearn_transf, index=submat.index),
            pd.DataFrame(pca_projections, columns=submat.columns))

def plotProjections(projections, pca):
    for i in range(6):
        plt.figure(figsize=(4,3));
        plt.bar(np.arange(6), projections.loc[i], facecolor='0.5', edgecolor='w');
        plt.xticks(np.arange(6)+0.5, projections.columns.tolist(), rotation=90);
        plt.ylabel('pc projection')
        plt.annotate('%4.1f%%'%(100*pca.explained_variance_ratio_[i]),
                     xy=(.05, .05), xycoords='axes fraction',
                     horizontalalignment='left', verticalalignment='bottom')
        plt.tight_layout()


def plotcontour(xx, yy, z, cdf=None, fill=False, fraction_of_max=None, levels=None,
                label=False, ax=None, linecolor='k', fillcolor='0.5', plot_line=True):
    x, y = seqfun.getCDF(np.ravel(z))

    if levels is None:
        if cdf is not None:
            # what x is closest to y = cdf
            levels = [x[np.abs(y-cdf).argmin()]]
        if fraction_of_max is not None:
            # what x is closest to fraction_of_max*max(x)
            if isinstance(fraction_of_max, list):
                levels= [i*x.max() for i in fraction_of_max]
            else:
                levels = [fraction_of_max*x.max()]
        
    if ax is None:
        fig = plt.figure();
        ax = fig.add_subplot(111)
        
    if fill:
        cs = ax.contourf(xx,yy,z, levels=[levels[0], z.max()], colors=fillcolor, alpha=0.5)
    if plot_line:
        cs2 = ax.contour(xx,yy,z, levels=levels, colors=linecolor, linewidth=1)
    if label:
        ax.clabel(cs2, fmt='%2.2f', colors='k', fontsize=11)
    
    


        

def statsmodels_bivariate_kde(x, y, bw='scott', gridsize=100, cut=3, clip=[[-np.inf, np.inf],[-np.inf, np.inf]]):
    """Compute a bivariate kde using statsmodels."""
    if isinstance(bw, str):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, clip[0])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z

def _kde_support(data, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    if clip[0] != -np.inf:
        support_min = clip[0]
    else:
        support_min = max(data.min() - bw * cut, clip[0])
    if clip[1] != np.inf:
        support_max = clip[1]
    else:
        support_max = min(data.max() + bw * cut, clip[1])
    return np.linspace(support_min, support_max, gridsize)
