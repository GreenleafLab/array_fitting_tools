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
        
def plotBarPlotdG_bind(results, lengths):
    colors = ["#3498db", "#95a5a6","#e74c3c", "#34495e"]
    ylim = [5.5, 8.5]
    plt.figure(figsize=(3,3));
    x = lengths
    plt.bar(x,
            -results.loc[['bind_%d'%length for length in lengths]],
            yerr=results.loc[['bind_%d_stde'%length for length in lengths]],
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

def plotBarPlotFrac(results, data):
    lengths = data.index.levels[0]
    seqs =  ['AUC', 'AUG', 'AUU', 'GCC', 'UUC', 'UGC', 'UCC', 'AGU', 'UGG', 'AAG',
             'AAC', 'ACC', 'AGC', 'AGG', 'UUG', 'ACG', 'UCG', 'ACU', 'GGC', 'AAU']
    fractions = pd.DataFrame(index=seqs, columns=[0, 1, 2] + ['stde_%d'%i for i in [0,1,2]])
    for seq in seqs:
        circPermutedSeqs = data.loc[lengths[0],seq].seq
        for permute in [0,1,2]:
            fractions.loc[seq, permute] = results.loc[circPermutedSeqs.loc[permute]]
            fractions.loc[seq, 'stde_%d'%permute] = results.loc[circPermutedSeqs.loc[permute]+'_stde']
            
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
            
        plt.bar(x,
                fractions.loc[:, permute],
                bottom=bottom,
                yerr=fractions.loc[:, 'stde_%d'%permute],
                color=colors[permute],
                error_kw={'ecolor': '0.3'})
    plt.xticks(x+0.5, fractions.index, rotation=90)
    plt.ylabel('fraction')
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    plt.tight_layout()
    
def plotScatterplotLoopChange(data):
    colors = {3:"#3498db", 4:"#95a5a6", 5:"#e74c3c", 6:"#34495e"}
    lengths = data.index.levels[0]
    plt.figure(figsize=(3,3));
    for length in lengths:
        plt.scatter((data.fit - data.not_fit).loc[length],
                    data.ddG_pred.loc[length],
                    c=colors[length]);
        index = data.dropna(subset=['fit', 'not_fit', 'ddG_pred']).index
        r, pvalue = st.pearsonr((data.fit-data.not_fit).loc[index].loc[length], data.loc[index].loc[length].ddG_pred)
        print length, r**2
    
    plt.xlabel('$\Delta \Delta G$ (kcal/mol)'); plt.ylabel('$\Delta \Delta G$ predicted (kcal/mol)'); 
    ax = plt.gca(); ax.tick_params(top='off', right='off')
    plt.tight_layout()
    
def plotScatterplotTestSet(data, leave_out_lengths):
    # set plotting parameters
    xlim = np.array([-11, -6.5])
    ylim = np.array([-2.5, 0])
    colors = {3:"#3498db", 4:"#95a5a6", 5:"#e74c3c", 6:"#34495e"}
    
    index = data.loc[data.isnull().dG_bind].dropna(subset=['fit', 'dG_conf']).index
    plt.figure(figsize=(3,3));
    annotateString = ''
    for length in leave_out_lengths:
        plt.scatter(data.loc[length].fit,
                    data.loc[length].dG_conf,
                    c=colors[length]);
        r, pvalue = st.pearsonr(data.loc[index].loc[length].fit, data.loc[index].loc[length].dG_conf)
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
    
def plotScatterPlotTrainingSet(data, other_lengths):
    # set plotting parameters
    xlim = np.array([-11, -6])
    ylim = np.array([-2.5, 0])
    colors = {3:"#3498db", 4:"#95a5a6", 5:"#e74c3c", 6:"#34495e"}
    
    # plot the training set
    index = data.dropna(subset=['fit', 'dG_conf', 'dG_bind']).index
    r, pvalue = st.pearsonr(data.loc[index].fit, (data.dG_conf + data.dG_bind).loc[index])
    plt.figure(figsize=(3,3));
    for length in other_lengths:
        plt.scatter(data.loc[length].fit,
                    (data.dG_conf + data.dG_bind).loc[length],
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

