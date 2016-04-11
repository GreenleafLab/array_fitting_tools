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
import scipy.spatial.distance as ssd
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1, 'axes.linewidth':1, 'text.color': 'k', 'axes.labelcolor': 'k'})
from plotFun import fix_axes
import fitFun
import seqfun
import hjh.junction
import clusterFun
from tectoThreeWay import returnFracKey, returnBindingKey, returnLoopFracKey

class tectoData():
    """ Class to store data from tecto experiment and plot tecto-specific plots.
    
    Parameters:
    -----------
    results : output table that has potentially more than one replicate defined data
    libChar : table that describes parameters of library.
    """
    def __init__(self, results, libChar):
        self.results = results
        self.libChar = libChar
        
    def getPivotedTable(self, param=None):
        if param is None: param='dG'
        subresults = pd.concat([self.libChar, self.results], axis=1).loc[self.libChar.sublibrary=='junction_conformations']
        subresults.loc[:, 'id'] = (
                           subresults.length.astype(int).astype(str) + '_' +
                           subresults.helix_one_length.astype(int).astype(str))
        #subresults.reset_index(inplace=True)
        id_order = ['8_1', '9_0', '9_1', '9_2', '10_1', '10_2', '10_3', '11_2', '12_3']
        pivot = subresults.pivot(index='junction_seq', columns='id', values=param).loc[:, id_order]

        new_index = []
        for s in pivot.index.tolist():
            side1, side2 = s.split('_')
            new_index.append('_'.join([side1[1:-1], side2[1:-1]]))
        pivot.index = new_index
        return pivot
    

    def plotChevronPlot(self, variants=None):
        if variants is None:
            variants = [627, 4635, 8643, 9979, 11315]
        x = self.libChar.loc[variants, 'effective_length']
        y = self.results.loc[variants, 'dG']
        yerr = [self.results.loc[variants, 'eminus'],
                self.results.loc[variants, 'eplus']]
        
        # change variants that are above threshold
        plotErrorbarLength(x, y, yerr)
        

        
    def plotMotif(self, motif, flank, c=None, linestyle=None, plot_all=None):
        motif_tuple,  n = getMotif(motif) 
        return self._plotMotif(motif_tuple,  n, flank, c=c, linestyle=linestyle, plot_all=plot_all)    
    
    def _plotMotif(self, motifs, n, flank, c=None, linestyle=None, plot_all=False):
        junction_seqs = getJunctionsSeqsFlank(motifs, n, flank)
        dGmat = self.getPivotedTable().loc[junction_seqs]
        if len(dGmat)==0:
            print 'Error: no junction seqs are present in table. Is flank right?'
            sys.exit()
            
        # only do a subset of columns for illustrative purposes
        ids = ['8_1', '9_1', '10_2', '11_2', '12_3']
        lengths = np.arange(8, 13)
        
        # cutoff dGs
        dGs = dGmat.mean()
        dGs_error = dGmat.std()
        
        parameters = fitFun.fittingParameters()
        index = dGs>parameters.cutoff_dG
        dGs.loc[index] = parameters.cutoff_dG
        dGs_error[index.values] = 0
        
        # plot
        if plot_all:
            plotViolinLength(lengths, dGmat.loc[:,ids], c=c)
        else:
            plotErrorbarLength(lengths, dGs.loc[ids], dGs_error.loc[ids], c=c, linestyle=linestyle)

        return dGs.loc[ids]

def getJunctionsSeqsFlank(motifs, n, flank):
    junction_seqs = []
    m = n
    for motif in motifs:
        seqs = hjh.junction.Junction(motif).sequences
        junction_seqs.append( (flank[:n] + seqs.side1 + flank[m:] + '_' +
                     seqfun.reverseComplement(flank[m:], rna=True) + seqs.side2 + seqfun.reverseComplement(flank[:n], rna=True)))
    return pd.concat(junction_seqs)
        
def getMotif( motif):
    if motif=='wc':
        motif_tuple= [('W', 'W')]
        n = 1
    
    elif motif=='m':
        motif_tuple= [('W', 'M'), ('M', 'W')]
        n = 1           
           
    elif motif=='mm':
        motif_tuple=[('M', 'M')]
        n = 1
    
    elif motif=='mb1':
        motif_tuple=[('M', 'B1', 'W'), ('W', 'B1', 'M')]
        n = 1

    elif motif=='mb2':
        motif_tuple=[('M', 'B2', 'W'), ('W', 'B2', 'M')]
        n = 1
    
    elif motif=='mmm1':
        motif_tuple=[('W', 'M', 'M', 'M')]
        n = 0

    elif motif=='mmm2':
        motif_tuple=[('M', 'M', 'M', 'W')]
        n = 0
    
    elif motif=='bulge1':
        motif_tuple=[('B1',)]
        n = 2
        
    elif motif=='bulge2':
        n = 2
        motif_tuple=[('B2',)]
        
    elif motif=='dbulge1':
        motif_tuple=[('B1', 'B1')]
        n = 2
    
    elif motif=='dbulge2':
        motif_tuple=[('B2', 'B2')]
        n = 2
    
    elif motif == 'tbulge1':
        motif_tuple=[('B1', 'B1', 'B1')]
        n=2
    
    elif motif == 'tbulge2':
        motif_tuple=[('B2', 'B2', 'B2')]
        n=2

    elif motif == 'mbb1':
        motif_tuple=[('M', 'B1', 'B1', 'W'), ('W', 'B1', 'B1', 'M')]
        n=1

    elif motif == 'mbb2':
        motif_tuple=[('M', 'B2', 'B2', 'W'), ('W', 'B2', 'B2', 'M')]
        n=1
    
    else:
        print "motif %s not found!"%motif
        return
    return motif_tuple, n

def plotErrorbarLength(lengths, dGs, dGs_error, ylim=[-12, -7], xlim=[7.5,12.5], ax=None, c=None, linestyle=None, marker='o', markersize=50, apply_cutoff=True, capsize=2):
    if c is None:
        c = 'r'
    if linestyle is None:
        linestyle='-'
        
    parameters = fitFun.fittingParameters()
    
    colors = np.array([c]*len(dGs), dtype='S20')
    if apply_cutoff:
        eps = 1E-6
        colors[(np.abs(dGs - parameters.cutoff_dG) < eps).values] = 'none'

    tight_layout = False
    if ax is None:
        fig = plt.figure(figsize=(2.75, 2.5))
        ax = fig.add_subplot(111)
        tight_layout=True
    ax.scatter(lengths, dGs, c=colors, edgecolor='k', marker=marker, s=markersize, linewidth=1)
    ax.errorbar(lengths, dGs, dGs_error, fmt='-', elinewidth=1, capsize=capsize, capthick=1,
                color=c, linestyle=linestyle, ecolor='k', linewidth=1)

    plt.xticks(np.arange(np.around(xlim[0]), np.around(xlim[1]+1)))
    plt.yticks(np.arange(ylim[0], ylim[1]+1))
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    fix_axes(ax)
    if tight_layout:
        plt.tight_layout()
    
def plotViolinLength(lengths, dGmat, c=None, linestyle=None, apply_cutoff=True, plot_lines=True):
    if c is None:
        c = 'r'
    if linestyle is None:
        linestyle='-'''
    figsize=(3, 3)
    ylim = [-12, -5]
    
    if apply_cutoff:
        parameters = fitFun.fittingParameters()

        dGmat[dGmat > parameters.cutoff_dG] = parameters.cutoff_dG
        ylim=[-12, -7]
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    #sns.violinplot(dGmat,  color="c", ax=ax, scale='width')
    xvalues = np.arange(dGmat.shape[1])

    for motif, dGs in dGmat.iterrows():
        jitter = st.norm.rvs(scale=0.05, size=len(xvalues))
        ax.scatter(xvalues+jitter, dGs, marker='.', c='k', s=5)

        if plot_lines:
            ax.plot(xvalues+jitter, dGs, '-', color='0.3', alpha=0.25, linewidth=0.5, linestyle=linestyle)
    
    sns.boxplot(dGmat,  color=c, ax=ax, width=0.5, linewidth=1, whis=np.inf)
    plt.xticks(xvalues, lengths)
    plt.yticks(np.arange(-12, -5))
    plt.ylim(ylim)
    plt.xlabel('')
    fix_axes(ax)
    plt.tight_layout()
    
class multipleFlowSequenceResults():
    """ Class to store data of multiple tectoData objects. """
    def __init__(self, objectlist, names, predicted_energies):
        self.objectlist = objectlist
        self.names = names
        self.predicted_energies=predicted_energies
        # asume all lbi chars are the same
        self.libChar = objectlist[0].libChar
        self.mat = self.getObservedAndPredicted()
        
    def getObservedAndPredicted(self ):
        observed_dGs = []
        for name, tectoobject in zip(self.names, self.objectlist):
            observed_dGs.append(pd.Series(tectoobject.results.dG, name=name,
                                          index=tectoobject.results.index))
        
        observed_dGs = pd.concat(observed_dGs, axis=1)
        observed_dGs.index = self.libChar.loc[observed_dGs.index].sequence
        return observed_dGs
    
    
    def plotScatterplot(self, names=None, colors=None, randomize=False):
        if names is None:
            names = self.predicted_energies.columns.tolist()

            
        # define colors
        cmap = mpl.cm.get_cmap('Spectral')
        
        observed_dGs = self.getObservedAndPredicted().loc[self.predicted_energies.index]
        plt.figure(figsize=(3,3))
        for i, name in enumerate(names):
            x = observed_dGs.loc[:, name]
            
            if randomize:
                y = [self.predicted_energies.loc[:, col].iloc[idx]
                     for idx, col in enumerate(
                        np.random.choice(names, size=len(self.predicted_energies)))]
            else:
                y = self.predicted_energies.loc[:, name]
            
            if colors is None:
                if len(names) > 1:
                    color = cmap(i/float(len(names)))
                else:
                    color = cmap(0.5)
            else:
                color = colors[i]
            plt.scatter(x, y, c=color, marker='o', s=4, edgecolor='none')
        ax = fix_axes(plt.gca())
        plt.xlabel('observed $\Delta G$ (kcal/mol)')
        plt.ylabel('predicted $\Delta \Delta G$ (kcal/mol)')
        plt.subplots_adjust(left=0.2, bottom=0.2, top=0.95, right=0.95)
        
            
    def compareAllPredictions(self):
        energies = self.predicted_energies
        g = sns.PairGrid(energies, palette=['0.5']);
        g.map_upper(plt.hexbin, cmap='Spectral_r', mincnt=1,
                    extent=[-2, 2, -2, 2], gridsize=100);
        g.map_diag(sns.distplot, hist_kws={'histtype':'stepfilled'})
        g.set(xlim=(-2, 2))
        g.set(ylim=(-2, 2))
    
        
    def compareAllMeasured(self, xlim=[-12.5, -9.5], mat=None, names=None):
        
        if mat is None:
            mat = self.getObservedAndPredicted().astype(float)
        if names is None:
            names = mat.columns.tolist()
        
        fig = plt.figure(figsize=(6.5,5.5))
        gs = gridspec.GridSpec(len(names), len(names),
                               wspace=0.25, right=0.975, top=0.975, )
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if j > i:
                    ax = fig.add_subplot(gs[i, j])
                    ax.hexbin(mat.loc[:, name2], mat.loc[:, name1],
                              cmap='copper', mincnt=1, extent=xlim+xlim, gridsize=150)
                    ax.set_xlim(xlim)
                    ax.set_ylim(xlim)
                    fix_axes(ax)

                elif j==i:
                    ax = fig.add_subplot(gs[i, j])
                    sns.distplot(mat.loc[:, name1], bins=np.linspace(*xlim),
                                 hist_kws={'histtype':'stepfilled'},
                                 color='0.5', kde=False,norm_hist=True)
                    ax.set_xlim(xlim)
                    fix_axes(ax)

    def plotDistributions(self, ):
        mat = self.getObservedAndPredicted().astype(float)
        libChar = self.libChar
        libChar.index = libChar.sequence
        results = pd.concat([libChar.loc[:, :'receptor'], mat], axis=1)
        results.loc[:, 'ddG'] = results.flow_7176 - results.flow_3455
        results.loc[:, 'tc'] = results.loop + '_' + results.receptor
        g = sns.FacetGrid(results.loc[(results.tc=="GGAA_11nt")&(results.sublibrary=="junction_conformations")],
                          col="length", col_wrap=3)
        g.map(sns.distplot, "ddG", hist_kws={'histtype':'stepfilled'});
    
    def plotScatterplotByLength(self, ):
        mat = self.getObservedAndPredicted().astype(float)
        libChar = self.libChar
        libChar.index = libChar.sequence
        results = pd.concat([libChar.loc[:, :'receptor'], mat], axis=1)
        results.loc[:, 'ddG'] = results.flow_7176 - results.flow_3455
        results.loc[:, 'tc'] = results.loop + '_' + results.receptor
        sns.regplot(x="flow_3455", y="flow_7176",
                       data=results.loc[(results.tc=="GGAA_11nt")&(results.sublibrary=="junction_conformations")],
                       hue="length", kind='scatter')
        g = sns.FacetGrid(results.loc[(results.tc=="GGAA_11nt")&(results.sublibrary=="junction_conformations")],
                          col="length", col_wrap=4)
        g.map(sns.distplot, "ddG");
        
        
        g = sns.FacetGrid(results.loc[(results.sublibrary=="tertiary_contacts")],
                          col="tc", col_wrap=4) 
        g.map(sns.distplot, "flow_3455");
        
    def getMatByTertiaryContact(self, receptor_loop="GGAA_A225U"):
        mat = self.mat.astype(float)
        names = mat.columns.tolist()
        libChar = self.libChar
        libChar.index = libChar.sequence
        results = pd.concat([libChar.loc[:, :'receptor'], mat], axis=1)
        results.loc[:, 'tc'] = results.loop + '_' + results.receptor
        submat = results.loc[results.tc == receptor_loop]
        submat.index = (submat.helix_seq + ':' + submat.junction_seq)

        submat2 = results.loc[(results.sublibrary=="tertiary_contacts")&(results.tc=="GGAA_11nt")]
        submat2.index = (submat2.helix_seq + ':' + submat2.junction_seq)
        mat = pd.concat([pd.concat([pd.Series(submat2.loc[submat.index, name], name='wt'),
                                    pd.Series(submat.loc[:, name], name=receptor_loop),
                                    pd.Series(name, index=submat.index, name='flow')], axis=1 )
                         for name in names])
        mat.loc[:, 'ddG'] = mat.loc[:, receptor_loop] - mat.wt
        return mat

def plotScatterPlotByTertContact(submat, name, flow_pieces=None, xlim=None, ylim=None, colors=None):
    if flow_pieces is None:
        flow_pieces = ['10bp', '9bp', 'flow_3455', 'flow_7176']
    if colors is None:
        cmap = mpl.cm.get_cmap('Dark2')
        colors = [cmap(i) for i in np.linspace(0,1,len(flow_pieces))]
    plt.figure(figsize=(3,3));
    
    for i, flow in enumerate(flow_pieces):
        index = submat.flow==flow
        plt.scatter(submat.loc[index].wt, submat.loc[index, name], c=colors[i],
                    rasterized=True,  marker='.', edgecolors='none', vmin=0, vmax=len(flow_pieces)-1);
    plt.xlim(xlim);
    plt.ylim(ylim)
    fix_axes(plt.gca());
    
    index = np.arange(len(submat))[submat.loc[:, ['wt', name]].isnull().any(axis=1).astype(bool).values]
    
    

def plotDeltaDeltaGHistByTertContact(submat, flows, cutoff=-9, colors=None, xlim=None):
    if colors is None:
        colors = [None]*len(flows)
    plt.figure(figsize=(3,3));
    for flow, color in zip(flows, colors):
        sns.distplot(submat.loc[(submat.flow==flow)&(submat.wt < cutoff), 'ddG'].dropna(),
                     hist_kws={'histtype':'stepfilled'}, kde_kws={'bw':0.075}, color=color);
    fix_axes(plt.gca());
    plt.subplots_adjust(bottom=0.15);
    plt.xlim(xlim);



class multipleResults():
    """ Class to store data of multiple tectoData objects. """
    def __init__(self, objectlist, names):
        self.objectlist = objectlist
        self.names = names
        table = self.getPivotedTable()
        eminus = self.getPivotedTable('eminus')
        eplus = self.getPivotedTable('eplus')
        
        # fix stuff
        parameters = fitFun.fittingParameters()
        table[table >= parameters.cutoff_dG] = parameters.cutoff_dG
        eminus[table >= parameters.cutoff_dG] = np.nan
        eplus[table >= parameters.cutoff_dG] = np.nan
        
        self.table = table.astype(float)
        self.eminus=eminus.astype(float)
        self.eplus = eplus.astype(float)
        

    def getPivotedTable(self, param=None, ):
        if param is None: param='dG'
        all_flows = []
        for tectoObject, name in itertools.izip(self.objectlist, self.names):
            subresults = tectoObject.getPivotedTable(param=param)
            subresults.columns = ['%s_%s'%(name, col) for col in subresults.columns.tolist()]
            all_flows.append(subresults)
        table = pd.concat(all_flows, axis=1)
        
        return table

def getJunctionSeqsABunch(flank=None):
    if flank is None:
        flank='GCGC'
    bulged_motifs = ['bulge1', 'bulge2','dbulge1','dbulge2',]
    two_base_motifs = ['wc', 'm', 'mm','mb1','mb2','mbb1', 'mbb2']
    three_base_motifs = ['mmm1', 'mmm2']
    motifs = bulged_motifs + two_base_motifs
    junction_seqs = []
    for motif in motifs:
        if motif in bulged_motifs: flank = flank
        elif motif in two_base_motifs: flank = flank[0] + flank[-1]
        elif motif in three_base_motifs: flank = ''
        else:
            print 'check motif %s'%motif
        
        # find sequences
        motif_tuple, n = getMotif(motif)
        seqs = getJunctionsSeqsFlank(motif_tuple, n, flank)
        junction_seqs.append(
            pd.concat([pd.Series(seqs, name='junction_seq'),
                       pd.Series(motif, name='motif', index=seqs.index),
                       pd.Series(flank, name='flank', index=seqs.index)], axis=1))
    junction_seqs = pd.concat(junction_seqs)
    junction_seqs.index=junction_seqs.junction_seq
    return junction_seqs, motifs

class multipleResultsPlots():
    def __init__(self, tableObject):
        self.table = tableObject.table
        self.eminus = tableObject.eminus
        self.eplus = tableObject.eplus

    def getMat(self, motif=None, flank=None):
        if motif is None:
            motif = 'b'
    
        if motif == 'a bunch':
            junction_seqs, tmp = getJunctionSeqsABunch(flank=flank)
            mat = self.table.loc[junction_seqs.junction_seq].dropna(how='all', axis=0)

        else:
            if motif == 'b':
                motifs = ['B1', 'B2']
            elif motif == 'm':
                motifs = ['W,M', 'M,W', 'W']
            elif motif == 'nn':
                motifs = ['N,N']
                
            pivot = self.table
            index = getJunctionSeqs(motifs, pivot.index, add_flank=False)
            mat = pivot.loc[index]
        return mat
    


    def plotSubsetPcs(self, motif='m', subtract_flank=True,
                      pcs=[0,1],
                      side=None,
                      color_by='seq',
                      position=0,
                      mm_type=None,
                      clusters=None,
                      plot_kde=True):
        mat = self.getMat(motif=motif)
        mat, pca, transformed, projections = processMat(mat,
                                                        subtract_flank=subtract_flank)
        annotations, colors = self.getAnnotations(mat, motif=motif)
        # plot PCs
        annot = pd.Series('other', index=mat.index)
        annot.loc[annotations[1][:,0]==0] = 'wc'
        annot.loc[(annotations[3][:,4:6]>0).any(axis=1)] = 'other'
        annot.loc[(annotations[3][:,:4]>0).any(axis=1)] = 'pu-pu'
        annot.loc[(annotations[3][:,-4:]>0).any(axis=1)] = 'pyr-pyr'

        
        flank_all = pd.Series(annotations[0][:,0], index=mat.index)
        side_all = pd.Series(annotations[1][:,0], index=mat.index)
        seq  = pd.DataFrame(annotations[2], index=mat.index)
        
        # get type
        seqs = np.array(['AA', 'AG', 'GA', 'GG', 'AC', 'CA', 'UG', 'GU', 'UU', 'UC', 'CU', 'CC'])
        mm_type_all = pd.Series('', index=side_all.index)
        for col in np.arange(annotations[3].shape[1]):
            mm_type_all.loc[annotations[3][:, col] > 0] = seqs[col]

        
        cmap = mpl.colors.ListedColormap(colors[0])
        
        # plot a couple examples
        if mm_type is None:
            index = pd.Series(1, index=annot.index).astype(bool)
        else:
            index = pd.concat([mm_type_all == mm for mm in mm_type], axis=1).any(axis=1)
        if side is not None:
            index = index&(side_all==side)
        
        if color_by == 'seq':
            if side is None:
                c = pd.concat([seq.loc[side_all==1, 0], seq.loc[side_all==2, 1]]).loc[index]
            else:
                c = seq.loc[index, position]
            cmap = mpl.colors.ListedColormap(colors[2])
            vmin = 0
            vmax = 4
        elif color_by == 'flank':
            c = flank_all.loc[index]
            cmap = mpl.colors.ListedColormap(colors[0])
            vmin = 0
            vmax = 1
        elif color_by == 'annot':
            c = annotations[3].sum(axis=1)[index.values]
            cmap = mpl.colors.ListedColormap(colors[3])
            vmin = 0
            vmax = 4           
        elif color_by == 'mm':
            c = np.searchsorted(seqs, mm_type_all.loc[index])
            cmap = 'Paired'
            vmin = 0
            vmax = len(seqs)-1      
        elif color_by == 'side':
            c = side_all.loc[index]
            cmap = mpl.colors.ListedColormap(colors[1])
            vmin = 0
            vmax = 2
        elif color_by == 'clusters':
            if clusters is None:
                print "error: must give clusters as input to color by clusters"
                sys.exit()
            c = clusters.loc[index]
            cmap = 'Paired'
            vmin = None
            vmax = None
            
        plt.figure(figsize=(3,3));
        if plot_kde:
            sns.kdeplot(transformed.iloc[:,pcs], n_levels=20, shade=True, cmap='Greys');
        plt.scatter(transformed.loc[index, pcs[0]],
                    transformed.loc[index, pcs[1]],
                    c=c,
                    cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xlabel('pc 1')
        plt.ylabel('pc 2')
        fix_axes(plt.gca())
        plt.subplots_adjust(bottom=0.2, left=0.25, top=0.95, right=0.95)
        
        
    def getAnnotations(self, mat, motif='b', flank=None):
        # make annotations      
        if motif == 'b':
            annotations, colors = getBulgeAnnotations(mat)
        elif motif=='m':
            annotations, colors = getMMAnnotations(mat)
        elif motif=='nn':
            annotations, colors = getNNannotations(mat)
        elif motif=='a bunch':
            annotations, colors = getBunchannotations(mat, flank=flank)
        else:
            print 'Motif %s not found!'%motif
            return
        return annotations, colors

    def consensusClusterOrder(self, mat, k, subtract_flank=False, use_pcs=None, heatmap_scale=2,plot_clusters=True, index=None):
        
        # find those that fit motif
        mat = self.getMat(motif=motif)
        mat, pca, transformed, projections = processMat(mat,
                                                        subtract_flank=subtract_flank,
                                                        use_pcs=use_pcs)
        # get anonations
        annotations, colors = self.getAnnotations(mat=mat, motif=motif)
        
        if index is not None:
            mat = mat.loc[index]
            annotations = [annotation[index.values] for annotation in annotations]
            transformed = transformed.loc[index]
        
        if use_pcs is None:
            use_pcs = transformed.columns
        
        # order and cluster 
        submat = fillNaMat(mat)
        labels, M, order = clusterFun.consensusCluster(transformed.loc[:, use_pcs], method=clusterFun.clusterKmeans, k=k, plot=False)
        labels, z = clusterFun.clusterHierarchical(M, k, return_z=True)
        
        info = pd.concat([pd.Series(np.arange(len(labels)), index=labels.index, name='order'),
                          pd.Series(labels, name='labels'),
                          pd.Series(transformed.loc[:, use_pcs].mean(axis=1), name='mean_pcs')], axis=1)
        new_clusters = np.argsort(np.argsort(info.groupby('labels')['mean_pcs'].mean()))
        info.loc[:, 'new_clusters'] = new_clusters.loc[info.loc[:, 'labels']].values
        
        #order = info.sort(['new_clusters', 'mean_pcs']).order.values
        
        index = info.iloc[order].index
        info.loc[index, 'new_order'] = np.arange(len(info))
        new_order = info.sort(['new_clusters', 'new_order']).order.values
        
        plotClusterGram(mat,
                        annotations=annotations, colors=colors, heatmap_scale=heatmap_scale,
                        order=new_order, clusters=labels, z=z, transformed=transformed.loc[:, use_pcs],
                        plot_meanplot=False, plot_clusters=plot_clusters, plot_dendrogram=False)
        
        return mat, annotations, labels, order
    
    def plotClusterGram(self, motif='b', ref_ind='GCGC_GCAGC', ref_vector=None,
                        use_wc_ref=True,
                        heatmap_scale=None, use_pcs=None, t=None, criterion=None,
                        plot_meanplot=None,
                        subtract_flank=False,
                        plot_clusters=None,
                        index=None,
                        order=None,
                        clusters=None,
                        z=None):
        pivot = self.table
        
        # deal with ref_vector
        if use_wc_ref:
            wc_index = getJunctionSeqs(['W'], pivot.index, add_flank=False)
            ref_vector = pivot.loc[wc_index].mean()
            
        # deal with the one point to plot
        vec = pivot.loc[ref_ind]
        error = [self.eminus.loc[ref_ind], self.eplus.loc[ref_ind]]
        
        # find those that fit motif
        mat = self.getMat(motif=motif)
        mat, pca, transformed, projections = processMat(mat,
                                                        subtract_flank=subtract_flank,
                                                        use_pcs=use_pcs)
                
        # make annotations      
        annotations, colors = self.getAnnotations(mat=mat, motif=motif)
        
        if index is not None:
            mat = mat.loc[index]
            annotations = [annotation[index.values] for annotation in annotations]
            transformed = transformed.loc[index]

        # cluster
        submat = transformed
        if use_pcs is not None:
            submat = transformed.loc[:, use_pcs]
        if z is None:
            method = 'average'
            z = sch.linkage(submat, method=method, metric='euclidean')
        
        if clusters is None:
            if t is not None and criterion is not None:
                clusters = pd.Series(sch.fcluster(z, t=t, criterion=criterion), index=submat.index)
            else:
                clusters = None
        
        if order is None:
            order = sch.leaves_list(z)
        #plot
        plotClusterGram(mat, ref_vector=ref_vector, vec=vec, error=error,
                        annotations=annotations, colors=colors, heatmap_scale=heatmap_scale,
                        t=t, criterion=criterion, order=order, clusters=clusters, z=z,transformed=submat,
                        plot_meanplot=plot_meanplot, plot_clusters=plot_clusters)

        return mat, annotations, clusters, order
    
    def findMinDistanceToWC(self, motif='m', subtract_flank=True, use_pcs=None):
        # find those that fit motif
        mat = self.getMat(motif=motif)
        mat, pca, transformed, projections = processMat(mat,
                                                        subtract_flank=subtract_flank,
                                                        use_pcs=use_pcs)
        if use_pcs is not None:
            transformed_mat = transformed.loc[:, use_pcs]
        else:
            transformed_mat = transformed
        euclidean_distance = pd.DataFrame(ssd.squareform(ssd.pdist(transformed_mat)), index=mat.index, columns=mat.index)

        # get type
        annotations, colors = self.getAnnotations(mat, motif=motif)
        seqs = np.array(['AA', 'AG', 'GA', 'GG', 'AC', 'CA', 'UG', 'GU', 'UU', 'UC', 'CU', 'CC'])
        mm_type_all = pd.Series('', index=mat.index)
        for col in np.arange(annotations[3].shape[1]):
            mm_type_all.loc[annotations[3][:, col] > 0] = seqs[col]

        allDistance = {}
        all_types = ['wc'] + list(seqs)
        for mm_type in all_types:
            
            allDistance[mm_type] = euclidean_distance.loc[mm_type_all==mm_type, mm_type_all==''].min(axis=1)

        allDistance['wc'] = ((euclidean_distance.loc[mm_type_all=='', mm_type_all==''])[euclidean_distance.loc[mm_type_all=='', mm_type_all==''] >0]).min()
        allDistance = pd.concat(allDistance, axis=1).loc[:, all_types]
        plt.figure(figsize=(3, 2.5))
        sns.boxplot(allDistance,
                    palette=['w']+['b']*4+['c']*2 + ['y']*2+['g']*4,
                    width=0.7, linewidth=1, fliersize=2)
        ax = fix_axes(plt.gca())
        xticks = ax.get_xticks()
        plt.xticks(xticks, all_types, rotation=90)
        #plt.ylabel('average distance to WC motifs')
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95)
        #x = 0
        #for mm_type in all_types:
        #    y = allDistance.loc[:, mm_type].dropna()
        #    jitter = st.norm.rvs(scale=0.01, size=len(y))
        #    plt.scatter(x+jitter, y, marker='.', c='k', s=10)
        #    x +=1
        return 
        
    def findPCA(self, motifs=None, motif=None):
        mat = self.getMat(motif=motif)
        mat, pca, transformed, projections = processMat(mat)    
        submat = fillNaMat(mat)
        return doPCA(submat)

def processMat( mat, subtract_flank=False, use_pcs=None):
    mat = mat.copy()
    # find flank
    if subtract_flank:
        mat = subtractFlank(mat)
    
    pca, transformed, projections = doPCA(mat) 
    if use_pcs is not None:

        mat = projectOnePc(mat, transformed, projections, use_pcs)
    
    return mat, pca, transformed, projections

def projectOnePc(mat, transformed, projections,use_pcs):
    submat = pd.DataFrame(np.dot(transformed.loc[:, use_pcs],
                                 projections.loc[use_pcs]).astype(float),
                          index=mat.index, columns=mat.columns)
    submat[mat.isnull()] = np.nan
    return submat + mat.mean().values

def reorderMat(x, labels, order):
    indices_not_ordered = labels.index.tolist()
    orderMat = pd.concat([pd.Series(labels.iloc[order], name='cluster'),
                          pd.Series(np.arange(len(labels)), index=labels.iloc[order].index, name='order')], axis=1)
    clusterMeans = pd.concat([orderMat.cluster, x], axis=1).groupby('cluster').mean()
    clusters = np.array(clusterMeans.index.tolist())
    clusters_ordered = clusters[np.array(sch.leaves_list(sch.linkage(clusterMeans, method='average')))]
    
    # sort
    index_ordered = []
    for cluster in clusters_ordered:
        index = orderMat.loc[orderMat.cluster==cluster].index.tolist()
        index_ordered+=index
    
    return [indices_not_ordered.index(i) for i in index_ordered]
        

def findSignificanceInCluster(clusters, annotation, annotation_val, order):
    u, ind = np.unique(clusters.iloc[order], return_index=True)
    clusters_ordered = u[np.argsort(ind)]
    annotationMat = pd.DataFrame(annotation, index=clusters.index)
    p = pd.DataFrame(index=clusters_ordered, columns=annotationMat.columns)
    M = len(annotationMat)
    for col in annotationMat:
        vec = annotationMat.loc[:, col]
        n = (vec==annotation_val).sum()
        for cluster in clusters_ordered:
            N = (clusters==cluster).sum()
            x = (vec.loc[clusters==cluster]==annotation_val).sum()
            p.loc[cluster, col] = 1-st.hypergeom.cdf(x, M, n, N)
    return p

def findSignificanceInClusterTwoTailed(clusters, annotation, cluster_val):
    annotationMat = pd.Series(annotation, index=clusters.index)
    annotationVals = np.unique(annotationMat)
    p = pd.Series(index=annotationVals)
    odds = pd.Series(index=annotationVals)
    
    
    for annotation_val in annotationVals:
        table = [
            [(annotationMat.loc[clusters==cluster_val]==annotation_val).sum(),
             (annotationMat.loc[clusters!=cluster_val]==annotation_val).sum()],
            [(annotationMat.loc[clusters==cluster_val]!=annotation_val).sum(),
             (annotationMat.loc[clusters!=cluster_val]!=annotation_val).sum() ]]
        
        odds.loc[annotation_val], p.loc[annotation_val] = st.fisher_exact(table)
    return p, odds


def subtractFlank(mat, plot=False):
    flank, color = getFlankAnnotation(mat)
    flank_series = pd.Series(np.ravel(flank), index=mat.index, dtype=bool)
    diff = mat.loc[~flank_series].mean() - mat.loc[flank_series].mean()
    
    if plot:
        n = mat.shape[1]
        fig = plt.figure(figsize=(5,1.5))
        ax = fig.add_subplot(111)
        x = np.arange(n)
        plotErrorbarLength(x-0.1, mat.loc[~flank_series].mean(), mat.loc[~flank_series].std(),
                           linestyle='',
                           xlim=[-0.5, n-0.5],
                           c='#00A79D',
                           marker='s',
                           capsize=0,
                           ax=ax,
                           markersize=20,
                           apply_cutoff=False)
        plotErrorbarLength(x+0.1, mat.loc[flank_series].mean(), mat.loc[flank_series].std(),
                           linestyle='',
                           xlim=[-0.5, n-0.5],
                           c='#9E1E62',
                           markersize=20,
                           capsize=0,
                           ax=ax,
                           apply_cutoff=False)
        plt.xticks([])
        ax.set_yticklabels([])
        plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
        
    return pd.concat([mat.loc[~flank_series]-diff*0.5, mat.loc[flank_series]+diff*0.5]).loc[mat.index]
    
def getFlankAnnotation(mat):
    flank = np.vstack([0 if s[0]=='G' else 1 for s in mat.index.tolist()])
    flank_color = ['#00A79D', '#9E1E62']
    return flank, flank_color

def getBulgeAnnotations(mat):
    flank, flank_color = getFlankAnnotation(mat)
    
    side_color = ['#1B75BB', '#8A5D3B']
    bulge_color = ['#921926', '#D6DF23', '#F05A28', '#979697']
    side = np.vstack([0 if len(x) > len(y) else 1 for x, y in [s.split('_') for s in mat.index.tolist()]])
    
    n = 2
    bulge_seq = np.searchsorted(['A', 'C', 'G', 'U'],
        np.vstack([x[n] if len(x) > len(y) else y[n] for x, y in [s.split('_') for s in mat.index.tolist()]]))
   
    annotations=[flank, side, bulge_seq]
    colors=[flank_color, side_color, bulge_color]
    return annotations, colors

def getMMAnnotations(mat):
    flank, flank_color = getFlankAnnotation(mat)
    
    structMat = processStructureMat(mat)
    position = np.vstack((structMat.iloc[:, 1:3] > 0).sum(axis=1) + (structMat.loc[:, 1] > 0)).astype(float)
    position_colors = ['w', '0.5', 'k']
    
    #wc_base = np.searchsorted(['A', 'C', 'G', 'U'],
    #    np.vstack([s[1] if struct.loc[1]==0 else s[2] for s, struct in structMat.iterrows()]))
    wc_base = np.array([np.searchsorted(['A', 'C', 'G', 'U'], list(s[1:3] + '_' + s[-3:-1])) for s in structMat.index.tolist()])
    wc_base_color = ['#921926', '#D6DF23', '#F05A28', '#979697', 'k']
    
    seqs = np.array(['AA', 'AG', 'GA', 'GG', 'AC', 'CA', 'UG', 'GU', 'UU', 'UC', 'CU', 'CC'])
    struct = np.array([seqs==s for s in [s[2]+s[-3] if struct.loc[1]==0 else s[1]+s[-2]
                                         for s, struct in structMat.iterrows()]]).astype(int)
    struct[:, 4:6] = struct[:, 4:6]*2
    struct[:, 6:8] = struct[:, 6:8]*3
    struct[:, 8:]  = struct[:, 8:]*4
    struct = np.hstack([np.vstack((struct.sum(axis=1) == 0).astype(int)*5), struct])
    
    struct_colors = ['w', 'b', 'c', 'y', 'g', 'k']
    annotations=[flank, position, wc_base, struct]
    colors=[flank_color, position_colors, wc_base_color, struct_colors]
    
    #annotations = [struct]
    #colors = [struct_colors]
    return annotations, colors 

def getBunchannotations(mat, flank=None):
    junction_seqs, motifs = getJunctionSeqsABunch(flank=flank)

    annotation_motif = np.vstack([motifs.index(m) for m in junction_seqs.loc[mat.index].motif])
    color_motif = [(0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0),
     (0.16678200852052832, 0.50226836052595392, 0.69296426282209511, 1.0),
     (0.59843138754367831, 0.82509804964065558, 0.46745100319385535, 1.0),
     (0.41837754950803863, 0.62089967657538025, 0.2915647927452536, 1.0),
     (0.94666666984558101, 0.40313726961612706, 0.40392158329486855, 1.0),
     '#B93D3E', '#7B121A', '#906685', '#621B56', '#D4A250','#9C6A43']
    
    annotation_bulge = np.vstack([np.searchsorted([-2, -1, 0, 1, 2],
        len(s.split('_')[1]) - len(s.split('_')[0])) for s in mat.index.tolist()])
    cmap = mpl.cm.get_cmap('coolwarm')
    color_bulge= [cmap(i) for i in np.linspace(0, 1, len(np.unique(annotation_bulge)))]
    return [annotation_motif, annotation_bulge], [color_motif, color_bulge]
    
def getNNannotations(mat):
    flank, flank_color = getFlankAnnotation(mat)
    
    structMat = processStructureMat(mat)
    mm = np.vstack((structMat.iloc[:, 1:3] > 0).sum(axis=1)).astype(float)
    mm_colors = ['w', 'r', 'b']
    
    annotations = [flank, mm]
    colors=[flank_color,mm_colors]
    return annotations, colors

def plotClusters(clusters, order, ax):
    previous_cluster = clusters.iloc[order].iloc[0]
    for i, cluster in enumerate(clusters.iloc[order]):
        if cluster==previous_cluster:
            pass
        else:
            ax.axhline(i-0.5, color='k', linewidth=0.5)
            previous_cluster = cluster
        
def plotClusterGram(mat, ref_vector=None, vec=None, error=None, annotations=[], colors=[],
                    heatmap_scale=None, t=None, criterion=None, plot_clusters=None,
                    transformed=None, plot_meanplot=None, pc_heatmap_scale=None,
                    order=None, clusters=None, z=None, plot_dendrogram=None, vmin=None, vmax=None,
                    cmap='coolwarm'):
    if heatmap_scale is None:
        heatmap_scale = 2
    if vmin is None:
        vmin = -heatmap_scale
    if vmax is None:
        vmax = heatmap_scale
    if pc_heatmap_scale is None:
        pc_heatmap_scale = 2.5
    if transformed is None:
        plot_transformation = False
    else:
        plot_transformation = True
    if plot_clusters is None:
        plot_clusters = True
    if plot_meanplot is None:
        plot_meanplot = True
    if plot_dendrogram is None:
        plot_dendrogram = True
    if t is None:
        t = 12
    if criterion is None:
        criterion='maxclust'

    mat = mat.astype(float) # to plot - includes Nans
    submat = fillNaMat(mat)  # do operations on this
    
    if z is None:
        if plot_transformation:
            z = sch.linkage(transformed, method='average', metric='euclidean')
        else:
            z = sch.linkage(submat, method='average', metric='euclidean')

    if clusters is None:
        clusters = pd.Series(sch.fcluster(z, t=t, criterion=criterion), index=submat.index)

    # cluster PCA
    if ref_vector is None:
        ref_vector = submat.mean()

    # plot
    width_ratios = [0.5, 4]
    if len(annotations) > 0:
        width_ratios += [0.1]
        for annotation in annotations:
            width_ratios += [0.1] + [0.2*annotation.shape[1]]
    if plot_transformation:
        width_ratios = width_ratios[:1] + [0.5, 0.1] + width_ratios[1:]
    
    height_ratios = [6]
    if plot_meanplot:
        height_ratios = [1.5, 0.3] + height_ratios

    with sns.axes_style('white', {'lines.linewidth': 0.5}):
        fig = plt.figure(figsize=(5, 4.5))
        gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios,
                               left=0.01, right=0.99, bottom=0.01, top=0.99,
                               hspace=0.1, wspace=0.01)
        ax_row_start = 1
        if plot_meanplot:
            ax_col_start = 2
        else:
            ax_col_start = 0

        # plot dendrogram
        if plot_dendrogram:
            dendrogramAx = fig.add_subplot(gs[ax_col_start,0])
            y = sch.dendrogram(z, orientation='right', ax=dendrogramAx, no_labels=True, link_color_func=lambda k: 'k')
            dendrogramAx.set_xticks([])
            sns.despine(ax=dendrogramAx, bottom=True, top=True, left=True, right=True)
            # find order from here
            if not np.all(np.array(y['leaves']) == order):
                print 'Error: dendrogram output not in same order as subsequent'
                
        # plot PCS
        if plot_transformation:
            pcaAx = fig.add_subplot(gs[ax_col_start,ax_row_start])
            pcaAx.imshow(transformed.iloc[order], cmap='RdBu_r', aspect='auto',
                         interpolation='nearest', origin='lower', vmin=-pc_heatmap_scale, vmax=pc_heatmap_scale)
            pcaAx.set_yticks([])
            pcaAx.set_xticks([])
            ax_row_start +=2
            if plot_clusters:
                plotClusters(clusters, order, pcaAx)
        
        # plot ddGs   
        heatmapAx = fig.add_subplot(gs[ax_col_start,ax_row_start])
        heatmapAx.imshow(mat.iloc[order] - ref_vector.loc[mat.columns], cmap=cmap, aspect='auto',
                     interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
        heatmapAx.set_yticks([])
        heatmapAx.set_xticks([])
        if plot_clusters:
            plotClusters(clusters, order, heatmapAx)

        # plot single ddG
        if plot_meanplot:
            miniheatmapAx = fig.add_subplot(gs[1,ax_row_start])
            miniheatmapAx.imshow(np.array([vec - ref_vector]).astype(float), cmap=cmap, aspect='auto',
                             interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax) 
        
            miniheatmapAx.set_yticks([])
            miniheatmapAx.set_xticks([])
            
        # for each annotation, plot heat map
        for i, (annotation, colormap) in enumerate(itertools.izip(annotations, colors)):
            cmap = mpl.colors.ListedColormap(colormap)
            annotAx = fig.add_subplot(gs[ax_col_start,ax_row_start+3+i*2])
            annotAx.imshow(annotation[order], cmap=cmap, aspect='auto',
                     interpolation='nearest', origin='lower', vmin=0, vmax=len(colormap)-1)
            #annotAx.imshow(annotation[order], cmap=None, aspect='auto',
            #         interpolation='nearest', origin='lower', vmin=0, vmax=5)
            annotAx.set_yticks([])
            annotAx.set_xticks([])
            if plot_clusters:
                plotClusters(clusters, order, annotAx)
                
    # plot mean plot
    if plot_meanplot:
        meanPlotAx = fig.add_subplot(gs[0,ax_row_start])
        x = np.arange(len(ref_vector))
        plotErrorbarLength(x, ref_vector, None,
                           linestyle='',
                           xlim=[-0.5, len(ref_vector)-0.5],
                           c=sns.xkcd_rgb['cobalt'],
                           marker='s',
                           ax=meanPlotAx,
                           markersize=30,
                           apply_cutoff=False)
        plotErrorbarLength(x, vec, error,
                           linestyle='',
                           xlim=[-0.5, len(ref_vector)-0.5],
                           c='#F05A28',
                           markersize=30,
                           ax=meanPlotAx,
                           apply_cutoff=False)
        plt.xticks([])
        meanPlotAx.set_yticklabels([])
        #plotMeanPlot(submat, ax=meanPlotAx) 
    return 


def plotBulgePCs(tableObject):
    pca, transformed, projections = tableObject.findPCA(['B1', 'B2'])
    side = np.array([len(i.split('_')[0]) - len(i.split('_')[1]) for i in transformed.index])
    flank = np.array([i[:2] for i in transformed.index])
    cmap = mpl.colors.ListedColormap(['#EF4036', '#F05A28', '#F7931D', '#F4F5F6', '#27A9E1', '#1B75BB', '#2E3092'][::-1])

    side = np.array([len(i.split('_')[0]) - len(i.split('_')[1]) for i in transformed.index])
    index = flank=='GC'
    ax = plotPCs(transformed.loc[index], c=side[index], marker='o', cmap=cmap, vmin=-1, vmax=1)

    index = flank=='CU'
    ax = plotPCs(transformed.loc[index], c=side[index], marker='s', cmap=cmap, ax=ax,vmin=-1, vmax=1)
    
    c_seq = np.array([0, 0, 2, 1, 3, 2, 1, 3, 0, 2, 0, 2, 1, 3, 1, 3])
    cmap_seq = mpl.colors.ListedColormap(['#921927', '#F1A07E', '#FFFFFF', '#989798', '#000000'])

    index = flank=='GC'
    ax = plotPCs(transformed.loc[index], c=c_seq[index], marker='o', cmap=cmap_seq, vmin=0, vmax=4, pcs=[1,2]);

    index = flank=='CU'
    plotPCs(transformed.loc[index], c=c_seq[index], marker='s', cmap=cmap_seq, vmin=0, vmax=4, pcs=[1,2], ax=ax);
    plt.xlim(-1.5, 1.5)
    
    index = flank=='GC'
    ax = plotPCs(transformed.loc[index], c=c_seq[index], marker='o', cmap=cmap_seq, vmin=0, vmax=4, pcs=[0,1]);

    index = flank=='CU'
    plotPCs(transformed.loc[index], c=c_seq[index], marker='s', cmap=cmap_seq, vmin=0, vmax=4, pcs=[0,1], ax=ax);
    



    
def plotPCs(transformed, pcs=[0,1], c='k', marker='o', ax=None, cmap=None, vmin=None, vmax=None):
    if ax is None:
        fig = plt.figure(figsize=(3,3));
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.2, left=0.25)

    plt.scatter(transformed.loc[:, pcs[0]], transformed.loc[:,pcs[1]], c=c,
                vmin=vmin, vmax=vmax, marker=marker, cmap=cmap, s=50)
    plt.xlabel('PC %d'%(pcs[0]+1))
    plt.ylabel('PC %d'%(pcs[1]+1));
    plt.xticks(np.arange(-2, 3))
    ax = fix_axes(ax)
    plt.tight_layout();
    return ax


def plotProjectionFlat(pca, transformed, projection, toplot=np.arange(3)):
    
    # plot projections
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #cmap = 'RdBu_r'
    plt.figure(figsize=(4,1))
    im = plt.imshow(projection.loc[toplot].astype(float), vmin=-0.5, vmax=0.5, cmap=cmap,
               interpolation='nearest', aspect='auto')
    ax = fix_axes(plt.gca())
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0.01, right=0.95, top=0.9, bottom=0.1)
    
    plt.figure(figsize=(1,1));
    n = 50
    plt.imshow(np.vstack(np.linspace(-0.5, 0.5, n)), vmin=-0.5, vmax=0.5, cmap=cmap,
               interpolation='nearest', aspect='auto', origin='lower')
    plt.xticks([])
    plt.yticks([-0.50, (n-1)*0.5-0.5, n-0.5], ['-0.5', '0.0', '0.5'])
    plt.subplots_adjust(left=0.5, right=0.8, top=0.9, bottom=0.1)
    fix_axes(plt.gca())
    
    # plot variance explained
    plt.figure(figsize=(1.5,1.5));
    (pd.Series(pca.explained_variance_ratio_[toplot][::-1], index=toplot[::-1]).
        plot(kind='barh', width=1, edgecolor='w', linewidth=1, color='0.5',
             ))
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.25)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.xlim([0, 0.6])
    fix_axes(plt.gca())
    
    # plot transformed matrix
    z = sch.linkage(transformed.loc[:, toplot], method='average', metric='euclidean')
    order = sch.leaves_list(z)
    with sns.axes_style('white', {'lines.linewidth': 0.5}):
        fig = plt.figure(figsize=(2, 4.5))
        gs = gridspec.GridSpec(1, 2,  left=0.01, right=0.99, bottom=0.01, top=0.99,
                               wspace=0)
        
        dendrogramAx = fig.add_subplot(gs[0,0])
        sch.dendrogram(z, orientation='right', ax=dendrogramAx, no_labels=True, link_color_func=lambda k: 'k')
        dendrogramAx.set_xticks([])
        sns.despine(ax=dendrogramAx, bottom=True, top=True, left=True, right=True)

        heatmapAx = fig.add_subplot(gs[0,1])
        heatmapAx.imshow(transformed.iloc[order].loc[:, toplot], cmap='RdBu_r', aspect='auto',
                     interpolation='nearest', origin='lower', vmin=-2.25, vmax=2.25)
        heatmapAx.set_yticks([])
        heatmapAx.set_xticks([])
    

def plotProjection(projection):
    mat = pd.concat([projection,
               pd.Series([s.split('_')[0] for s in projection.index.tolist()], index=projection.index),
               pd.Series([int(s.split('_')[1]) for s in projection.index.tolist()], index=projection.index),
               pd.Series([int(s.split('_')[2]) - (int(s.split('_')[1])-4)/2+1 for s in projection.index.tolist()], index=projection.index),], axis=1)
    mat.columns=['proj', 'flow', 'chip', 'pos']
    mat.loc[:, 'id'] = mat.chip.astype(str) + '_' + mat.pos.astype(str)
    
    ids = ['8_0', '9_-1', '9_0', '9_1', '10_-1', '10_0', '10_1', '11_0', '12_0']
    keys = ['9bp', '10bp', '11bp']
    pivot = mat.pivot('flow', 'id', 'proj').astype(float).loc[keys, ids]
    
    plt.figure(figsize=(3.5,2.5))
    sns.heatmap(pivot, vmin=-0.5, vmax=0.5)
    plt.tight_layout()
    ax = fix_axes(plt.gca())


class simulatedEnsembles():
    def __init__(self):
        self.mat = pd.DataFrame(0, index=np.arange(50), columns=np.arange(70))
    
    def addDensity(self, center=[2,1], cov=[[2.0, 0.3], [0.3, 0.5]]):
        mat = self.mat
        normobject = st.multivariate_normal(center, cov)
        for i, j in itertools.product(mat.index.tolist(), mat.columns.tolist()):
            mat.loc[i, j] = normobject.pdf([i,j])
        
        self.mat = mat
    
    def plotDensity(self):
        fig = plt.figure(figsize=(2,2))
        plt.imshow(self.mat, interpolation='nearest', aspect='equal')
        fix_axes(plt.gca())
        plt.xticks([])
        plt.yticks([])

        
    def addRectangle(self, corners):
        for i in range(4):
            plt.plot([corners[i][0], corners[i-1][0]],
                [corners[i][1], corners[i-1][1]], 'k')

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

def fillNaMat(mat):
    submat = mat.copy()
    for col in mat:
        index =  mat.loc[:, col].isnull()
        submat.loc[index, col] = mat.loc[:, col].mean()
    return submat



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
                        'M,M,M,W':('M', 'M', 'M', 'W'),
                            'N,N':('W', 'N', 'N', 'W')}
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

def findStandardizeMat(submat):
    return (submat - submat.mean())/submat.std()

def doPCA(submat, whiten=None, fillNa=True, standardizeMat=False):
    if fillNa:
        submat = fillNaMat(submat)
    if standardizeMat:
        submat = findStandardizeMat(submat)
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
