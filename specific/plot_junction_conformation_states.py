import datetime
import scipy.cluster.hierarchy as sch
import scipy.cluster as sc
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy.spatial.distance as ssd
import IMlibs
import fitFun

sys.path.append('/home/sarah/array_image_tools_SKD/specific/')
#sys.path.append('/Users/Sarah/python/array_image_tools_SKD/specific/')
from mergeReplicates import errorPropagateAverage
import clusterFun
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1})

parameters = fitFun.fittingParameters()

junction_seq_ref = 'UGAUCU_AGAUCA'

# load consensus mat, labels, seq order
M = pd.read_pickle('final_WC/clustering/consensus_mat.subsample_0.8.sample_1E4.clusters_15.mat.pkl')
labels =  pd.read_pickle('final_WC/clustering/consensus_mat.subsample_0.8.sample_1E4.clusters_15.labels.pkl')
seq_order = np.loadtxt('final_WC/clustering/seq_order.txt', dtype=str)
newresults = pd.read_pickle('final_WC/clustering/data_to_cluster.mat.pkl')
#pvalues = pd.read_pickle('final_WC/consensus_mat.21_clusters.subsample_0.5.500_samples.pvalues.pkl')
#fraction = pd.read_pickle('final_WC/consensus_mat.21_clusters.subsample_0.5.500_samples.fraction.pkl')


# load data and cluster labels
index_ss = newresults.groupby('junction_seq')['ss_correct'].all()
pivot = newresults.pivot(index='junction_seq', columns='id', values='dG')
cols = (['wc10_%s'%i for i in ['8_1', '9_0', '9_1','9_2','10_1','10_2','10_3','11_2','12_3',]]
       +['wc11_%s'%i for i in ['9_0', '9_1','9_2','10_1','10_2','10_3','11_2']])
dGs = pivot.loc[index_ss,cols].astype(float)

pivot[pivot >= parameters.cutoff_dG] = parameters.cutoff_dG
ddGs = (pivot - pivot.loc[junction_seq_ref]).loc[index_ss, cols].astype(float)

# plot consensus index heatmap
with sns.axes_style('white'):
    fig = plt.figure(figsize=(5,5));
    plt.subplots_adjust(bottom=0.25, top=0.97, right=0.9, left=0.15)
    cg = sns.heatmap(M.loc[seq_order, seq_order],
                     yticklabels=False,
                     xticklabels=False,
                     square=True)

# plot heatmap of delta delta Gs
with sns.axes_style('white'):
    fig = plt.figure(figsize=(4,6));
    plt.subplots_adjust(bottom=0.25, top=0.97, right=0.9, left=0.15)
    ax = fig.add_subplot(111)
    sns.heatmap(ddGs.loc[seq_order], yticklabels=False, ax=ax,
                cbar_kws={"label": "$\Delta \Delta G$"});
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # add lines delineating clusters
    for cluster in np.unique(labels):
        ax.plot(xlim, [ylim[-1]-np.where(labels.loc[seq_order]==cluster)[0][-1]]*2,
                'k--', alpha=0.5, linewidth=0.5)
    
# find fraction of each ss in each cluster
junction_ss = newresults.groupby('junction_seq').first().loc[labels.index].junction_SS
junctionType = pd.DataFrame(index=labels.index, columns=np.unique(junction_ss))
for ss in junctionType:
    junctionType.loc[:, ss] = junction_ss.loc[junctionType.index] == ss

numTypes = junction_ss.value_counts()
enrichment = pd.DataFrame(index=np.unique(labels),
                          columns=numTypes.index)
pvalue = pd.DataFrame(index=np.unique(labels),
                          columns=numTypes.index)
fraction = pd.DataFrame(index=np.unique(labels),
                          columns=numTypes.index)
for i, ss in itertools.product(enrichment.index, enrichment.columns):
    #num_junctions = (junction_ss.loc[labels.index] == ss).sum()
    num_ss_in_cluster = (junction_ss.loc[labels.index].loc[labels==i] == ss).sum()
    num_ss_total = (junction_ss.loc[labels.index] == ss).sum()
    num_in_cluster = (labels==i).sum()
    num_total = len(labels)
    fraction_of_cluster = num_in_cluster/float(num_total)
    if num_ss_in_cluster > 0:
        enrichment.loc[i, ss] = np.log2(num_ss_in_cluster/(num_ss_total*fraction_of_cluster))
    pvalue.loc[i, ss] = st.hypergeom.sf(num_ss_in_cluster, num_total,
                                                  num_ss_total, num_in_cluster)
    fraction.loc[i, ss] = num_ss_in_cluster/float(num_ss_total)
    #enrichment.loc[:, ss] = [(junction_ss.loc[labels.index].loc[labels==i] == ss).sum()
    #     for i in enrichment.index]
    #
logpvalue = -np.log10((pvalue + 1E-12).astype(float))
# cluster topologies
cg_ss = sns.clustermap(logpvalue, row_cluster=False, figsize=(5,4.5))
plt.subplots_adjust(bottom=.25, left=0.025, top=0.975)
plt.savefig(os.path.join(figDirectory, 'clustermap.pvalues.15_clusters.png'))

order = pvalue.iloc[:,cg_ss.dendrogram_col.reordered_ind].columns

# make matching heatmap
fig = plt.figure(figsize=(5,6))
plt.subplots_adjust(bottom=0.25, top=0.97, right=0.9, left=0.15)
ax = fig.add_subplot(111)
sns.heatmap(logpvalue.loc[labels.loc[seq_order]].loc[:, order], ax=ax, yticklabels=False,
            cbar_kws={"label": "$-log_{10}$$p$"});
xlim = ax.get_xlim()
ylim = ax.get_ylim()


#### other stuff #####
# plot MSE in same order
with sns.axes_style('white'):
    plt.figure()
    sns.heatmap(M.loc[seq_order, seq_order], yticklabels=False, xticklabels=False)

rmse = pd.DataFrame(0, index=ddGs.index, columns=ddGs.index)
rmse.loc[:] = ssd.squareform(ssd.pdist(ddGs))
rmse = rmse.loc[M.index, M.columns]

with sns.axes_style('white'):
    plt.figure()
    sns.heatmap(rmse.loc[seq_order, seq_order],yticklabels=False, xticklabels=False)

# plot average rmse between clusters
clusters = np.unique(labels)
average_rmse = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
for i, j in itertools.product(clusters, clusters):
    average_rmse.loc[i, j] = np.mean(np.ravel(rmse.loc[labels==i, labels==j]))

with sns.axes_style('white'):
    plt.figure()
    sns.heatmap(average_rmse)

with sns.axes_style('white'):
    plt.figure(figsize=(6,5))
    sns.heatmap(average_rmse.loc[labels.loc[seq_order], labels.loc[seq_order]].astype(float),
                yticklabels=False, xticklabels=False,  vmin=1, vmax=5)

# plot chevron plot
dG_ref = dGs.loc[junction_seq_ref]
dG_ref.loc[dG_ref > parameters.cutoff_dG] = parameters.cutoff_dG
for cluster in clusters:
    index = labels.loc[labels==cluster].index
    tectoPlots.plotChevronPlot3(dGs, index=index, flow_id='wc11', dG_ref=dG_ref,
                                cutoff=parameters.cutoff_dG)
    plt.savefig(os.path.join(figDirectory, 'chevron_plots2.wc_11.cluster_%d.pdf'%cluster))
    plt.close()
    
# print info about each cluster
qvalues = pd.concat([pd.Series(multipletests(pvalues.loc[:, col])[1],
                               index=pvalues.index,
                               dtype=float) for col in pvalues],
                    axis=1, keys=pvalues.columns)

table = newresults.groupby('junction_seq').first().loc[labels.index]
pvalue_threshold = 0.05/15
for cluster in clusters:
    # find signficant junctions
    pvalue = pvalues.loc[cluster].astype(float)
    fractions = fraction.loc[cluster]
    fractions.sort()
    fractions = fractions.loc[fractions > 0]
    
    counts = table.loc[labels==cluster].junction_SS.value_counts()
    tectoPlots.plotClusterStats(fractions, pvalue, counts=counts, pvalue_threshold=pvalue_threshold)
    plt.savefig(os.path.join(figDirectory, 'pie_plot_topology.cluster_%d.pdf'%cluster)); plt.close()
    plt.savefig(os.path.join(figDirectory, 'enrichment_topology.cluster_%d.pdf'%cluster))

