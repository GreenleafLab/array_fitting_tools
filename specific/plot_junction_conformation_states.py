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
pvalues = pd.read_pickle('final_WC/consensus_mat.21_clusters.subsample_0.5.500_samples.pvalues.pkl')
fraction = pd.read_pickle('final_WC/consensus_mat.21_clusters.subsample_0.5.500_samples.fraction.pkl')

# dGs
index_ss = newresults.groupby('junction_seq')['ss_correct'].all()
pivot = newresults.pivot(index='junction_seq', columns='id', values='dG')
cols = (['wc10_%s'%i for i in ['8_1', '9_0', '9_1','9_2','10_1','10_2','10_3','11_2','12_3',]]
       +['wc11_%s'%i for i in ['9_0', '9_1','9_2','10_1','10_2','10_3','11_2']])
dGs = pivot.loc[index_ss,cols].astype(float)

pivot[pivot >= parameters.cutoff_dG] = parameters.cutoff_dG
ddGs = (pivot - pivot.loc[junction_seq_ref]).loc[index_ss, cols].astype(float)

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
for cluster in clusters:
    index = labels.loc[labels==cluster].index
    tectoPlots.plotChevronPlot3(dGs, index=index, flow_id='wc10')
    plt.savefig(os.path.join(figDirectory, 'chevron_plots2.wc_10.cluster_%d.pdf'%cluster))
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

