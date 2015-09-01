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
import IMlibs
import fitFun
sys.path.append('/home/sarah/array_image_tools_SKD/specific/')
#sys.path.append('/Users/Sarah/python/array_image_tools_SKD/specific/')
from mergeReplicates import errorPropagateAverage
import clusterFun
sns.set_style("white", {'lines.linewidth': 1})

parameters = fitFun.fittingParameters()
libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.txt'
#libCharFile = 'all_10expts.library_characterization.txt'

libChar = pd.read_table(libCharFile).loc[:, :'ss_correct']

#wcFile = 'flowWC.150607_150605.combined.results.pkl'
wcFile = '/lab/sarah/RNAarray/final_WC/flowWC.150607_150605.combined.results.pkl'    
results = pd.concat([libChar, pd.read_pickle(wcFile)], axis=1)
subresults = results.loc[results.sublibrary=='junction_conformations'].copy()
subresults.sort(['effective_length', 'offset'], inplace=True)
subresults.loc[:, 'id'] = ('wc10_' +
                           subresults.length.astype(int).astype(str) + '_' +
                           subresults.helix_one_length.astype(int).astype(str))
#subresults = subresults.loc[subresults.ss_correct.astype(bool)]

# also add 11bp data
wc11bpFile = '/lab/sarah/RNAarray/final_WC/flowWC_11bp.1506025.results.pkl'
results = pd.concat([libChar, pd.read_pickle(wc11bpFile)], axis=1)
subresults2 = results.loc[results.sublibrary=='junction_conformations'].copy()
subresults2.sort(['effective_length', 'offset'], inplace=True)
subresults2.loc[:, 'id'] = ('wc11_' +
                           subresults2.length.astype(int).astype(str) + '_' +
                           subresults2.helix_one_length.astype(int).astype(str))

newresults = pd.concat([subresults, subresults2])

junction_seq_ref = 'UGAUCU_AGAUCA'

# try with a table of just dG


pivot = newresults.pivot(index='junction_seq', columns='id', values='dG')
pivot = pivot.loc[:,['wc10_%s'%i for i in ['8_1', '9_0', '9_1','9_2','10_1',
                     '10_2','10_3','11_2','12_3',]]
                  +['wc11_%s'%i for i in ['9_0', '9_1','9_2','10_1',
                     '10_2','10_3','11_2']]]
pivot[pivot >= parameters.cutoff_dG] = parameters.cutoff_dG

index = newresults.groupby('junction_seq')['ss_correct'].all()

ddGs = (pivot - pivot.loc[junction_seq_ref]).loc[index].astype(float)
D = clusterFun.scrubInput(ddGs)


figDirectory = os.path.join(os.path.dirname(wcFile), 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)

# cluster?
numClusters = 15
labels, M = clusterFun.consensusCluster(D, subsample=0.8, n_samples=500, k=numClusters,
                                     plot=False, numCores=1)
labels, z = clusterFun.clusterHierarchical(M, numClusters, return_z=True)
cg = sns.clustermap(M, yticklabels=False, xticklabels=False, row_linkage=z,
                    col_linkage=z, figsize=(6,6))
seq_order = M.iloc[cg.dendrogram_row.reordered_ind].index
plt.savefig(os.path.join(figDirectory, 'heatmap_consensus.15_clusters.png'))

# plot heatmap of delta delta Gs
fig = plt.figure(figsize=(4,6));
plt.subplots_adjust(bottom=0.25, top=0.97, right=0.9, left=0.15)
ax = fig.add_subplot(111)
sns.heatmap(ddGs.loc[seq_order], yticklabels=False, ax=ax,
            cbar_kws={"label": "$\Delta \Delta G$"});
xlim = ax.get_xlim()
ylim = ax.get_ylim()

for cluster in np.unique(labels):
    ax.plot(xlim, [ylim[-1]-np.where(labels.loc[seq_order]==cluster)[0][-1]]*2,
            'k--', alpha=0.5, linewidth=0.5)
plt.savefig(os.path.join(figDirectory, 'heatmap_ddGs.15_clusters.png'))

# find fraction of each ss in each cluster
junction_ss = subresults.groupby('junction_seq').first().loc[labels.index].junction_SS

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

for cluster in np.unique(labels):
    ax.plot(xlim, [ylim[-1]-np.where(labels.loc[seq_order]==cluster)[0][-1]]*2,
            'k--', alpha=0.5, linewidth=0.5)
plt.savefig(os.path.join(figDirectory, 'heatmap.pvalues.15_clusters.png'))

cg_ss = sns.clustermap(fraction.astype(float), row_cluster=False, figsize=(5,4.5))
plt.subplots_adjust(bottom=.25, left=0.025, top=0.975)
plt.savefig(os.path.join(figDirectory, 'clustermap.fraction.15_clusters.png'))

# make matching heatmap
fig = plt.figure(figsize=(5,6))
plt.subplots_adjust(bottom=0.25, top=0.97, right=0.9, left=0.15)
ax = fig.add_subplot(111)
sns.heatmap(fraction.astype(float).loc[labels.loc[seq_order]].loc[:, order], ax=ax, yticklabels=False,
            cbar_kws={"label": "fraction in cluster"});
xlim = ax.get_xlim()
ylim = ax.get_ylim()

for cluster in np.unique(labels):
    ax.plot(xlim, [ylim[-1]-np.where(labels.loc[seq_order]==cluster)[0][-1]]*2,
            'k--', alpha=0.5, linewidth=0.5)
plt.savefig(os.path.join(figDirectory, 'heatmap.fraction.15_clusters.png'))

# find optimal number of clusters
cdfs = clusterFun.optimizeNumClusters(D, ks=np.arange(5, 25))