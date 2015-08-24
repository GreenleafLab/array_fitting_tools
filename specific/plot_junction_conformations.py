import datetime
import scipy.cluster.hierarchy as sch
import scipy.cluster as sc
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IMlibs
import fitFun
sys.path.append('/home/sarah/array_image_tools_SKD/specific/')
sys.path.append('/Users/Sarah/python/array_image_tools_SKD/specific/')
from mergeReplicates import errorPropagateAverage

parameters = fitFun.fittingParameters()
#libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.txt'
libCharFile = 'all_10expts.library_characterization.txt'

libChar = pd.read_table(libCharFile).loc[:, :'ss_correct']

wcFile = '/lab/sarah/RNAarray/final_WC/flowWC.150607_150605.combined.results.pkl'
wcFile = 'flowWC.150607_150605.combined.results.pkl'

results = pd.concat([libChar, pd.read_pickle(wcFile)], axis=1)

figDirectory = os.path.join(os.path.dirname(wcFile), 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)

subresults = results.loc[results.sublibrary=='junction_conformations'].copy()
subresults.sort(['effective_length', 'offset'], inplace=True)
#subresults = subresults.loc[subresults.ss_correct.astype(bool)]

junction_seq_ref = 'UGAUCU_AGAUCA'

# try with a table of just dG
subresults.loc[:, 'id'] = (subresults.length.astype(int).astype(str) + '_' +
                           subresults.helix_one_length.astype(int).astype(str))

pivot = subresults.pivot(index='junction_seq', columns='id', values='dG')
pivot = pivot.loc[:,['8_1', '9_0', '9_1','9_2','10_1',
                     '10_2','10_3','11_2','12_3',]]
pivot[pivot >= parameters.cutoff_dG] = parameters.cutoff_dG

ddGs = (pivot - pivot.loc[junction_seq_ref]).astype(float)

# cluster?
labels, M = consensusCluster(ddGs, n_samples=500, k=21,
                                     plot=False)


possible_junctions = np.unique(junctions)
fig = plt.figure(figsize=(5,7))
gs = gridspec.GridSpec(numClusters, 1)
for i, (name, group) in enumerate(labeled_junctions.groupby('class')):
    vec = group.junction.value_counts()
    x = np.searchsorted(possible_junctions, vec.index)
    y = vec.values
    ax = fig.add_subplot(gs[i])
    ax.bar(x, y)
    ax.set_xlim(0, len(possible_junctions))
    ax.set_xticks(np.arange(len(possible_junctions))+0.5)
    ax.set_xticklabels([])
    ax.tick_params(top='off', right='off')
ax.set_xticklabels(possible_junctions, rotation=90)
#plt.tight_layout()

