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
from collections import deque
import IMlibs
import fitFun
#sys.path.append('/home/sarah/array_image_tools_SKD/specific/')
sys.path.append('/Users/Sarah/python/array_image_tools_SKD/specific/')
from mergeReplicates import errorPropagateAverage

sys.path.append('/Users/Sarah/python/JunctionLibrary')
import hjh.junction
import hjh.tecto_assemble

sns.set_style("white", {'lines.linewidth': 1})

parameters = fitFun.fittingParameters()
#libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.txt'
libCharFile = 'all_10expts.library_characterization.txt'
libChar = pd.read_table(libCharFile)
results = pd.read_pickle('flowWC.150607_150605.combined.results.pkl')
variant_table = pd.concat([libChar, results.astype(float)], axis=1)
subtable = variant_table.loc[(libChar.sublibrary=='three_way_junctions')].copy()
subtable.loc[:, 'insertions'] = subtable.junction_seq.str.slice(2) 
subtable.loc[:, 'junction_seq']  = (subtable.flank.str.get(0) + '_' +
                                subtable.flank.str.get(1) + '_' +
                                subtable.flank.str.get(2))

loop = 'GGAA_UUCG'
g = sns.FacetGrid(subtable.loc[subtable.loop==loop], col="helix_one_length")
g.map(sns.distplot, "dG")


junction_type = 3
lengths = np.unique(subtable.helix_one_length)[:-1]
seqs = hjh.junction.Junction(('W', 'W', 'W')).sequences.side1
seqs =  seqs.iloc[:20]
#seqs = (seqs.str.get(0) + '_' + seqs.str.get(1) + '_' + seqs.str.get(2))
#seqs = ['A', 'U', 'G']
insertions = ['', '', 'A']

y = {}
for length in lengths:
    y[length] = {}
    done_seqs = []
    for seq in seqs:
        if np.all([i==seq[0] for i in seq]):
            print '\tSkipping junction %s because no permutations'%seq
        elif seq in done_seqs:
            print '\tJunction %s already done'%seq
        else:
            print 'doing junction %s'%seq
            
            y[length][seq] = pd.DataFrame(index=np.arange(junction_type),
                                          columns=['dG', 'seq', 'variant'])
            g = deque(np.arange(junction_type))
            for idx in np.arange(junction_type):
                
                ins = [insertions[i] for i in list(g)]
                seq_list = [seq[i] for i in list(g)]
                done_seqs.append(''.join(seq_list))
                index = ((subtable.helix_one_length == length)&
                         (subtable.junction_seq == '_'.join(seq_list))&
                         (subtable.insertions   == '_'.join(ins))&
                         (subtable.loop == loop))
                if index.sum() == 1:
                    y[length][seq].loc[idx, 'dG'] = subtable.loc[index].dG.values[0]
                    y[length][seq].loc[idx, 'variant'] = index.loc[index].index[0]
                y[length][seq].loc[idx, 'seq'] = '_'.join(seq_list)
                
                # rotate g
                g.rotate(1)
    y[length] = pd.concat(y[length])
y = pd.concat(y)

# make matrix holding bind constant
seqs = y.loc[3].seq
#numSeqs = len(seqs)
#confMat = np.identity(numSeqs)
#bindMat = np.vstack([np.repeat([vec], numSeqs, axis=0)
#                     for vec in [[1,0,0], [0,1,0], [0,0,1]]])

inds = ['bind_%d'%i for i in lengths] + seqs.tolist()
A = pd.DataFrame(0, index=y.index, columns=inds)
A.loc[3, 'bind_3'] = 1
A.loc[4, 'bind_4'] = 1
A.loc[5, 'bind_5'] = 1
for idx in y.index:
    A.loc[idx, y.loc[idx, 'seq']] = 1


# find indices that aren't NaN
x = pd.Series(index=inds)
row_index = (~y.dG.isnull())
col_index = ~(A.loc[row_index] == 0).all(axis=0)
x.loc[col_index] = np.linalg.lstsq(A.loc[row_index, col_index], y.loc[row_index].dG)[0]

# plot
plt.figure();
plt.scatter(np.dot(A.loc[row_index, col_index],
                   x.loc[col_index]),
            y.loc[row_index].dG, c=np.searchsorted(np.unique(y.loc[row_index].seq),
                                                   y.loc[row_index].seq))

# do fit twice
fit_all = pd.concat([x, x1]).groupby(level=0).mean()

# what is correlation with nearest neighbor?
nn = pd.read_table('~/python/JunctionLibrary/seq_params/nearest_neighbor_rules.txt',
                   index_col=0).astype(float)
colors = {'A':'g', 'C':'b', 'G':'k', 'U':'r'}

plt.figure()
indices = ['']*len(nn)
for i, di in enumerate(nn.index):
    seq_list = list(di)
    for base in ['A', 'C', 'G', 'U']:
        index = '_'.join([seq_list[0], base, seq_list[1]])
        if index in fit_all.index:
            plt.scatter(nn.loc[di], fit_all.loc[index], c=colors[base])
        
    indices[i] = ['_'.join([seq_list[0], base, seq_list[1]]) for base in ['A', 'C', 'G', 'U']]
    
plt.figure()
deltaG_pred = pd.Series(index = fit_all.index[:-3])
for idx in deltaG_pred.index:
    seq_list = idx.split('_')

    seq1 = ''.join([seq_list[0], seq_list[2]]).replace('T', 'U')
    seq2 = ''.join([seq_list[0], seq_list[1]]).replace('T', 'U')
    seq3 = ''.join([seqfun.reverseComplement(seq_list[2]), seq_list[1]]).replace('T', 'U')
    deltaG_pred.loc[idx] = (nn.loc[seq1] - nn.loc[seq2] - nn.loc[seq3]).values[0]
    plt.scatter(fit_all.loc[idx], deltaG_pred.loc[idx], c=np.searchsorted(nn.index.tolist(), seq1),
                cmap = 'Set1', vmin=0, vmax=15)
    
    

    

pd.DataFrame(fit_all, columns=['dG', 'stack'])


# what is fraction in each conformer?
dG_conf = y.loc[3, ['seq']].copy()
dG_conf.loc[:, 'dG'] = np.nan
dG_conf.loc[:, 'frac'] = np.nan
for seq in x.index[3:]:
    dG_conf.loc[dG_conf.seq==seq, 'dG'] = x.loc[seq]
    dG_conf.loc[dG_conf.seq==seq, 'frac'] = np.exp(-x.loc[seq]/parameters.RT)

dG_conf.iloc[np.searchsorted(dG_conf.seq, x.index[3:]), 'dG'] = x.loc[x.index[3:]]
y.loc[y.seq==x.index]