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

parameters = variantFun.Parameters()


flowpiece = 'rigid'
name = 'binding_curves'
fittedBindingFilename = '%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPfitted'%(name, flowpiece) 
#table = IMlibs.loadFittedCPsignal(fittedBindingFilename, index_by_cluster=True)
#table = table.sort_index(axis=0).sort('variant_number')

variantFittedFilename = '%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'%(name, flowpiece)
variant_table = pd.read_table(variantFittedFilename)

saveDirectory = os.path.join('linear_model', str(datetime.date.today()))
if not os.path.exists(saveDirectory): os.mkdir(saveDirectory)
# set filters on variants to actually fit
cutoff = pd.Series()
cutoff['dG_error'] = 0.5 # kcal/mol
cutoff['qvalue']  = 0.05
cutoff['numTests'] = 2
cutoff['total_length'] = 10
cutoff['receptor'] = 'R1'
cutoff['loop'] = 'goodLoop'

indxFit = pd.Series()
indxFit['dG_error'] = (variant_table.dG_ub - variant_table.dG_lb).values < cutoff['dG_error']
indxFit['qvalue']   = variant_table.qvalue.values <= cutoff['qvalue']
indxFit['numTests'] = variant_table.numTests.values >= cutoff['numTests']
for param in ['total_length', 'receptor', 'loop']:
    indxFit[param] = variant_table[param].values == cutoff[param]


numVariantsEach  = np.sum(np.vstack(indxFit.values), axis=1)
numVariantsTotal = np.sum(np.all([indxFit[name] for name in indxFit.index], axis=0))

print numVariantsEach
print numVariantsTotal

# save table format
helixLengthTotal = cutoff['total_length']
indexParam = np.arange(helixLengthTotal - 1)

# print expected spead in the data
subset_index=np.all([indxFit[name] for name in indxFit.index], axis=0)
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
histogram.compare([variant_table.loc[subset_index, 'dG_ub'].values -
                   variant_table.loc[subset_index, 'dG_lb'].values],
    xbins =np.linspace(-0.05, 1.1, 50), bar=True, normalize=False)
ax.set_xlabel('Confidence interval width (kcal/mol)')
ax.set_ylabel('Number of variants')
ax.tick_params( direction='out', top='off', right='off')
ax.legend_ = None
plt.tight_layout()
plt.savefig(os.path.join(saveDirectory, 'histogram.confidence_intervals.below_cutoff.flowpiece_%s.length_%dbp.pdf'%(flowpiece,helixLengthTotal) ))

# estimated sigma pervariant = CI width /(2*1.96)
variant_table.loc[:, 'sigma'] = (variant_table.loc[:, 'dG_ub'] - variant_table.loc[:, 'dG_lb'])/(2*1.96)
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
histogram.compare([variant_table.loc[subset_index, 'sigma']],
    xbins =np.linspace(-0.01, 0.5, 50), bar=True, normalize=False)
ax.set_xlabel('measurement stdev (kcal/mol)')
ax.set_ylabel('Number of variants')
ax.tick_params( direction='out', top='off', right='off')
ax.legend_ = None
plt.tight_layout()
plt.savefig(os.path.join(saveDirectory, 'histogram.sigma.below_cutoff.flowpiece_%s.length_%dbp.pdf'%(flowpiece,helixLengthTotal) ))

# Predict secondary structure, parse into dibases, save parameters for each dibase
indx_wt = 1 # WC
dibases_wt = fitFun.getWildtypeDibases(variant_table, indx_wt)
variant_subtable = variant_table.loc[subset_index]
numCores = 20
workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
#table = [fitFun.multiprocessParametrization(variant_table, dibases_wt, indx) for indx in variant_subtable.index[200:1000]]
tableList = workerPool.map(functools.partial(fitFun.multiprocessParametrization, variant_table, dibases_wt), variant_subtable.index)
workerPool.close(); workerPool.join()
table = pd.DataFrame.from_records(tableList, index=variant_subtable.index).dropna(axis=(0,1), how='all')

# get a sense of how many variants there are per parameter
min_num_variants_to_fit = 200
num_params = table.shape[1]
num_tests = float(table.shape[0])
plt.figure(figsize=(7, 4))
plt.bar(np.arange(num_params), table.sum(axis=0)/num_tests )
plt.xticks(np.arange(num_params)+0.5, [name for name in table], rotation=90)
plt.tick_params(direction='out', top='off', right='off')
plt.plot([0, num_params+1], [min_num_variants_to_fit/num_tests, min_num_variants_to_fit/num_tests], 'k:')
plt.ylim((0, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(saveDirectory, 'number_variants_for_each_param.flowpiece_%s.length_%dbp.pdf'%(flowpiece,helixLengthTotal)))

# find parameters with few variants
min_num_variants_to_fit = 10
bad_params = table.sum(axis=0) < min_num_variants_to_fit

# find variants that have no variation in the bad parameters 
good_variants = table.loc[:,bad_params].sum(axis=1) == 0

# remove bad params and keep good variants
tableFinal = table.loc[good_variants, np.logical_not(bad_params)]
tableFinal.loc[:, 'ddG'] = variant_table.loc[tableFinal.index, 'dG'] - variant_table.loc[indx_wt, 'dG']
index = variant_subtable.loc[tableFinal.index].sort('topology').index
tableFinal.loc[index].to_csv(os.path.join(saveDirectory, 'param.flowpiece_%s.%dbp.mat'%(flowpiece, helixLengthTotal)), sep='\t', index=True)

# also remove variants that have a bluge of length 3 at position 7k
#tableFinal = tableFinal.loc[tableFinal.loc[:,'insertions_k_7'] <= 2]
# Save test set and training set

# sort by topology and take every other
index = variant_subtable.loc[tableFinal.index].sort('topology').index

indexTest = index[np.arange(0, len(index), 2)]
indexTraining =  index[np.arange(1, len(index), 2)]

tableFinal.loc[indexTraining].to_csv(os.path.join(saveDirectory, 'param.flowpiece_%s.%dbp.training.mat'%(flowpiece, helixLengthTotal)), sep='\t', index=True)
tableFinal.loc[indexTest].to_csv(os.path.join(saveDirectory, 'param.flowpiece_%s.%dbp.test.mat'%(flowpiece, helixLengthTotal)), sep='\t', index=True)

# load and plot results from R

a = np.loadtxt(os.path.join(saveDirectory, 'param_est.flowpiece_%s.%dbp.txt'%(flowpiece,helixLengthTotal) ), skiprows=14, dtype=str, delimiter='\n')[:-5]
a = np.vstack([row.replace('< ', '').strip().strip('*').strip('.').split() for row in a])
fitParams = pd.DataFrame(data=a, columns=['parameter', 'estimate', 'stderr', 'tvalue', 'p'])
fitParams.loc[fitParams.estimate == 'NA', ['estimate', 'stderr', 'tvalue', 'p']] = np.nan
fitParams.loc[0, 'parameter'] = 'intercept'
fitParams = pd.DataFrame(data=fitParams.iloc[:,1:].astype(float).values,
                         index=fitParams.parameter,
                         columns = ['estimate', 'stderr', 'tvalue', 'p'],
                         dtype=float)

plt.figure()
plt.bar(np.arange(1,len(fitParams)), fitParams.estimate.iloc[1:])
plt.xticks(np.arange(1,len(fitParams))+0.5,fitParams.index[1:], rotation=90, fontsize=10)
plt.tick_params(direction='out', top='off', right='off')

plt.ylabel('dG (kcal/mol)')
plt.tight_layout()
plt.savefig(os.path.join(saveDirectory,'parameters.pdf'))

# plot individual terms
fitParamsParsed = fitFun.convertFitParamsToMatrix(fitParams, indexParam)
params = fitParamsParsed.index.levels[0]
cNorm  = colors.Normalize(vmin=0, vmax=len(params))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='coolwarm')
plt.figure(figsize=(6.5,4))
ax = plt.gca()
for i, name in enumerate(params):
    color = scalarMap.to_rgba(i)
    if name.find('insertions') > -1:
        offset = 0.5
    else:
        offset = 0
    xvalues = fitParamsParsed.loc[name].index + offset
    yvalues = fitParamsParsed.loc[name].estimate
    yerr = fitParamsParsed.loc[name].stderr
    ax.errorbar(xvalues, yvalues,yerr, fmt='o-', label=name, color=color, ecolor='k')
ax.plot([1, 8], [0,0], 'k:')
ax.set_xlim((1.75, 8))
ax.set_ylim((-1.1, 1.5))
plt.xticks(np.arange(2, 9), (np.arange(1, 8)-1)[::-1])
ax.set_xlabel('bp from loop')
ax.set_ylabel('dG (kcal/mol)')
ax.tick_params(direction='out', top='off', right='off')
plt.tight_layout()

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left=0.15, right=0.55)
plt.savefig(os.path.join(saveDirectory,'parameters_plot.errorbars.flowpiece_%s.length_%dbp.pdf'%(flowpiece, helixLengthTotal)))

break_ind = [(int(name.strip('1')[-1]), name) for name in fitParams.index if name.find('bp_break')>-1 and not name.find(':') >-1]
nc_ind = [(int(name.strip('1')[-1]), name) for name in fitParams.index if name.find('bp_break')>-1 and not name.find(':') >-1]
# plot individual terms
nc_ind = [name for name in fitParams.index if name.find('nc_j')>-1]
break_ind = [name for name in fitParams.index if name.find('bp_break_j')>-1 and not name.find('insertions') >-1]
insertion1k_ind = [name for name in fitParams.index if name.find('insertions_k1')>-1 and not name.find('bp_break')>-1]
insertion2k_ind = [name for name in fitParams.index if name.find('insertions_k2')>-1 and not name.find('bp_break')>-1]
insertion3k_ind = [name for name in fitParams.index if name.find('insertions_k3')>-1 and not name.find('bp_break')>-1]

insertion1z_ind = [name for name in fitParams.index if name.find('insertions_z1')>-1 and not name.find('bp_break')>-1]
insertion2z_ind = [name for name in fitParams.index if name.find('insertions_z2')>-1 and not name.find('bp_break')>-1]
insertion3z_ind = [name for name in fitParams.index if name.find('insertions_z3')>-1 and not name.find('bp_break')>-1]

allinds = {'noncan bp formed':nc_ind,
           'bp broken':break_ind,
           '1 base insertion top':insertion1k_ind,
           '2 base insertion top':insertion2k_ind,
           '3 base insertion top':insertion3k_ind,
           '1 base insertion bottom':insertion1z_ind,
           '2 base insertion bottom':insertion2z_ind,
           '3 base insertion bottom': insertion3z_ind}

keys = ['noncan bp formed',
           'bp broken',
           '1 base insertion top',
           '2 base insertion top',
           '3 base insertion top',
           '1 base insertion bottom',
           '2 base insertion bottom',
           '3 base insertion bottom']

cNorm  = colors.Normalize(vmin=0, vmax=len(allinds))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='coolwarm')
plt.figure(figsize=(6.5,4))
ax = plt.gca()
for i, label in enumerate(keys):
    ind = allinds[label]
    color = scalarMap.to_rgba(i)
    if i >=2: offest=-0.5
    else: offest=0
    ax.errorbar(np.array([name[1] for name in ind]).astype(float)+offest,
                fitParams.loc[ind, 'estimate'], yerr=fitParams.loc[ind, 'stderr'], fmt='o-', label=label, color=color, ecolor='k')
ax.plot([1, 8], [0,0], 'k:')
ax.set_xlim((1.75, 8))
ax.set_ylim((-1.1, 1.5))
plt.xticks(np.arange(2, 9), (np.arange(1, 8)-1)[::-1])
ax.set_xlabel('bp from loop')
ax.set_ylabel('dG (kcal/mol)')
ax.tick_params(direction='out', top='off', right='off')
plt.tight_layout()

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left=0.15, right=0.55)
plt.savefig(os.path.join(saveDirectory,'parameters.errorbars.pdf'))

# load predicted values for dG

predicted_dGs = pd.read_table('linear_model/2015-01-31/predicted_dGs.test.txt', index_col=0, header=0, sep=' ')
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
im = ax.scatter(tableTest.loc[predicted_dGs.index, 'dG'], predicted_dGs, alpha=0.25, edgecolor='0.1',
           c=variant_table.loc[predicted_dGs.index,'junction_length'], cmap='coolwarm')
ax.set_xlabel('actual dG (kcal/mol)')
ax.set_ylabel('predicted dG (kcal/mol)')
ax.tick_params(direction='out', top='off', right='off')
plt.xticks([-11,-10.5, -10, -9.5 ,-9, -8.5, ], ['-11', '', '-10', '', '-9', '', ])
plt.yticks([-11,-10.5, -10, -9.5 ,-9, -8.5, ], ['-11', '', '-10', '', '-9', '', ])
ax.set_xlim((-11.5, -8.5))
ax.set_ylim((-11.5, -8.5))
ax.grid()
plt.tight_layout()
plt.colorbar(im)
plt.savefig(os.path.join(saveDirectory,'scatterplot.test.actual_vs_predicted.pdf'))


# check what's going horribly wrong
filenames = ['linear_model/param.wc.10bp.%s.mat'%'training', 'linear_model/param.wc.10bp.%s.mat'%'test']
tableOld = pd.concat([pd.read_table(filenames[0], index_col=0), pd.read_table(filenames[1], index_col=0)])
tableOld.sort_index(inplace=True)
# save exact same rows to new file and fit that one
tableFinal.loc[tableOld.index].to_csv(os.path.join(saveDirectory, 'param.flowpiece_%s.%dbp.fed.mat'%(flowpiece, helixLengthTotal)), sep='\t', index=True)

# saving same rows doesn't help (R2=0.42 for old, 0.38 for new). Now see if parameterization AND rows are the same
paramDict = {}
for i in [2, 3, 4, 5, 6, 7]:
    for param in ['bp_break', 'nc']:
        paramDict['%d_%s_j'%(i, param)] = '%s_%d'%(param,i+1)
    for param in ['insertions_k', 'insertions_z']:
        paramDict['%d_%s'%(i, param)] = '%s_%d'%(param, i)

for key in paramDict:
    if key in [name for name in tableOld] and paramDict[key] in [name for name in tableFinal]:
       print key, (tableFinal.loc[tableOld.index, paramDict[key]] != tableOld.loc[:, key]).sum()
    else:
        if key not in [name for name in tableOld]:
            print '\t%s not in old table'%key
        if paramDict[key] not in [name for name in tableFinal]:
            print '\t%s not in new table'%key
# of the ones that exist, they are the same. Not error in parsing

# try fit on the exact same parameters and rows. See if it's an issue in fitting
keys_old = np.intersect1d(paramDict.keys(), [name for name in tableOld])

keys_diff = np.setdiff(paramDict.keys(), [name for name in tableOld])
keys_new = [paramDict[key] for key in keys_old] + ['dG']
tableFinal.loc[tableOld.index, keys_new].to_csv(os.path.join(saveDirectory, 'param.flowpiece_%s.%dbp.fed.mat'%(flowpiece, helixLengthTotal)), sep='\t', index=True)


filename = 'linear_model/2015-02-11/param.flowpiece_rigid.10bp.mat'
tableOld = pd.read_table(filename, index_col=0)

# trouble shooting
tableOld = tableFinal.copy()
keyDict = fitFun.makeCategoricalHeaders()
cols = np.hstack([keyDict[name] for name in tableOld if not name == 'ddG'])

tableFinal = table.loc[tableOld.index, cols]
tableFinal.dropna(axis=0, how='any', inplace=True)
tableFinal.loc[:, 'ddG'] = variant_table.loc[tableFinal.index, 'dG'] - variant_table.loc[indx_wt, 'dG']
index = variant_subtable.loc[tableFinal.index].sort('topology').index
tableFinal.loc[index].to_csv(os.path.join(saveDirectory, 'param.flowpiece_%s.%dbp.mat'%(flowpiece, helixLengthTotal)), sep='\t', index=True)

# some of these only have one variant!
min_num_variants_to_fit = 10
table = table.loc[tableOld.index, cols]
bad_params = table.sum(axis=0) < min_num_variants_to_fit

# find variants that have no variation in the bad parameters 
good_variants = table.loc[:,bad_params].sum(axis=1) == 0

# remove bad params and keep good variants
tableFinal = table.loc[good_variants, np.logical_not(bad_params)]
