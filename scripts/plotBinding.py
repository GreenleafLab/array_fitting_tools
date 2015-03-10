#!/usr/bin/env python

# Methods for plotting binding curves by variant number, etc
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
import variantFun
import IMlibs
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import functools
import itertools
import seaborn as sns

import generateVariantNumber
parameters = variantFun.Parameters()

flowpiece = 'rigid'
figDirectory = os.path.join('flow_%s'%flowpiece, 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory): os.makedirs(figDirectory)

affinityCol = 0; offRateCol = 1; onRateCol=2
fittedBindingFilenames = ['%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPfitted'%(name, flowpiece) for name in ['binding_curves', 'off_rates', 'on_rates']]
table_affinity = IMlibs.loadFittedCPsignal(fittedBindingFilenames[affinityCol], index_by_cluster=True)
table_affinity = table_affinity.sort_index(axis=0).sort('variant_number')
table_offrates = IMlibs.loadFittedCPsignal(fittedBindingFilenames[offRateCol], index_by_cluster=True)
table_offrates = table_offrates.sort_index(axis=0).sort('variant_number')
timeFilename = 'off_rates_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.times'%flowpiece
times = pd.read_table(timeFilename, index_col=0)

# Make a new table with both dG, and toff, koff and kon
fixed_ind = np.where([name=='total_length' for name in table_affinity])[0][0] + 1
seqinfoCols  = [name for name in table_affinity][:fixed_ind]
affinityCols = [name for name in table_affinity][fixed_ind:]
offratesCols = [name for name in table_offrates][fixed_ind:]
if flowpiece == 'wc':
    onratesCols  = [name for name in table_offrates][fixed_ind:]
# concat all together
# concat all together 
pieces = {'seqinfo':  table_affinity.loc[:,seqinfoCols],
          'affinity': table_affinity.loc[:,affinityCols],
          'offrates': table_offrates.loc[:,offratesCols],}
#if flowpiece == 'wc':
#    pieces['onrates'] = variant_table_onrates.loc[:,onratesCols]
table = pd.concat(pieces, axis=1)

# load the reduced variant tables as well
variantFittedFilename = 'off_rates_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'%flowpiece
variant_table_offrates = pd.read_table(variantFittedFilename)

variantFittedFilename = 'binding_curves_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'%flowpiece
variant_table_affinity = pd.read_table(variantFittedFilename)

if flowpiece == 'wc':
    variantFittedFilename = 'on_rates_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'%flowpiece
    variant_table_onrates = pd.read_table(variantFittedFilename)

# Make a new table with both dG, and toff, koff and kon
fixed_ind = 12
seqinfoCols  = [name for name in variant_table_affinity][:fixed_ind]
affinityCols = [name for name in variant_table_affinity][fixed_ind:]
offratesCols = [name for name in variant_table_offrates][fixed_ind:]
if flowpiece == 'wc':
    onratesCols  = [name for name in variant_table_onrates][fixed_ind:]
notnaninds = np.arange(len(variant_table_affinity))[np.logical_not(np.isnan(variant_table_affinity.loc[:,'dG']).values)]

variant_table_summary = pd.DataFrame(index = variant_table_offrates.index, columns=['dG', 'koff', 'kon'])
variant_table_summary.loc[notnaninds, 'dG'] = variant_table_affinity.loc[notnaninds, 'dG'].values
variant_table_summary.loc[notnaninds, 'kd'] = variantFun.find_Kd_from_dG(variant_table_affinity.loc[notnaninds, 'dG'].values)/parameters.concentration_units
variant_table_summary.loc[notnaninds, 'koff'] = 1/variant_table_offrates.loc[notnaninds, 'toff'].values
variant_table_summary.loc[notnaninds, 'kon'] = 1/(variantFun.find_Kd_from_dG(variant_table_affinity.loc[notnaninds, 'dG'].values)*variant_table_offrates.loc[notnaninds, 'toff'].values)

variant_table_summary.loc[notnaninds, 'kobs_est'] = variant_table_summary.loc[notnaninds, 'koff'] + variant_table_summary.loc[notnaninds, 'kon']*2.73E-9
if flowpiece == 'wc':
    variant_table_summary.loc[notnaninds, 'kobs'] = 1/variant_table_onrates.loc[notnaninds, 'ton'].values

# propagate error
variant_table_summary.loc[notnaninds, 'kd_ub'] = variant_table_summary.loc[notnaninds, 'kd'] + variant_table_summary.loc[notnaninds, 'kd']*(variant_table_affinity.loc[notnaninds, 'dG_ub'].values -variant_table_affinity.loc[notnaninds, 'dG'].values)
variant_table_summary.loc[notnaninds, 'kd_lb'] = variant_table_summary.loc[notnaninds, 'kd'] - variant_table_summary.loc[notnaninds, 'kd']*(variant_table_affinity.loc[notnaninds, 'dG'].values -variant_table_affinity.loc[notnaninds, 'dG_lb'].values)

variant_table_summary.loc[notnaninds, 'koff_ub'] = variant_table_summary.loc[notnaninds, 'koff'] + variant_table_summary.loc[notnaninds, 'koff']*(variant_table_offrates.loc[notnaninds, 'toff_ub'].values - variant_table_offrates.loc[notnaninds, 'toff'].values)/variant_table_offrates.loc[notnaninds, 'toff'].values
variant_table_summary.loc[notnaninds, 'koff_lb'] = variant_table_summary.loc[notnaninds, 'koff'] - variant_table_summary.loc[notnaninds, 'koff']*(variant_table_offrates.loc[notnaninds, 'toff'].values - variant_table_offrates.loc[notnaninds, 'toff_lb'].values)/variant_table_offrates.loc[notnaninds, 'toff'].values

variant_table_summary.loc[notnaninds, 'kon_ub'] = variant_table_summary.loc[notnaninds, 'kon'] + variant_table_summary.loc[notnaninds, 'kon']*np.sqrt(np.power((variant_table_summary.loc[notnaninds, 'kd_ub'] - variant_table_summary.loc[notnaninds, 'kd'])/variant_table_summary.loc[notnaninds, 'kd'], 2) +
                                                                                                                                                      np.power((variant_table_offrates.loc[notnaninds, 'toff_ub'] - variant_table_offrates.loc[notnaninds, 'toff'])/variant_table_offrates.loc[notnaninds, 'toff'], 2))
variant_table_summary.loc[notnaninds, 'kon_lb'] = variant_table_summary.loc[notnaninds, 'kon'] - variant_table_summary.loc[notnaninds, 'kon']*np.sqrt(np.power((variant_table_summary.loc[notnaninds, 'kd'] - variant_table_summary.loc[notnaninds, 'kd_lb'])/variant_table_summary.loc[notnaninds, 'kd'], 2) +
                                                                                                                                                      np.power((variant_table_offrates.loc[notnaninds, 'toff'] - variant_table_offrates.loc[notnaninds, 'toff_lb'])/variant_table_offrates.loc[notnaninds, 'toff'], 2))


# concat all together 
pieces = {'seqinfo':  variant_table_affinity.loc[:,seqinfoCols],
          'affinity': variant_table_affinity.loc[:,affinityCols],
          'offrates': variant_table_offrates.loc[:,offratesCols],
          'summary': variant_table_summary}
if flowpiece == 'wc':
    pieces['onrates'] = variant_table_onrates.loc[:,onratesCols]

variant_table = pd.concat(pieces, axis=1)
#variant_table.dropna(axis=0, subset=['summary'], inplace=True, how='any')

lims = {'qvalue':(2e-4, 1e0),
        'dG':(-12, -4),
        'koff':(1e-5, 1E-2),
        'kon' :(1e-2, 1e5),
        'kobs':(1e-5, 1e-2)}

# plot average error
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
histogram.compare([variant_table.affinity.numTests],
    xbins = np.arange(-0.5, 45.5), bar=True)
ax.set_xlabel('Number of measurements per variant')
ax.set_ylabel('Fraction of variants')
ax.set_xlim((0, 40))
ax.tick_params( direction='out', top='off', right='off')
ax.legend_ = None
plt.tight_layout()
plt.savefig(os.path.join(figDirectory,'number_measurements_per_variant.dG.pdf'))


# plot average error
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
histogram.compare([variant_table['affinity'].dG_ub.values - variant_table['affinity'].dG_lb.values],
    xbins = np.linspace(-1, 10, 100), bar=True, normalize=False)
ax.set_xlabel('Confidence interval width (kcal/mol)')
ax.set_ylabel('Number of variants')
ax.set_xlim((0, 7))
ax.tick_params( direction='out', top='off', right='off')
ax.legend_ = None
plt.tight_layout()
plt.savefig(os.path.join(figDirectory,'confidence_interval_width.dG.pdf'))

# plot average error
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
histogram.compare([(variant_table['affinity'].loc[variant_table.affinity.qvalue<0.05, 'dG_ub'].values - variant_table['affinity'].loc[variant_table.affinity.qvalue<0.05, 'dG_lb'].values)/(1*1.96)],
    xbins = np.linspace(-1, 10, 100), bar=True, normalize=False)
ax.set_xlabel('standard deviation (kcal/mol)')
ax.set_ylabel('Number of variants')
ax.set_xlim((0, 2))
ax.tick_params( direction='out', top='off', right='off')
ax.legend_ = None
plt.tight_layout()
plt.savefig(os.path.join(figDirectory,'sigma.above_qvalue.dG.pdf'))

# plot average error
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
histogram.compare([(variant_table['affinity'].loc[:, 'dG_ub'].values - variant_table['affinity'].loc[:, 'dG_lb'].values)/(1*1.96)],
    xbins = np.linspace(-1, 10, 100), bar=True, normalize=False)
ax.set_xlabel('standard deviation (kcal/mol)')
ax.set_ylabel('Number of variants')
ax.set_xlim((0, 7))
ax.tick_params( direction='out', top='off', right='off')
ax.legend_ = None
plt.tight_layout()
plt.savefig(os.path.join(figDirectory,'sigma.all.dG.pdf'))

# plot per junction info
seq = 'AA_AA'
helix_context = 'rigid'
loop = 'goodLoop'
offset=1
topology = 'M_B1'
variants = variantFun.findVariantNumbers(table.seqinfo, {'helix_context':helix_context, 'loop':loop,
                                                         'receptor':'R1', 'helix_one_length':4, 'topology':junction_topology, 'total_length':10})
successes = np.zeros(len(variants), dtype=bool)
for i, variant in enumerate(variants):
    dibases, successes[i] = fitFun.parseSecondaryStructure(variant_table.seqinfo.loc[variant, 'sequence'], variant_table.seqinfo.loc[variant, 'total_length'])
    
variant_subtable = pd.concat([variant_table['seqinfo'].loc[variants[successes]], variant_table['affinity'].loc[variants[successes]]], axis=1)
variantFun.plot_over_coordinate(variant_subtable); plt.close()
plt.savefig(os.path.join(figDirectory, 'junction_seq_%s.flowpiece_%s.helix_%s.%s.pdf'%(seq, flowpiece, helix_context, loop)))

figDirectoryPer = os.path.join(figDirectory,  'junction_%s'%seq)
if not os.path.exists(figDirectoryPer): os.mkdir(figDirectoryPer)

variant_subtable = variant_table.loc[variants]
variantFun.plot_over_coordinate(pd.concat([variant_subtable['seqinfo'], variant_subtable['affinity']], axis=1))
plt.savefig(os.path.join(figDirectoryPer, 'dG.scatterplot.pdf'))
param1 = 'koff'
param2 = 'kon'
variantFun.plot_scatterplot(pd.concat([variant_subtable['seqinfo'], variant_subtable['offrates'], variant_subtable['summary']], axis=1), param1, param2)
plt.savefig(os.path.join(figDirectoryPer, 'off_rates.scatterplot.pdf'))
for variant in variants:
    # binding curve
    subtable = table.loc[table['seqinfo']['variant_number']==variant]
    ax, concentrationsAll = variantFun.plotBindingCurveVariant(subtable['affinity'], parameters.concentrations, plotAllTraces=False)
    ax.plot(concentrationsAll, variantFun.bindingCurve(concentrationsAll, variant_table['affinity'].loc[variant, 'dG_ub']), 'r:')
    ax.plot(concentrationsAll, variantFun.bindingCurve(concentrationsAll, variant_table['affinity'].loc[variant, 'dG_lb']), 'r:')
    ax.fill_between(concentrationsAll, variantFun.bindingCurve(concentrationsAll, variant_table['affinity'].loc[variant, 'dG_ub']),
                    variantFun.bindingCurve(concentrationsAll, variant_table['affinity'].loc[variant, 'dG_lb']), facecolor='0.5', alpha=0.1)
    title = variantFun.getInfo(variant_table['seqinfo'].loc[variant])
    plt.title(title, fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(figDirectory, 'binding_curves.q_%4.2f.dG_%4.1f.variant_%s.pdf'%(variant_table['affinity'].loc[variant, 'qvalue'],
                                                                                             variant_table['affinity'].loc[variant, 'dG'],
                                                                                             title)))
    # histogram dG
    param = 'dG'
    ymax=5
    variantFun.plotHistogram(subtable['affinity'].loc[subtable['affinity'].fraction_consensus >=80], param)
    ax = plt.gca();
    ymax = np.max([ax.get_ylim()[-1], ymax])
    ax.set_ylim((0, ymax))
    plt.title(title, fontsize=10); plt.tight_layout()
    ax.fill_between([variant_table['affinity'].loc[variant, 'dG_lb'], variant_table['affinity'].loc[variant, 'dG_ub']], 0, ymax, color='0.25', alpha=0.1)
    ax.plot([variant_table['affinity'].loc[variant, 'dG'], variant_table['affinity'].loc[variant, 'dG']], [0, ymax], 'k:', linewidth=2)
    plt.savefig(os.path.join(figDirectory, 'histogram.dG.q_%4.2f.dG_%4.1f.variant_%s.pdf'%(variant_table['affinity'].loc[variant, 'qvalue'],
                                                                                             variant_table['affinity'].loc[variant, 'dG'],
                                                                                             title)))
    # off rate
    ax, timeBinCenters = variantFun.plotOffRateVariant(subtable['offrates'], times, errorbar=False)
    ax.plot(timeBinCenters, variantFun.offRateCurve(timeBinCenters, variant_table['offrates'].loc[variant, 'toff_lb']), 'r:')
    ax.plot(timeBinCenters, variantFun.offRateCurve(timeBinCenters, variant_table['offrates'].loc[variant, 'toff_ub']), 'r:')
    title = variantFun.getInfo(variant_table['seqinfo'].loc[variant])
    plt.title(title,fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(figDirectory, 'off_rates.q_%4.2f.toff_%4.1f.variant_%s.pdf'%(variant_table['offrates'].loc[variant, 'qvalue'],
                                                                                             variant_table['offrates'].loc[variant, 'toff'],
                                                                                             title)))
    # histogram toff
    param = 'toff'
    ymax=5
    variantFun.plotHistogram(subtable['offrates'], param)
    ax = plt.gca();
    ymax = np.max([ax.get_ylim()[-1], ymax])
    ax.set_ylim((0, ymax))
    ax.fill_between(np.log10([variant_table['offrates'].loc[variant, param+'_lb'], variant_table['offrates'].loc[variant, param+'_ub']]), 0, ymax, color='0.25', alpha=0.1)
    ax.plot(np.log10([variant_table['offrates'].loc[variant, param], variant_table['offrates'].loc[variant, param]]), [0, ymax], 'k:', linewidth=2)
    
    plt.title(title, fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(figDirectory, 'histogram.toff.q_%4.2f.toff_%4.1f.variant_%s.pdf'%(variant_table['offrates'].loc[variant, 'qvalue'],
                                                                                             variant_table['offrates'].loc[variant, 'toff'],
                                                                                             title)))
# koff contribution
index = variant_table['offrates'].loc[:, 'qvalue'] < 0.005
koff_rel = variant_table['summary'].loc[0, 'koff']
kon_rel = variant_table['summary'].loc[0, 'kon']
dg_rel = variant_table['summary'].loc[0, 'dG']

variant_subtable = (variant_table.loc[:, 'summary'].loc[:, ['koff', 'kon', 'dG', 'koff_lb', 'koff_ub', 'kon_lb', 'kon_ub'] ]).astype(float)
variant_subtable.loc[:, 'dG_lb'] = variant_table.loc[:, 'affinity'].loc[:, 'dG_lb'].astype(float)
variant_subtable.loc[:, 'dG_ub'] = variant_table.loc[:, 'affinity'].loc[:, 'dG_ub'].astype(float)

delta_koff_mut = parameters.RT*np.log(variant_subtable.loc[index, 'koff']/koff_rel)
delta_kon_mut = parameters.RT*np.log(variant_subtable.loc[index, 'kon']/kon_rel)
delta_delta_g = variant_subtable.loc[index, 'dG'] - dg_rel
histogram.compare([delta_koff_mut[delta_delta_g!=0]/delta_delta_g[delta_delta_g!=0]])

# plot scatterplot
plt.figure(figsize = (4,4))
plt.scatter( delta_delta_g,delta_koff_mut, alpha=0.1, linewidth=0)
plt.tick_params(direction="out", right='off', top='off')
plt.xlim((-0.5, 1.5))
plt.ylim((-0.5, 1.5))
plt.xlabel('ddG')
plt.ylabel('ddG double dagger')
plt.plot([-1, 2], [-1, 2], 'k')
plt.plot([-1, 2], [0, 0], 'k:')
plt.plot([0, 0], [-1, 2], 'k:')
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'ddG_versus_ddg_double_dagger.scatterplot.png'))

# plot jointplot
color ='vermillion'
xlim = [-0.5, 1.5]
ylim = xlim
with sns.axes_style("white"):
    name1 = 'cell'; name2='tissue'
    g = sns.JointGrid(delta_delta_g, delta_koff_mut, size=5, ratio=5, space=0, dropna=True, xlim=xlim, ylim=xlim)
    g.plot_marginals(sns.distplot, kde=True, color=sns.xkcd_rgb[color])
    g.plot_joint(plt.hexbin, bins='log', cmap='Greys', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], gridsize=50)
    g.set_axis_labels('ddG', 'ddG koff effect')
    g.annotate(st.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");

plt.savefig(os.path.join(figDirectory, 'koff_effect.png'))
    
color ='vermillion'
xlim = [-0.5, 1.5]
ylim = [-1, 1]

with sns.axes_style("white"):
    g = sns.JointGrid(delta_delta_g, delta_kon_mut, size=5, ratio=5, space=0, dropna=True, xlim=xlim, ylim=ylim)
    g.plot_marginals(sns.distplot, kde=True, color=sns.xkcd_rgb[color])
    g.plot_joint(plt.hexbin, bins='log', cmap='Greys', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], gridsize=50)
    g.set_axis_labels('ddG', 'ddG kon effect')
    g.annotate(st.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");

plt.savefig(os.path.join(figDirectory, 'kon_effect.png'))


#plot histogram
plt.figure(figsize = (6,4))
plt.tick_params(direction="out", right='off', top='off')
histogram.compare([delta_delta_g, delta_delta_g-delta_koff_mut], xbins=np.linspace(-2, 2, 50), labels=['Explained by koff', 'Unexplained'], bar=True, cmap='Paired')

ax = plt.gca()
plt.xlim((-0.5, 1.5))
ax.set_xlabel('ddG (kcal/mol)')
ax.set_ylabel('fraction')
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.55)
plt.savefig(os.path.join(figDirectory, 'ddG.histograms.pdf'))

# find distance between variants

helix_context='rigid'
shorter_helix='helix_two'
junction_maxlength = 3
allvariants, seqs = generateVariantNumber.allThreeByThreeVariants(variant_table, helix_context=helix_context, offset=1, shorter_helix=shorter_helix, maxjunction_length=junction_maxlength)
allvariant_subset = list(itertools.chain.from_iterable([allvariants.iloc[i, j] for i,j  in itertools.product(range(junction_maxlength), range(junction_maxlength))]))
seqs_subset = np.array(list(itertools.chain.from_iterable([seqs.iloc[i, j] for i,j  in itertools.product(range(junction_maxlength), range(junction_maxlength))])))

numtests = len(allvariant_subset)
rsq = pd.DataFrame(index=seqs_subset, columns=seqs_subset)
numCores = 20

for k in range(numtests-1):
    numtests_per_seq = len(allvariant_subset[k+1:numtests])
    print 'comparing %s to %d others'%(seqs_subset[k], numtests_per_seq)
    #rsq.loc[k,k] = 1
    workerPool = multiprocessing.Pool(processes=np.min([numCores, numtests_per_seq])) 
    rsq.iloc[k,k+1:] = workerPool.map(functools.partial(fitFun.correlationBetweenVariants, variant_table, allvariant_subset[k]),
                                     allvariant_subset[k+1:numtests])
    workerPool.close(); workerPool.join()
    rsq.iloc[(k+1):,k] = rsq.iloc[k,k+1:]
    
reorder = range(0, 5) + range(21, 25) + range(5,21) + range(85, 101) + range(37, 85) + range(101, 149) + range(25, 37) + range(149, len(seqs_subset))
rsqmod = pd.DataFrame(rsq.iloc[reorder, reorder], index=seqs_subset[reorder], columns=seqs_subset[reorder])
index = rsqmod.dropna(axis=0, how='all').dropna(axis=0, how='any', thresh=180).index
cNorm  = colors.Normalize(vmin=0.1, vmax=1)
cMap = 'coolwarm'
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
im = plt.imshow(np.array(rsqmod.loc[index, index], dtype=float), interpolation='nearest', cmap=cMap, norm=cNorm)
plt.colorbar(im)
plt.xticks(np.arange(len(index)), np.array(index), fontsize=6, rotation=90)
plt.yticks(np.arange(len(index)), np.array(index), fontsize=6)
plt.tick_params(direction='out', top='off', right='off')



plt.savefig(os.path.join(figDirectory, 'rsq_matrix.flowpiece_%s.helix_%s.shorter_%s.maxlength_%d.pdf'%(flowpiece, helix_context, shorter_helix, junction_maxlength)))
rsq.to_csv(os.path.join(figDirectory, 'rsq_matrix.flowpiece_%s.helix_%s.shorter_%s.maxlength_%d.mat'%(flowpiece, helix_context, shorter_helix, junction_maxlength)),
           index=True, header=True)
allvariants.to_csv(os.path.join(figDirectory, 'allvariants.flowpiece_%s.helix_%s.shorter_%s.maxlength_%d.mat'%(flowpiece, helix_context, shorter_helix, junction_maxlength)),
                   index=True, header=True)

index = rsq.dropna(axis=(0,1), how='all').index
rsqmod = np.array(rsq.loc[index, index], dtype=float)
for k in range(rsqmod.shape[0]): rsqmod[k,k] = 1
rsqmod = pd.DataFrame(rsqmod, index=index, columns=index)
index = rsqmod.dropna(how='any').index
index = rsqmod.dropna(how='all', thresh=len(rsqmod)-5).index
rsqmod = rsqmod.loc[index, index]
plt.figure(figsize=(10,10))
heatmapfun.plotHeatMap(rsqmod.values, rowlabels=np.array([label.replace('_', '-') for label in rsqmod.index]),
                       columnlabels=np.array([label.replace('_', '-') for label in rsqmod.index]), fontSize=4, cmap=cMap, vmin=0.1, vmax=1, metric='euclidean')
plt.savefig(os.path.join(figDirectory, 'rsq_matrix.flowpiece_%s.helix_%s.shorter_%s.maxlength_%d.clustered.pdf'%(flowpiece, helix_context, shorter_helix, junction_maxlength)))

#plot a few variants

seq = ['_', 'AG_T', '_CG', 'AT_G', 'A_GA']
for variants in np.array(allvariant_subset)[np.array([np.where(seqs_subset==seq)[0][0] for seq in ['_', 'AG_T', '_CG', 'AT_G', 'A_GA']])]:
    variant_subtable = pd.concat([variant_table['seqinfo'].loc[variants], variant_table['affinity'].loc[variants]], axis=1)
    variantFun.plot_over_coordinate(variant_subtable)
    plt.close()
    junction_seq = variant_subtable.junction_sequence.values[0].replace('_', '-')
    plt.title(variant_subtable.junction_sequence.values[0])
    plt.savefig(os.path.join(figDirectory, 'junction_seq_%s.flowpiece_%s.helix_%s.shorter_%s.maxlength_%d.pdf'%(junction_seq, flowpiece, helix_context, shorter_helix, junction_maxlength)))

# plot changes in delta delta G in different helix contexts
seq = '_'
helix_context = ''
loop = 'goodLoop'
total_length = 10
offset=1
variants = variantFun.findVariantNumbers(variant_table.seqinfo, {'junction_sequence': seq, 'loop':loop,
                                                         'total_length': total_length,
                                                         'receptor':'R1', 'offset':offset})


indx = variant_table.loc[variants].affinity.dropna(axis=0, how='all', subset=['dG']).index
columns = ['dG', 'dG_ub', 'dG_lb']
no_junction = pd.DataFrame(columns=columns,
                           index=variant_table.seqinfo.loc[indx, 'helix_context'].values,
                           data=variant_table.affinity.loc[indx, columns].values,
                           dtype=float)
all_seqs = np.unique(variant_table.seqinfo.loc[variant_table.seqinfo.helix_context == 'h21', 'junction_sequence'].values).astype(str)
helix_contexts = no_junction.sort('dG').index
delta_delta_G = {}

for newseq in all_seqs:

    delta_delta_G[newseq] = pd.DataFrame(columns=helix_contexts, index=['median', 'eminus', 'eplus'], dtype=float)
    for helix_context in helix_contexts:
        variants = variantFun.findVariantNumbers(variant_table.seqinfo, {'junction_sequence': newseq, 'loop':loop,
                                                             'total_length': total_length,
                                                             'helix_context':helix_context,
                                                             'receptor':'R1', 'offset':offset})
        if not variant_table.affinity.loc[variants, 'dG'].empty:
            delta_delta_G[newseq].loc['median', helix_context] = (variant_table.affinity.loc[variants, 'dG'] -
                                                                  no_junction.loc[helix_context, 'dG']).values[0]
            delta_delta_G[newseq].loc['eminus', helix_context] = np.sqrt(np.power(variant_table.affinity.loc[variants, 'dG_lb'] - variant_table.affinity.loc[variants, 'dG'], 2) +
                                                                         np.power(no_junction.loc[helix_context, 'dG_lb'] - no_junction.loc[helix_context, 'dG'] , 2)
                                                                        ).values[0]
            delta_delta_G[newseq].loc['eplus', helix_context] = np.sqrt(np.power(variant_table.affinity.loc[variants, 'dG_ub'] - variant_table.affinity.loc[variants, 'dG'], 2) +
                                                                        np.power(no_junction.loc[helix_context, 'dG_ub'] - no_junction.loc[helix_context, 'dG'], 2)
                                                                       ).values[0]
junction_seqs = np.array(delta_delta_G.keys())

#### average change for all relative to mean
helix_contexts = np.array(['h21', 'wc', 'rigid', 'h08', 'h02', 'h25', 'h14', 'h28', 'h06'])
yvalues_all = np.array([delta_delta_G[newseq].loc['median',helix_contexts].values for newseq in junction_seqs], dtype=float)

helix_contexts = helix_contexts[np.argsort(np.median(yvalues_all, axis=0))]
xvalues = np.arange(len(helix_contexts))
yvalues_all = np.array([delta_delta_G[newseq].loc['median',helix_contexts].values for newseq in junction_seqs], dtype=float)
yerr_all = np.array([[delta_delta_G[newseq].loc['eminus',helix_contexts].values,
                      delta_delta_G[newseq].loc['eplus', helix_contexts].values] for newseq in junction_seqs])
fig = plt.figure(figsize=(5,4))
gs1 = gridspec.GridSpec(1, 2, width_ratios=[8,1])
ax =  plt.subplot(gs1[0,0])

for i in range(len(junction_seqs)):
    indx = np.logical_not(np.any(np.isnan(yerr_all[i]), axis=0))
    yvalues = yvalues_all[i, indx]
    yerr  = yerr_all[i, :, indx]
    dx = DescrStatsW(yvalues, 1/np.sum(yerr, axis=1))

    ax.plot(xvalues[indx], dx.demeaned, 'k-', alpha=0.05)
    #ax.errorbar(xvalues[1:], yvalues[1:], yerr = yerr[:,1:], fmt='-', color='k', ecolor='k', alpha=0.025, linewidth=2, capsize=0)
weights = (yvalues_all.shape[1] - np.sum(np.isnan(yvalues_all), axis=1))/float(yvalues_all.shape[1])
for i in range(yvalues_all.shape[1]):
    yvalues_all[np.isnan(yvalues_all)[:,i],i] = np.nanmean(yvalues_all, axis=0)[i]
dx = DescrStatsW(yvalues_all, weights=weights)
cmap = 'Reds'
cNorm  = colors.Normalize(vmin=0, vmax=3)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for i in range(yvalues_all.shape[1]):
    color = scalarMap.to_rgba(-np.log10(dx.ttest_mean(np.mean(dx.mean))[1][i]))
    ax.errorbar(xvalues[i], dx.mean[i] - np.mean(dx.mean),
                yerr=np.vstack(np.abs(dx.zconfint_mean() - dx.mean)[:,i]), fmt='o-', color=color, ecolor='k')
ax.plot([xvalues[0]-1, xvalues[-1]+1], [0,0], 'k:')
ax.tick_params(direction='out', top='off', right='off')
ax.set_xticks(xvalues)
ax.set_xticklabels(helix_contexts, rotation=90)
ax.set_ylabel('dddG (kcal/mol)')
ax.set_xlabel('helix sequence')
ax.set_xlim([xvalues[0]-1, xvalues[-1]+1])
ax.set_ylim([-1.5, 2.5])


ax1 =  plt.subplot(gs1[0,1])
cb1 = mpl.colorbar.ColorbarBase(ax1,cmap=cmap+'_r',
                                   norm= colors.LogNorm(vmin=0.001, vmax=1),
                                   orientation='vertical',
                                   )
cb1.set_label('pvalue')
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'ddG.all_junctions.changing_helix_context.sort_by_ddG.rel_to_mean.colored.pdf'))

