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
import generateVariantNumber
parameters = variantFun.Parameters()

flowpiece = 'wc'
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
variant_table.dropna(axis=0, subset=['variant_number', 'dG'], inplace=True, how='any')

lims = {'qvalue':(2e-4, 1e0),
        'dG':(-12, -4),
        'koff':(1e-5, 1E-2),
        'kon' :(1e-2, 1e5),
        'kobs':(1e-5, 1e-2)}


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

# plot per junction info
seq = '_'
criteria_dict = {'junction_sequence': seq, 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1'}
variants = variantFun.findVariantNumbers(table.seqinfo, criteria_dict)

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
    ax, concentrationsAll = variantFun.plotBindingCurveVariant(subtable['affinity'], parameters.concentrations)
    ax.plot(concentrationsAll, variantFun.bindingCurve(concentrationsAll, variant_table['affinity'].loc[variant, 'dG_ub']), 'r:')
    ax.plot(concentrationsAll, variantFun.bindingCurve(concentrationsAll, variant_table['affinity'].loc[variant, 'dG_lb']), 'r:')
    title = variantFun.getInfo(variant_table['seqinfo'].loc[variant])
    plt.title(title, fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(figDirectory, 'binding_curves.q_%4.2f.dG_%4.1f.variant_%s.pdf'%(variant_table['affinity'].loc[variant, 'qvalue'],
                                                                                             variant_table['affinity'].loc[variant, 'dG'],
                                                                                             title)))
    # histogram dG
    param = 'dG'
    ymax=5
    variantFun.plotHistogram(subtable['affinity'], param)
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
    ax, timeBinCenters = variantFun.plotOffRateVariant(subtable['offrates'], times)
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
    ax.fill_between([variant_table['offrates'].loc[variant, param+'_lb'], variant_table['offrates'].loc[variant, param+'_ub']], 0, ymax, color='0.25', alpha=0.1)
    ax.plot([variant_table['offrates'].loc[variant, param], variant_table['offrates'].loc[variant, param]], [0, ymax], 'k:', linewidth=2)
    
    plt.title(title, fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(figDirectory, 'histogram.toff.q_%4.2f.toff_%4.1f.variant_%s.pdf'%(variant_table['offrates'].loc[variant, 'qvalue'],
                                                                                             variant_table['offrates'].loc[variant, 'toff'],
                                                                                             title)))
# koff contribution
index = variant_table['offrates'].loc[:, 'qvalue'].values < 0.005
koff_rel = variant_table['summary'].loc[0, 'koff']
dg_rel = variant_table['summary'].loc[0, 'dG']
delta_koff_mut = parameters.RT*np.log(np.array(variant_table['summary'].loc[index, 'koff'], dtype=float)/koff_rel)
delta_delta_g = (variant_table['summary'].loc[index, 'dG'].values - dg_rel).astype(float)
histogram.compare([delta_koff_mut[delta_delta_g!=0]/delta_delta_g[delta_delta_g!=0]])

# plot scatterplot
plt.figure(figsize = (5,5))
plt.scatter( delta_delta_g,delta_koff_mut, alpha=0.05, linewidth=0 )
plt.tick_params(direction="out")
plt.xlim((-1, 2))
plt.ylim((-1, 2))
plt.xlabel('ddG')
plt.ylabel('ddG double dagger')
plt.plot([-1, 2], [-1, 2], 'k')
plt.plot([-1, 2], [0, 0], 'k:')
plt.plot([0, 0], [-1, 2], 'k:')
plt.tight_layout()
plt.savefig('flow_wc/ddG_versus_ddg_double_dagger.scatterplot.png')

#plot histogram
plt.figure(figsize = (5,5))
plt.tick_params(direction="out")
histogram.compare([delta_koff_mut[delta_delta_g!=0]-delta_delta_g[delta_delta_g!=0]], xbins=np.linspace(-2, 2, 50), bar=True)
ax = plt.gca()
ax.legend_ = None
ax.set_xlabel('ddG double dagger - ddG (kcal/mol)')
ax.set_ylabel('fraction')
plt.tight_layout()
plt.savefig('flow_wc/ddG_versus_ddg_double_dagger.histogram.png')

# find distance between variants
variant_set = ['','']
seq = '_'; helix_context = ['rigid', 'wc']
criteria_dict = {'junction_sequence': seq, 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1'}
variant_set = [variantFun.findVariantNumbers(table.seqinfo, {'junction_sequence': seq, 'helix_context':helix, 'loop':'goodLoop', 'receptor':'R1'}) for helix in helix_context]

   

    
allvariants, seqs = generateVariantNumber.allThreeByThreeVariants(variant_table, helix_context='wc', offset=1, shorter_helix='helix_one')

i = 1; j = 0
for variantlist, seq in itertools.izip(allvariants.loc[i,j], seqs.loc[i,j]):
    variant_set = [variantlist, allvariants.loc[0,0][0]]

    variant_subtable = pd.concat([variant_table['seqinfo'].loc[variantlist], variant_table['affinity'].loc[variantlist]], axis=1)
    variantFun.plot_over_coordinate(variant_subtable.dropna(subset=['dG'], axis=0))
        
    deltaG = fitFun.distanceBetweenVariants(variant_table, variant_set)
    fitFun.plotDeltaDeltaG(deltaG)
    plt.title('ddG from wc to junction %s'%seq, fontsize=10)
    

