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
#cutoff['dG_error'] = 0.5 # kcal/mol
#cutoff['qvalue']  = 0.05
cutoff['numTests'] = 2
cutoff['total_length'] = 10
cutoff['receptor'] = 'R1'
cutoff['loop'] = 'goodLoop'

indxFit = pd.Series()
#indxFit['dG_error'] = (variant_table.dG_ub - variant_table.dG_lb).values < cutoff['dG_error']
#indxFit['qvalue']   = variant_table.qvalue.values <= cutoff['qvalue']
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
histogram.compare([np.power(variant_table.loc[subset_index, 'sigma'], 2)],
    xbins =np.linspace(-0.01, 0.2, 50), bar=True, normalize=False)
ax.set_xlabel('measurement variance (kcal/mol)')
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
subset_parse = np.arange(len(variant_subtable.index))
#subset_parse = np.arange(50)
if args.tableFile is None:
    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    #table = [fitFun.multiprocessParametrization(variant_table, dibases_wt, indx) for indx in variant_subtable.index[200:1000]]
    tableList = workerPool.map(functools.partial(fitFun.multiprocessParametrization, variant_table, dibases_wt), variant_subtable.index[subset_parse])
    workerPool.close(); workerPool.join()
    table = pd.DataFrame.from_records(tableList, index=variant_subtable.index[subset_parse]).dropna(axis=(0,1), how='all')
    table.to_csv(os.path.join('linear_model', 'param.flowpiece_%s.length_%d.original.mat'%(flowpiece, helixLengthTotal)))
else:
    table = pd.read_table(args.tableFile, header=[0,1])
# get a sense of how many variants there are per parameter
min_num_variants_to_fit = 10
num_params = table.shape[1]
num_tests = float(table.shape[0])
for param in np.unique([name[0] for name in table]):
    num_params = table.loc[:, param].shape[1]
    for cat in np.unique([name[0] for name in table.loc[:, param]]):
        num_params = len(table.loc[:, param].loc[:, cat].sum(axis=0))
        plt.figure(figsize=(4, 5))
        plt.bar(np.arange(num_params), table.loc[:, param].loc[:, cat].sum(axis=0)/num_tests )
        plt.xticks(np.arange(num_params)+0.5, [name for name in table.loc[:, param].loc[:, cat]], rotation=90)
        plt.tick_params(direction='out', top='off', right='off' ,)
        plt.subplots_adjust(left = 0.15, bottom=0.2)
        plt.plot([0, num_params+1], [min_num_variants_to_fit/num_tests, min_num_variants_to_fit/num_tests], 'k:')
        plt.ylabel('fraction tested')
        #plt.ylim((0, num_params))
        plt.xlabel('location relative to loop')
        plt.title('%s %d'%(param, cat))
        plt.tight_layout()
        plt.savefig(os.path.join(saveDirectory, 'number_variants.flowpiece_%s.length_%dbp.param_%s.cat_%d.pdf'%(flowpiece,helixLengthTotal, param, cat)))
# do something special for length
param = 'length'
num_params = table.loc[:, param].shape[1]
plt.figure(figsize=(4, 5))
plt.bar(np.arange(num_params), table.loc[:, param].sum(axis=0)/num_tests )
plt.xticks(np.arange(num_params)+0.5, [7+name[0] for name in table.loc[:, param]], rotation=90)
plt.tick_params(direction='out', top='off', right='off' ,)
plt.subplots_adjust(left = 0.15, bottom=0.2)
plt.plot([0, num_params+1], [min_num_variants_to_fit/num_tests, min_num_variants_to_fit/num_tests], 'k:')
plt.ylabel('fraction tested')
#plt.ylim((0, num_params))
plt.xlabel('length (bp)')
plt.title('%s %d'%(param, cat))
plt.tight_layout()
plt.savefig(os.path.join(saveDirectory, 'number_variants.flowpiece_%s.length_%dbp.param_%s.pdf'%(flowpiece,helixLengthTotal, param)))
# find parameters with few variants
min_num_variants_to_fit = 10
bad_params = table.sum(axis=0) < min_num_variants_to_fit

# find variants that have no variation in the bad parameters 
good_variants = table.loc[:,bad_params].sum(axis=1) == 0
bad_params.loc['bp_bp', 0] = True  # remove bp bp with offset 0

# remove bad params and keep good variants
tableFinal = fitFun.flattenMatrix(table.loc[good_variants, np.logical_not(bad_params)])
tableFinal = tableFinal.dropna(axis=1, thresh=1000).dropna(axis=0, how='any')
weight = True
tableFinal.loc[:, 'ddG'] = variant_table.loc[tableFinal.index, 'dG'] - variant_table.loc[indx_wt, 'dG']
if weight:
    tableFinal.loc[:, 'weights'] = 1/variant_table.loc[tableFinal.index, 'sigma']
    weighted='yes'
else:
    tableFinal.loc[:, 'weights'] = np.ones(len(tableFinal))
    weighted = 'no'

index = tableFinal.loc[np.isfinite(tableFinal).all(axis=1)].index
index = variant_subtable.loc[tableFinal.loc[index].index].sort('topology').index
outfile = os.path.join(saveDirectory, 'param.flowpiece_%s.length_%d.weighted_%s.mat'%(flowpiece, helixLengthTotal, weighted))
fitfile = os.path.join(saveDirectory, 'param_est.flowpiece_%s.length_%d.weighted_%s'%(flowpiece, helixLengthTotal, weighted))
tableFinal.loc[index].to_csv(outfile, sep='\t', index=True)

# send to R
try:
    subprocess.check_call("Rscript ~/array_image_tools_SKD/scripts/linear_model.r %s %s"%(outfile, fitfile), shell=True)
except:
    pass

# load and plot results from R
which_model = 'lasso'
exVec = table.loc[good_variants, np.logical_not(bad_params)].iloc[0]
fitParams = fitFun.loadFitParams(fitfile+'.'+which_model, exVec)
params = fitParams.index.levels[0].tolist()

# load predicted values
test_pred = pd.read_table(fitfile+'.'+which_model+'.test.mat', header=0, names=['predicted_ddG'], index_col=0)
param_est.flowpiece_rigid.length_10.weighted_yes.lasso.test.mat

paramsToPlot = ['bp_break', 'nc', 
                'seq_change_x', 'seq_change_i', 'loop']
                
paramNames = 'basepairing'
fitFun.plotFit(fitParams, paramsToPlot=paramsToPlot)
plt.savefig(os.path.join(saveDirectory,'parameters_plot.flowpiece_%s.length_%dbp.weighted_%s.model_%s.%s.pdf'%(flowpiece, helixLengthTotal, weighted, which_model, paramNames)))

paramsToPlot = ['insertions_k', 'insertions_z',
                'seq_insert_k', 'seq_insert_z',
                'loop']
paramNames = 'insertions'
fitFun.plotFit(fitParams, paramsToPlot=paramsToPlot)
plt.savefig(os.path.join(saveDirectory,'parameters_plot.flowpiece_%s.length_%dbp.weighted_%s.model_%s.%s.pdf'%(flowpiece, helixLengthTotal, weighted, which_model, paramNames)))

paramsToPlot = ['bp_bp',
 'bp_ins_k1',
 'bp_ins_k2',
 'bp_ins_z1',
 'bp_ins_z2',]
paramNames = 'interactions'
fitFun.plotFit(fitParams, paramsToPlot=paramsToPlot)
plt.savefig(os.path.join(saveDirectory,'parameters_plot.flowpiece_%s.length_%dbp.weighted_%s.model_%s.%s.pdf'%(flowpiece, helixLengthTotal, weighted, which_model, paramNames)))

key = 'bp_bp'
fitFun.plotInteractions(fitParams, key)
plt.savefig(os.path.join(saveDirectory,'parameters_heatmap.flowpiece_%s.length_%dbp.weighted_%s.model_%s.%s.pdf'%(flowpiece, helixLengthTotal, weighted, which_model, key)))

key = 'bp_ins_z2'
fitFun.plotInteractions(fitParams, key)
plt.savefig(os.path.join(saveDirectory,'parameters_heatmap.flowpiece_%s.length_%dbp.weighted_%s.model_%s.%s.pdf'%(flowpiece, helixLengthTotal, weighted, which_model, key)))

# plot only mismatches
junction_topology = 'B2'
variants = variantFun.findVariantNumbers(variant_table, {'helix_context':flowpiece, 'loop':'goodLoop',
                                                         'receptor':'R1', 'topology':junction_topology, 'total_length':10})
predicted = test_pred.loc[variants].dropna()
variantFun.plot_scatterplot_errorbars(variant_table.loc[predicted.index], yvalues=predicted.values, parameter='dG', )