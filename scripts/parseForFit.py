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
import variantFun
import IMlibs
import fitFun
parameters = variantFun.Parameters()


flowpiece = 'wc'
name = 'binding_curves'
fittedBindingFilename = '%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPfitted'%(name, flowpiece) 
table = IMlibs.loadFittedCPsignal(fittedBindingFilename, index_by_cluster=True)
table = table.sort_index(axis=0).sort('variant_number')

variantFittedFilename = '%s_%s/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'%(name, flowpiece)
variant_table = pd.read_table(variantFittedFilename)

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
headers = np.hstack([np.hstack(['%d_%s_%s'%(idx, 'bp_break', 'i'), '%d_%s_%s'%(idx, 'nc', 'i'),
                                 '%d_%s_%s'%(idx, 'bp_break', 'j'), '%d_%s_%s'%(idx, 'nc', 'j'),
                                 '%d_insertions_k'%idx,
                                 '%d_insertions_z'%idx])
                 for idx in indexParam])

# Predict secondary structure, parse into dibases, save parameters for each dibase
variant_subtable = variant_table.loc[np.all([indxFit[name] for name in indxFit.index], axis=0)]
numCores = 20
workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
table = workerPool.map(functools.partial(fitFun.multiprocessParametrization, variant_table, helixLengthTotal), variant_subtable.index)
workerPool.close(); workerPool.join()

tableFinal = pd.DataFrame(data=np.vstack(table), index=variant_subtable.index, columns = headers)
tableFinal.loc[variant_subtable.index, 'dG'] = variant_subtable.loc[:, 'dG']
# Save test set and training set

# sort by topology and take every other
indexTest = variant_subtable.sort('topology').index[np.arange(0, len(variant_subtable), 2)]
indexTraining =  variant_subtable.sort('topology').index[np.arange(1, len(variant_subtable), 2)]

tableFinal.loc[indexTraining].to_csv('linear_model/param.wc.10bp.training.mat', sep='\t', index=True)
