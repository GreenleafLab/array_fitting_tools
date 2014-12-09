"""
Given that we are imposing constraints on fmax, figure out how stable
the resulting dG is for some subset of clusters to changes in fmax

"""
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
import CPlibs
import IMlibs
import fittingParameters
from matplotlib import rc
rc('text', usetex=True)

# may want to make sure path is right here
parameters = fittingParameters.Parameters()

# load fitted dGs to be able to get a somewhat reresentative subset
fittedBindingFilename = 'binding_curves_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPfitted'
table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
indx_subset = np.array(np.argsort(table['dG'])[np.arange(0, len(table), 100)])
table_reduced = table.iloc[indx_subset]

# load null scores
signalNamesByTileDict = {'003': 'binding_curves_rigid/AAYFY_ALL_tile003_Bottom_filtered.CPsignal'}
filterSet = 'tecto'
null_scores = IMlibs.loadNullScores(signalNamesByTileDict, filterSet)

# Load binding series before fitting
numConcentrations = 8
bindingSeries = np.transpose([np.array(table_reduced[i]) for i in range(numConcentrations)])
allClusterSignal = np.array(table_reduced['all_cluster_signal'])

# now split into parts
annotatedSignalFilename = 'binding_curves_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPsignal'
bindingSeriesSplit = np.array_split(bindingSeries, numCores)
allClusterSignalSplit = np.array_split(allClusterSignal, numCores)
bindingSeriesFilenameParts = IMlibs.getBindingSeriesFilenameParts(annotatedSignalFilename, numCores)
fitParametersFilenameParts = IMlibs.getfitParametersFilenameParts(bindingSeriesFilenameParts)

# fit with different fmaxes
fmax_lbs = np.arange(0, 2, 0.1)
num_clusters = len(indx_subset)
num_lbs = len(fmax_lbs)
resulting_dGs = np.zeros((num_lbs, num_clusters))
resulting_var_dGs = np.zeros((num_lbs, num_clusters))

for j, fmax_lb in enumerate(fmax_lbs):
    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
        sio.savemat(bindingSeriesFilename, {'concentrations':concentrations,
                                            'binding_curves': bindingSeriesSplit[i].astype(float),
                                            'all_cluster':allClusterSignalSplit[i].astype(float),
                                            'null_scores':null_scores,
                                            })
        workerPool.apply_async(IMlibs.findKds, args=(bindingSeriesFilename, fitParametersFilenameParts[i],
                                                     fmax_lb, parameters.fmax_max, parameters.fmax_initial,
                                                     parameters.dG_min, parameters.dG_max, parameters.dG_initial,
                                                     parameters.fmin_min, parameters.fmin_max, parameters.fmin_initial),
                               )
    workerPool.close()
    workerPool.join()
    
    fitParameters = IMlibs.joinTogetherFitParts(fitParametersFilenameParts)
    resulting_dGs[j] = fitParameters['dG']
    resulting_var_dGs[j] = fitParameters['dG_var']
    resulting_qvalues = fitParameters['qvalue']
    
# plot with error bars
dirname = 'binding_curves_rigid'
subset = np.arange(num_clusters)[np.array(resulting_qvalues < 0.05)]
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
for i in subset[:200]:
    ax.errorbar(fmax_lbs, resulting_dGs[:,i]-resulting_dGs[0,i], yerr=resulting_var_dGs[:,i], fmt='-o', color = 'r', ecolor='k', alpha = 0.1)
ax.set_ylim((-2, 7))
ax.set_xlabel('lower bound fmax')
ax.set_ylabel('ddG')
plt.tight_layout()
plt.savefig('%s/lower_bound_fmax_stability.below_qvalue.png'%(dirname))

subset = np.arange(num_clusters)[np.array(resulting_qvalues > 0.05)]
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
for i in subset[:200]:
    ax.errorbar(fmax_lbs, resulting_dGs[:,i]-resulting_dGs[0,i], yerr=resulting_var_dGs[:,i], fmt='-o', color = 'b', ecolor='k', alpha = 0.1)
ax.set_ylim((-2, 7))
ax.set_xlabel('lower bound fmax')
ax.set_ylabel('ddG')
plt.tight_layout()
plt.savefig('%s/lower_bound_fmax_stability.above_qvalue.png'%(dirname))

# get fraction that change from previous point
threshold = 0.1 # kcal/mol
matrix = np.zeros((num_lbs, num_clusters), dtype=bool)
for i in range(1, num_lbs):
    # for each lower bound, is it different from previous?
    matrix[i] = np.abs(resulting_dGs[i] - resulting_dGs[0]) > resulting_var_dGs[i]
# plot
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
# first do those below qvalue
subset_below = np.arange(num_clusters)[np.array(resulting_qvalues < 0.05)]
subset_above = np.arange(num_clusters)[np.array(resulting_qvalues > 0.05)]
ax.plot(fmax_lbs, np.sum(matrix[:, subset_below], axis=1)/float(len(subset_below)), 'r', label='below FDR')
ax.plot(fmax_lbs, np.sum(matrix[:, subset_above], axis=1)/float(len(subset_above)), 'b', label='above FDR')
ax.set_xlabel('lower bound fmax')
ax.set_ylabel('fraction that significantly differ from lb=0')
ax.set_ylim((0, 1))
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.savefig('%s/lower_bound_fmax_stability.fraction_below_var_from0.pdf'%(dirname))