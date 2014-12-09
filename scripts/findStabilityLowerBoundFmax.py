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
resulting_dGs = np.zeros((len(fmax_lbs), len(indx_subset)))
resulting_var_dGs = np.zeros((len(fmax_lbs), len(indx_subset)))

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