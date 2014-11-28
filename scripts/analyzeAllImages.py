#!/usr/bin/env python

# Main Pipeline for Doing Analysis of Images
# ---------------------------------------------
#
# This script requires you to already have quantified images with another pipeline.
#
# Inputs:
#   Sequence data (.CPseq files)
#   Parameters (globalvars.py file; used to define fitting parameters,
#                             which column in CPseq is the barcode,
#                             which column in CPseq is the sequence data,
#                             whether it is reverse complemented,
#                             and what images correspond to what binding concentration, offrate, etc)
#   CPfluor files (.CPfluor files or directories)
#   Classifier files (.characterization)
#
# Outputs:
#   CPsignal files labeled with classifiers and fits
#
# Sarah Denny
# November 2014


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

import CPlibs
import IMlibs



### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description="master script for the fitting clusters to binding curves pipeline")
parser.add_argument('-ftd','--filtered_tile_dir', help='directory that holds (or will hold) the filtered sequence data', required=True)
parser.add_argument('-fd','--CPfluor_dirs_to_quantify', help='text file giving the dir names to look for CPfluor files in', required=True)
parser.add_argument('-os','--signal_dir', help='save signal files to here. default = filtered_tile_dir/signals')
parser.add_argument('-osr','--signal_dir_reduced', default=None,help='save signal files reduced by filter set to here. default = signal_dir/reduced_signals')
parser.add_argument('-fs','--filter_set', help='name of filter to fit binding curve', required=True)
parser.add_argument('-bd','--barcode_dir', help='save files associated with barcode mapping here. default=signal_dir_reduced/barcode_mapping')
parser.add_argument('-n','--num_cores', help='maximum number of cores to use')
parser.add_argument('-gv','--global_vars_path', help='path to the directory in which the "globalvars.py" parameter file for the run can be found')

if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()

#add global vars path to system
if args.global_vars_path is not None:
    sys.path.insert(0, args.global_vars_path)

# import parameters
import globalvars
parameters = globalvars.Parameters()

# import CPseq filtered files split by tile
print 'Finding CPseq files in directory "%s"...'%args.filtered_tile_dir
filteredCPseqFilenameDict = CPlibs.findTileFilesInDirectory(args.filtered_tile_dir, ['_filtered.CPseq'], [])
tileList = filteredCPseqFilenameDict.keys()

# import directory names to analyze
print 'Finding CPfluor files in directories given in "%s"...'%args.CPfluor_dirs_to_quantify
xValuesFilename, fluor_dirs_red, fluor_dirs, concentrations = IMlibs.loadMapFile(args.CPfluor_dirs_to_quantify)
fluorNamesByTileDict = IMlibs.getFluorFileNames(fluor_dirs, tileList)
fluorNamesByTileRedDict = IMlibs.getFluorFileNames(fluor_dirs_red, tileList)

# initiate multiprocessing
if args.num_cores is None:
    numCores = 1
else: numCores = int(args.num_cores)

################ Make signal files ################
if args.signal_dir is not None:
    signal_directory = args.signal_dir
else:
    signal_directory = os.path.join(args.filtered_tile_dir,'signals')

print 'Converting CPfluor files into CPsignal files in directory "%s"'%signal_directory

signalNamesByTileDict = IMlibs.getCPsignalDictfromCPseqDict(filteredCPseqFilenameDict, signal_directory)
already_exist = np.zeros(len(signalNamesByTileDict), dtype=bool)
if os.path.isdir(signal_directory): #if the CPsignal file is already present
    print 'SignalDirectory "' + signal_directory + '" already exists...'
    print '   finding tile .CPsignal files in directory "%s"...'%(signal_directory)
    for i, tile in enumerate(tileList):
        already_exist[i] = os.path.isfile(signalNamesByTileDict[tile])
        if already_exist[i]:
            print "   found tile %s: %s. Skipping..."%(tile, signalNamesByTileDict[tile])
        else: print "   missing tile %s"%tile
else:
    os.makedirs(signal_directory) #create a new output directory if it doesn't exist
    
if np.sum(already_exist) < len(filteredCPseqFilenameDict):
    # initiate multiprocessing
    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    for i, tile in enumerate(tileList):
        if not already_exist[i]:
            currCPseqfile = filteredCPseqFilenameDict[tile]
            currGreenCPfluors = fluorNamesByTileDict[tile]
            currRedCPfluors = fluorNamesByTileRedDict[tile]
            currCPseqSignalOut = signalNamesByTileDict[tile]
            print "Making signal file %s from %s"%(currCPseqSignalOut, currCPseqfile)
            # make CP signal files
            workerPool.apply_async(IMlibs.findCPsignalFile, args=(currCPseqfile, currRedCPfluors, currGreenCPfluors, currCPseqSignalOut))
        else:
            print "Signal file %s already exists"%currCPseqSignalOut
    workerPool.close()
    workerPool.join()


################ Reduce signal files ################
if args.signal_dir_reduced is not None:
    reduced_signal_directory = args.signal_dir_reduced
else:
    reduced_signal_directory = os.path.join(signal_directory,'reduced_signals')
filterSet = args.filter_set
    
print 'Converting CPsignal files into reduced CPsignal files in directory "' +reduced_signal_directory

# make directory if necessary or see if files already exist
reducedSignalNamesByTileDict = IMlibs.getReducedCPsignalDictFromCPsignalDict(signalNamesByTileDict, filterSet, reduced_signal_directory)
already_exist = np.zeros(len(reducedSignalNamesByTileDict), dtype=bool)
if os.path.isdir(reduced_signal_directory): #if the CPsignal file is already present
    print 'Reduced SignalDirectory "' + reduced_signal_directory + '" already exists...'
    print '   finding tile %s.CPsignal files in directory "%s"'%(filterSet, reduced_signal_directory)
    for i, tile in enumerate(tileList):
        already_exist[i] = os.path.isfile(reducedSignalNamesByTileDict[tile])
        if already_exist[i]:
            print "   found tile %s: %s. Skipping..."%(tile, reducedSignalNamesByTileDict[tile])
        else: print "   missing tile %s"%tile
else:
    print 'Making directory %s for reduced signals'%reduced_signal_directory
    os.makedirs(reduced_signal_directory) #create a new output directory if it doesn't exist
    
if np.sum(already_exist) < len(filteredCPseqFilenameDict):
    # initiate multiprocessing

    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    for i, tile in enumerate(tileList):
        
        # check if that file already exists in directory
        if not already_exist[i]:
            # define filenames
            cpSignalFilename = signalNamesByTileDict[tile]
            reducedCPsignalFilename = reducedSignalNamesByTileDict[tile]
            print "Making reduced signal file %s from %s with filter set %s"%(reducedCPsignalFilename, cpSignalFilename, filterSet)
            # make CP signal files
            workerPool.apply_async(IMlibs.reduceCPsignalFile, args=(cpSignalFilename, filterSet, reducedCPsignalFilename))
        else:
            print "Signal file %s already exists"%currCPseqSignalOut
    workerPool.close()
    workerPool.join()


################ Map to variants ################
if args.barcode_dir is not None:
    barcode_directory = args.barcode_dir
else:
    barcode_directory = os.path.join(reduced_signal_directory,'barcode_mapping')

print 'Concatenating reduced CPsignal files, sorting, and finding barcodes in "%s"'%barcode_directory

if os.path.isdir(barcode_directory): #if the CPsignal file is already present
    print 'Barcode Directory "' + barcode_directory + '" already exists...'
else:
    print 'Making directory %s for barcodes'%barcode_directory
    os.makedirs(barcode_directory) #create a new output directory if it doesn't exist

# sort and concatenate
sortedAllCPsignalFile = IMlibs.getAllSortedCPsignalFilename(reducedSignalNamesByTileDict, barcode_directory)
if os.path.isfile(sortedAllCPsignalFile):
    print 'Sorted and concatenated CPsignal file exists "%s". Skipping...'%sortedAllCPsignalFile
else:
    IMlibs.sortConcatenateCPsignal(reducedSignalNamesByTileDict, parameters.barcode_col, sortedAllCPsignalFile)

# map barcodes to consensus


"""

# now save binding curve as text file, send to matlab to do constrained fitting, and return parameters and rmse
# find binding curves
xvalueFilenameFull = os.path.join(signal_directory, xvalueFilename)
np.savetxt(xvalueFilename, concentrations)

# reduce CPsignal file
# must have filterset and all cluster signal can't be nan
filterSet = args.filter_set
reducedSignalNamesByTileDict = IMlibs.getReducedCPsignalDictFromCPsignalDict(signalNamesByTileDict, filterSet)
bindingSeriesFilenameDict = IMlibs.getBSfromCPsignal(reducedSignalNamesByTileDict)
for tile in tileList:
    cpSignalFilename = signalNamesByTileDict[tile]
    reducedCPsignalFilename = reducedSignalNamesByTileDict[tile]
    bindingSeriesFilename = bindingSeriesFilenameDict[tile]
    # reduce
    IMlibs.reduceCPsignalFile(cpSignalFilename, filterSet, reducedCPsignalFilename)
    # get binding series
    bindingSeries, allClusterImage = IMlibs.loadBindingCurveFromCPsignal(reducedCPsignalFilename)
    np.savetxt(bindingSeriesFilename, bindingSeries)

allFilters = np.unique(np.loadtxt(signalNamesByTileDict[tileList[0]], usecols=(1,), dtype=str))
bindingSeriesAll = ['']*3
bindingSeriesAllNotNorm = ['']*3
for i, filterSet in enumerate(allFilters[[0, 2, 19]]):
    reducedSignalNamesByTileDict = IMlibs.getReducedCPsignalDictFromCPsignalDict(signalNamesByTileDict, filterSet)
    bindingSeriesFilenameDict = IMlibs.getBSfromCPsignal(reducedSignalNamesByTileDict)
    tile = '015'
    cpSignalFilename = signalNamesByTileDict[tile]
    reducedCPsignalFilename = reducedSignalNamesByTileDict[tile]
    bindingSeriesFilename = bindingSeriesFilenameDict[tile]
    # reduce
    #IMlibs.reduceCPsignalFile(cpSignalFilename, filterSet, reducedCPsignalFilename)
    # get binding series
    bindingSeries, allClusterImage = IMlibs.loadBindingCurveFromCPsignal(reducedCPsignalFilename)
    bindingSeriesAll[i] = bindingSeries
    bindingSeriesAllNotNorm[i] = bindingSeries*np.vstack(allClusterImage)
plt.figure()
histogram.compare([bindingSeriesAll[i][:,-1] for i in range(3)], allFilters[[0, 2, 19]], xbins=np.arange(0, 2.5, 0.1)-0.05)
plt.figure()
histogram.compare([bindingSeriesAllNotNorm[i][:,-1] for i in range(3)], labels = allFilters[[0, 2, 19]])


fitParametersFilenameDict = IMlibs.getFPfromCPsignal(reducedSignalNamesByTileDict)
workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
for tile in tileList:
    bindingSeriesFilename = bindingSeriesFilenameDict[tile]
    fitParametersFilename = fitParametersFilenameDict[tile]
    # send to matlab for constrained fitting
    IMlibs.findKds(bindingSeriesFilename, xvalueFilename, fitParametersFilename, parameters.fmax_min, parameters.fmax_max, parameters.fmax_initial, parameters.kd_min, parameters.kd_max, parameters.kd_initial)
"""