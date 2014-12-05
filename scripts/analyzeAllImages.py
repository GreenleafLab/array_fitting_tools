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
import scipy.io as sio
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
parser.add_argument('-lc','--library_characterization', help='file with the characterization data of the deisgned library', required=True)
parser.add_argument('-bd','--barcode_dir', help='save files associated with barcode mapping here. default=signal_dir_reduced/barcode_mapping')
parser.add_argument('-n','--num_cores', help='maximum number of cores to use')
parser.add_argument('-gv','--fitting_parameters_path', help='path to the directory in which the "globalvars.py" parameter file for the run can be found')

if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()

#add global vars path to system
if args.fitting_parameters_path is not None:
    sys.path.insert(0, args.fitting_parameters_path)

# import parameters
import fittingParameters
parameters = fittingParameters.Parameters()

# FUNCITONS
def collectLogs(inLog): #multiprocessing callback function to collect the output of the worker processes
    logFilename = inLog[0]
    logText = inLog[1]
    resultList[logFilename] = logText


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
    print 'Making sorted and concatenated CPsignal file "%s"...'%sortedAllCPsignalFile
    IMlibs.sortConcatenateCPsignal(reducedSignalNamesByTileDict, parameters.barcode_col, sortedAllCPsignalFile)

# map barcodes to consensus
compressedBarcodeFile = IMlibs.getCompressedBarcodeFilename(sortedAllCPsignalFile)
if os.path.isfile(compressedBarcodeFile):
    print 'Compressed barcode file exists "%s". Skipping...'%compressedBarcodeFile
else:
    print 'Making compressed barcode file "%s"...'%compressedBarcodeFile
    IMlibs.compressBarcodes(sortedAllCPsignalFile, parameters.barcode_col, parameters.sequence_col, compressedBarcodeFile)

# map barcode to sequence
barcodeToSequenceFilename = IMlibs.getBarcodeMapFilename(compressedBarcodeFile)
if os.path.isfile(barcodeToSequenceFilename):
    print 'barcode to sequence map file exists "%s". Skipping...'%barcodeToSequenceFilename
else:
    print 'Making Barcode to sequence map "%s"...'%barcodeToSequenceFilename
    IMlibs.barcodeToSequenceMap(compressedBarcodeFile, args.library_characterization, barcodeToSequenceFilename)

# map barcode to CP signal
annotatedSignalFilename = IMlibs.getAnnotatedSignalFilename(barcodeToSequenceFilename)
if os.path.isfile(annotatedSignalFilename):
    print 'annotated CPsignal file already exists "%s". Skipping...'%annotatedSignalFilename
else:
    print 'Making annotated CPsignal file "%s"...'%annotatedSignalFilename
    IMlibs.matchCPsignalToLibrary(barcodeToSequenceFilename, sortedAllCPsignalFile, annotatedSignalFilename)

# get binding series
fittedBindingFilename = IMlibs.getFittedFilename(annotatedSignalFilename)

if os.path.isfile(fittedBindingFilename):
    print 'fitted CPsignal file exists "%s". Skipping...'%fittedBindingFilename
else:
    print 'Making fitted CP signal file "%s"...'%annotatedSignalFilename
    # get binding series
    bindingSeries, allClusterImage = IMlibs.loadBindingCurveFromCPsignal(annotatedSignalFilename)
    bindingSeriesSplit = np.array_split(bindingSeries, numCores)
    bindingSeriesFilenameParts = IMlibs.getBindingSeriesFilenameParts(annotatedSignalFilename, numCores)
    fitParametersFilenameParts = IMlibs.getfitParametersFilenameParts(bindingSeriesFilenameParts)
    
    # split into parts and fit
    resultList = {}
    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
        sio.savemat(bindingSeriesFilename, {'concentrations':concentrations, 'binding_curves': bindingSeriesSplit[i]})
        workerPool.apply_async(IMlibs.findKds, args=(bindingSeriesFilename, fitParametersFilenameParts[i],
                                                     parameters.fmax_min, parameters.fmax_max, parameters.fmax_initial,
                                                     parameters.kd_min, parameters.kd_max, parameters.kd_initial,
                                                     parameters.fmin_min, parameters.fmin_max, parameters.fmin_initial),
                               callback=collectLogs)
    workerPool.close()
    workerPool.join()
    
    #print logs
    for currFile,currLog in resultList.items():
        print '[++++++++++++++++++++++++++++++ MATLAB ANALYSE IMAGE LOG FOR ' + currFile + ' ++++++++++++++++++++++++++++++]'
        print currLog[-100:]
    
    # remove binding series filenames
    IMlibs.removeFilenameParts(bindingSeriesFilenameParts)
    
    # join together parts
    fitParameters = IMlibs.joinTogetherFitParts(fitParametersFilenameParts)
    
    # save fittedBindingFilename
    fitParametersFilename = IMlibs.getFitParametersFilename(annotatedSignalFilename)
    IMlibs.saveDataFrame(fitParameters, fitParametersFilename, index=False, float_format='%4.3f')
    IMlibs.removeFilenameParts(fitParametersFilenameParts)
    IMlibs.makeFittedCPsignalFile(fitParametersFilename,annotatedSignalFilename, fittedBindingFilename)
    
# Now reduce into variants. Save the variant number, characterization info, number
# of times tested, only if fraction_consensus is greater than 0.67 (2/3rd),
# and save median of normalized binding amount, median of fit parameters, (and quartiles?),
# error in fit parameters?