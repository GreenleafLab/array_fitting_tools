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
from joblib import Parallel, delayed
import multiprocessing
import shutil
import uuid
import numpy as np
import scipy.io as sio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import datetime
import glob
import IMlibs
import findSeqDistribution
import singleClusterFits
import bootStrapFits
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='master script for processing data')
group = parser.add_argument_group('required arguments for processing data')
group.add_argument('-fs', '--filtered_CPseqs', required=True,
                    help='directory that holds the filtered sequence data (CPseq)')
group.add_argument('-mf', '--map_CPfluors', required=True,
                    help='map file giving the dir names to look for CPfluor files')

group = parser.add_argument_group('optional arguments for processing data')
group.add_argument('-od','--output_dir', default="binding_curves",
                    help='save output files to here. default = ./binding_curves')
group.add_argument('-fp','--filterPos', nargs='+', help='set of filters '
                    'that designate clusters to fit. If not set, use all')                        

group = parser.add_argument_group('other settings')
group.add_argument('-n','--num_cores', type=int, default=1,
                    help='maximum number of cores to use. default=1')
group.add_argument('-r','--rates', default=False, action="store_true",
                    help='flag if you wish to process rate data instead of binding data')


if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()

# import CPseq filtered files split by tile
print 'Finding CPseq files in directory "%s"...'%args.filtered_CPseqs
filteredCPseqFilenameDict = IMlibs.findTileFilesInDirectory(args.filtered_CPseqs,
                                                            ['CPseq'])
tileList = filteredCPseqFilenameDict.keys()
numCores = args.num_cores

# import directory names to analyze
print 'Finding CPfluor files in directories given in "%s"...'%args.map_CPfluors
fluorDirsAll, fluorDirsSignal = IMlibs.loadMapFile(args.map_CPfluors)
if args.rates:
    fluorNamesByTileDict, timeDeltaDict = IMlibs.getFluorFileNamesOffrates(fluorDirsSignal, tileList)
    timeDeltaFile = os.path.join(args.output_dir, 'rates.timeDict.pkl')
    IMlibs.saveTimeDeltaDict(timeDeltaFile, timeDeltaDict)
else:
    fluorNamesByTileDict = IMlibs.getFluorFileNames(fluorDirsSignal, tileList)
fluorNamesByTileRedDict = IMlibs.getFluorFileNames(fluorDirsAll, tileList)

# make output base directory
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# save times

################ Make signal files ################
signalDirectory = os.path.join(args.output_dir, 'CPsignal')
signalNamesByTileDict = IMlibs.getCPsignalDictfromCPseqDict(filteredCPseqFilenameDict,
                                                            signalDirectory)
print 'Converting CPfluor files into CPsignal files in directory "%s"'%signalDirectory
already_exist = np.zeros(len(signalNamesByTileDict), dtype=bool)
if os.path.isdir(signalDirectory): #if the CPsignal file is already present
    print 'SignalDirectory "' + signalDirectory + '" already exists...'
    print '   finding tile .CPsignal files in directory "%s"...'%(signalDirectory)
    for i, tile in enumerate(tileList):
        already_exist[i] = os.path.isfile(signalNamesByTileDict[tile])
        if already_exist[i]:
            print "   found tile %s: %s. Skipping..."%(tile, signalNamesByTileDict[tile])
        else: print "   missing tile %s"%tile
else:
    os.makedirs(signalDirectory) #create a new output directory if it doesn't exist
  
if np.sum(already_exist) < len(filteredCPseqFilenameDict):
    
    # parallelize making the CPsignal files for each tile
    (Parallel(n_jobs=numCores, verbose=10)
        (delayed(IMlibs.findCPsignalFile)(filteredCPseqFilenameDict[tile],
                                          fluorNamesByTileRedDict[tile],
                                          fluorNamesByTileDict[tile],
                                          signalNamesByTileDict[tile],
                                          tile=tile)
                 for i, tile in enumerate(tileList) if not already_exist[i]))

    # check to make sure they are all made
    if np.all([os.path.exists(filename) for filename in signalNamesByTileDict.values()]):
        print 'All signal files successfully generated'
    else:
        print 'Error: not all signal files successfully generated'
        print '\tAre CPfluor files and CPseq files matching?'
        sys.exit()
        
    
################ Make concatenated, reduced signal file ########################
reducedSignalDirectory = args.output_dir
reducedSignalNamesByTileDict = IMlibs.getReducedCPsignalDict(signalNamesByTileDict,
                                                  directory=reducedSignalDirectory)
reducedCPsignalFile = IMlibs.getReducedCPsignalFilename(reducedSignalNamesByTileDict)
pickleCPsignalFilename = reducedCPsignalFile + '.pkl'

print ('Making reduced CPsignal file in directory "%s"'
       %reducedSignalDirectory)

# if file already exists, skip
if os.path.exists(reducedCPsignalFile):
    print 'Reduced CPsignal file already exists. Skipping... %s'%reducedCPsignalFile
elif os.path.exists(pickleCPsignalFilename):
    print 'Pickled reduced CPsignal file already exists. Skipping... %s'%pickleCPsignalFilename
else:
    if not os.path.exists(reducedSignalDirectory):
        os.mkdir(reducedSignalDirectory)
    
    # parallelize making reduced CPsignal files with filterPos
    (Parallel(n_jobs=numCores, verbose=10)
        (delayed(IMlibs.reduceCPsignalFile)(signalNamesByTileDict[tile],
                                            reducedSignalNamesByTileDict[tile],
                                            args.filterPos)
                 for i, tile in enumerate(tileList)))
    
    if np.all([os.path.exists(filename) for filename in reducedSignalNamesByTileDict.values()]):
        print 'All temporary reduced signal files successfully generated'
    else:
        print 'Error: Not all temporary reduced signal files successfully generated. Exiting.'
        print '\tDoes filterPos exist in CPseq files?'
        sys.exit()
        
    # concatenate temp files
    try:
        IMlibs.sortConcatenateCPsignal(reducedSignalNamesByTileDict,
                                       reducedCPsignalFile,
                                       pickled_output=pickleCPsignalFilename)
    except:
        print 'Error: Could not concatenate reduced signal files. Exiting.'
        sys.exit()
    
    # remove temp files
    for filename in reducedSignalNamesByTileDict.values():
        os.remove(filename)
        
    

