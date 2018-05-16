#!/usr/bin/env python
""" Process data into CPsignal file.

 This script requires you to already have quantified images with another pipeline.

 Inputs:
   Sequence data (.CPseq files)
   CPfluor files (.CPfluor files or directories)
   filter for which clusters you wish to further analyze

 Outputs:
   CPsignal files 

 Sarah Denny """

import sys
import os
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import datetime
import itertools
from fittinglibs import (processing, fileio)



### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='master script for processing data')
group = parser.add_argument_group('required arguments for processing data')
group.add_argument('-mf', '--map_CPfluors', required=True,
                    help='map file giving the dir names to look for CPfluor files')

group = parser.add_argument_group('optional arguments for processing data')
group.add_argument('-od','--output_dir', default="binding_curves",
                    help='save output files to here. default = ./binding_curves')
group.add_argument('-cf', '--clusters_to_keep_file',
                   help='file containing list of clusters to keep (i.e. CPseq or CPannot.pkl file)')
group.add_argument('-fp','--filter_pos', nargs='+', help='set of filters '
                    'that designate clusters to fit. In absence of -cf option, '
                    'will make a file of clusterIDs that have these filters. '
                    'If not set and -cf option not given, will keep all clusters.')
group.add_argument('-fs', '--filtered_CPseqs', 
                    help='directory that holds the filtered sequence data (CPseq).'
                    'Required if you are using -fp option.')

group = parser.add_argument_group('other settings')
group.add_argument('-n','--num_cores', type=int, default=20,
                    help='maximum number of cores to use. default=20')
group.add_argument('-r','--rates', default=False, action="store_true",
                    help='flag if you wish to process rate data instead of binding data')


if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()

numCores = args.num_cores

# make output base directory
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
    
# import directory names to analyze
print 'Finding CPfluor files in directories given in "%s"...'%args.map_CPfluors
fluorDirsAll, fluorDirsSignal = processing.loadMapFile(args.map_CPfluors)
if args.rates:
    fluorNamesByTileDict, timeDeltaDict = processing.getFluorFileNamesOffrates(fluorDirsSignal)
    timeDeltaFile = os.path.join(args.output_dir, 'rates.timeDict.p')
    processing.saveTimeDeltaDict(timeDeltaFile, timeDeltaDict)
else:
    fluorNamesByTileDict = processing.getFluorFileNames(fluorDirsSignal)
fluorNamesByTileRedDict = processing.getFluorFileNames(fluorDirsAll)



################ Make signal files ################
signalDirectory = os.path.join(args.output_dir, 'CPseries')
signalNamesByTileDict = processing.getCPseriesDictfromFluorDict(fluorNamesByTileDict,
                                                            signalDirectory)
redSignalDirectory = os.path.join(args.output_dir, 'redCPseries')
redSignalNamesByTileDict = processing.getCPseriesDictfromFluorDict(fluorNamesByTileRedDict,
                                                            redSignalDirectory)
if not os.path.exists(signalDirectory):
    os.mkdir(signalDirectory)
if not os.path.exists(redSignalDirectory):
    os.mkdir(redSignalDirectory)

for directory, fluorDirs, outputFiles in itertools.izip([signalDirectory, redSignalDirectory],
                                             [fluorNamesByTileDict, fluorNamesByTileRedDict],
                                             [signalNamesByTileDict, redSignalNamesByTileDict]):
    print 'Converting CPfluor files into CPseries files in directory "%s"'%directory
    # get outputs
    tileList = np.sort(fluorDirs.keys())

    # check if file already exists
    already_exist = np.zeros(len(outputFiles), dtype=bool)
    for i, tile in enumerate(tileList):
        already_exist[i] = os.path.isfile(outputFiles[tile])
        if already_exist[i]:
            print "   found tile %s: %s. Skipping..."%(tile, outputFiles[tile])
        else: print "   missing tile %s"%tile
    
    # parallelize making them
    if np.sum(already_exist) < len(outputFiles.values()):
        (Parallel(n_jobs=numCores, verbose=10)
            (delayed(processing.makeCPseriesFile)(outputFiles[tile],
                                              fluorDirs[tile])
                     for i, tile in enumerate(tileList) if not already_exist[i]))
    
    # check to make sure they are all made
    if np.all([os.path.exists(filename) for filename in outputFiles.values()]):
        print 'All signal files successfully generated'
    else:
        print 'Error: not all signal files successfully generated'
        for filename in np.sort(outputFiles.values()):
            if not os.path.exists(filename):
                print '\tMissing tile %s'%filename
        sys.exit()  
    
################ Make concatenated, reduced signal file ########################
reducedSignalDirectory = args.output_dir
reducedCPsignalFile = processing.getReducedCPsignalFilename(signalNamesByTileDict, reducedSignalDirectory)
#reducedRedCPsignalFile = processing.getReducedCPsignalFilename(redSignalNamesByTileDict, reducedSignalDirectory, 'red')
tileOutputFile = processing.getTileOutputFilename(signalNamesByTileDict, reducedSignalDirectory)
# if file was given that has index of clusters to keep, use only these clusters
if os.path.exists(reducedCPsignalFile): #and os.path.exists(reducedRedCPsignalFile):
    print 'All reduced signal files already generated.'
else:    
    fileWithIndexToKeep = args.clusters_to_keep_file
    if fileWithIndexToKeep is None:
        if args.filter_pos is None:
            # use all clusters
            print 'Finding using all clusters...'
            indices = None
        else:       
            # make file with positive filters
            print 'Finding clusterIDs with filter name incuding one of: {%s}'%(' '.join(args.filter_pos))
            clustersToKeepFile = os.path.join(reducedSignalDirectory, 'indices_to_keep.txt')
            print 'Finding CPseq files in directory "%s"...'%args.filtered_CPseqs
            filteredCPseqFilenameDict = processing.findTileFilesInDirectory(args.filtered_CPseqs,
                                                                        ['CPseq'])
            processing.makeIndexFile(filteredCPseqFilenameDict, args.filter_pos, clustersToKeepFile)
            indices = fileio.loadFile(clustersToKeepFile)
    else:
        print 'Using clusterIDs in %s'%(fileWithIndexToKeep)
        indices = fileio.loadFile(fileWithIndexToKeep).index
    
    print 'Concatenating CPseries files...'
    alreadySavedTiles = False
    for outputFiles, reducedOutputFile in itertools.izip([signalNamesByTileDict,],
                                                         [reducedCPsignalFile,]):
        print '\tSaving to: %s'%reducedOutputFile
        if not alreadySavedTiles:
            processing.reduceCPseriesFiles(outputFiles, reducedOutputFile, indices, tileOutputFile)
            alreadySavedTiles = True
        else:
            processing.reduceCPseriesFiles(outputFiles, reducedOutputFile, indices)   
