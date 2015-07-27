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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import datetime
import glob
import IMlibs
import findSeqDistribution
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='master script for the fitting '+
                                 'clusters to binding curves pipeline')
group = parser.add_argument_group('required arguments for fitting single clusters')
group.add_argument('-fs', '--filtered_CPseqs', required=True,
                    help='directory that holds the filtered sequence data (CPseq)')
group.add_argument('-mf', '--map_CPfluors', required=True,
                    help='map file giving the dir names to look for CPfluor files')

group = parser.add_argument_group('optional arguments for fitting single clusters')
group.add_argument('-od','--output_dir', default="binding_curves",
                    help='save output files to here. default = ./binding_curves')
group.add_argument('-fp','--filterPos', nargs='+', help='set of filters '
                    'that designate clusters to fit. If not set, use all')                        
group.add_argument('-fn','--filterNeg',  nargs='+', help='set of filters '
                     'that designate "background" clusters. If not set, assume '
                     'complement to filterPos')
group.add_argument('-bp', '--binding_point', type=int, default=-1,
                    help='point in binding series to use for null scores (Default is '
                    'last concentration. -2 for second to last concentration)', )

group = parser.add_argument_group('arguments to map variants in CPsignal file')
group.add_argument('-an', '--annotated_clusters',
                   help='file containing clusterIds and indexed variant number'
                   'if not given, script will attempt to find variants with '
                   'library characterization file')
group.add_argument('-lc', '--library_characterization',
                   help='file that lists unique variant sequences')
group.add_argument('-bar', '--unique_barcodes',
                   help='barcode map file. if given, the variant sequences are '
                   'mapped to the barcode rather than directly on to the sequence data')

group = parser.add_argument_group('additional option arguments to map variants')
group.add_argument('--barcodeCol', default='index1_seq',
                   help='if using barcode map, this indicates the column of CPsignal'
                   'file giving the barcode. Default is "index1_seq"')
group.add_argument('--seqCol', default='read2_seq',
                   help='when looking for variants, look within this sequence')
group.add_argument('--noReverseComplement', default=True, action="store_false",
                   help='when looking for variants, default is to look for the'
                   'reverse complement. Flag if you want to look for forward sequence')

group = parser.add_argument_group('other settings')
group.add_argument('-n','--num_cores', type=int, default=1,
                    help='maximum number of cores to use. default=1')
group.add_argument('-gv','--fitting_parameters_path',
                    help='path to the directory in which the "fittingParameters.py" '
                    'parameter file for the run can be found')


if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()

#add global vars path to system
if args.fitting_parameters_path is not None:
    sys.path.insert(0, args.fitting_parameters_path)
import fittingParameters

# import CPseq filtered files split by tile
print 'Finding CPseq files in directory "%s"...'%args.filtered_CPseqs
filteredCPseqFilenameDict = IMlibs.findTileFilesInDirectory(args.filtered_CPseqs,
                                                            ['CPseq'])
tileList = filteredCPseqFilenameDict.keys()

# import directory names to analyze
print 'Finding CPfluor files in directories given in "%s"...'%args.map_CPfluors
fluorDirsAll, fluorDirsSignal, concentrations = IMlibs.loadMapFile(args.map_CPfluors)
fluorNamesByTileDict = IMlibs.getFluorFileNames(fluorDirsSignal, tileList)
fluorNamesByTileRedDict = IMlibs.getFluorFileNames(fluorDirsAll, tileList)

# make output base directory
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

numCores = args.num_cores

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
            workerPool.apply_async(IMlibs.findCPsignalFile, args=(currCPseqfile,
                                                                  currRedCPfluors,
                                                                  currGreenCPfluors,
                                                                  currCPseqSignalOut))
    workerPool.close()
    workerPool.join()

# check to make sure they are all made
if np.all([os.path.exists(filename) for filename in signalNamesByTileDict.values()]):
    print 'All signal files successfully generated'
else:
    print 'Error: not all signal files successfully generated'
    print '\tAre CPfluor files and CPseq files matching?'
    sys.exit()
    
    
################ Make concatenated, reduced signal file ########################
filterPos = args.filterPos
reducedSignalDirectory = os.path.join(args.output_dir, 'CPfitted')
reducedSignalNamesByTileDict = IMlibs.getReducedCPsignalDict(signalNamesByTileDict,
                                                  directory=reducedSignalDirectory)
reducedCPsignalFile = IMlibs.getReducedCPsignalFilename(reducedSignalNamesByTileDict)
print ('Making reduced CPsignal file in directory "%s"'
       %reducedSignalDirectory)

# if file already exists, skip
if os.path.exists(reducedCPsignalFile):
    print 'Reduced CPsignal file already exists. Skipping... %s'%reducedCPsignalFile
else:
    if not os.path.exists(reducedSignalDirectory):
        os.mkdir(reducedSignalDirectory)
    workerPool = multiprocessing.Pool(processes=numCores) 
    for i, tile in enumerate(tileList):
        # make temporary reduced signal file
        cpSignalFilename = signalNamesByTileDict[tile]
        reducedCPsignalFilename = reducedSignalNamesByTileDict[tile]
        print ("Making reduced signal file %s from %s with filter set %s"
               %(reducedCPsignalFilename, cpSignalFilename, filterPos))
        workerPool.apply_async(IMlibs.reduceCPsignalFile,
                               args=(cpSignalFilename, filterPos,
                                     reducedCPsignalFilename))
    workerPool.close()
    workerPool.join()
    
    if np.all([os.path.exists(filename) for filename in reducedSignalNamesByTileDict.values()]):
        print 'All temporary reduced signal files successfully generated'
    else:
        print 'Error: Not all temporary reduced signal files successfully generated. Exiting.'
        print '\tDoes filterPos exist in CPseq files?'
        sys.exit()
        
    # concatenate temp files
    try:
        IMlibs.sortConcatenateCPsignal(reducedSignalNamesByTileDict, reducedCPsignalFile)
    except:
        print 'Error: Could not concatenate reduced signal files. Exiting.'
        sys.exit()
    
    # remove temp files
    for filename in reducedSignalNamesByTileDict.values():
        os.remove(filename)
    

################ Fit ################
fittedBindingFilename = IMlibs.getFittedFilename(reducedCPsignalFile)
figDirectory = os.path.join(os.path.dirname(fittedBindingFilename),
                            'figs_%s'%str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)
    
if os.path.isfile(fittedBindingFilename):
    print 'CPfitted file exists "%s". Skipping...'%fittedBindingFilename
else:
    print 'Fitting single cluster fits "%s"...'%fittedBindingFilename
    
    # save fittedBindingFilename
    #fitParametersFilename = IMlibs.getFitParametersFilename(annotatedSignalFilename)
    #IMlibs.saveDataFrame(fitConstrained, fitParametersFilename, index=False, float_format='%4.3f')
    fitConstrained, fitParameters = fitSingleClusters.bindingSeriesByCluster(
        reducedCPsignalFile, concentrations, args.binding_point, numCores=numCores,
        signalNamesByTileDict=signalNamesByTileDict,
        filterPos=filterPos, filterNeg=filterNeg, num_clusters=None, subset=True)
    
    # try saving Figs
    try:
        plt.savefig(os.path.join(figDirectory, 'constrained_fmax.pdf')); plt.close()
    except: pass
    try:
        plt.savefig(os.path.join(figDirectory, 'constrained_fmin.pdf')); plt.close()
    except: pass
    try:
        plt.savefig(os.path.join(figDirectory, 'fluorescence_in_binding_point_column.pdf')); plt.close()
    except: pass
    try:
        plt.savefig(os.path.join(figDirectory, 'all_cluster_signal.pdf')); plt.close()
    except: pass
    
    # save CPfitted
    fitConstrained.to_csv(fittedBindingFilename, index=True, header=True, sep='\t')
    
    # save fitParameters
    fitParametersFilename = os.path.splitext(fittedBindingFilename)[0] + '.fitParameters'
    fitParameters.to_csv(fitParametersFilename, index=True, header=True, sep='\t')

################ Map to variants ################

# resulting file is a CPannotated file, which contains the clusterID, variant number,
# and bools for barcode_good if barcode was used

# first check if CPannot already exists
if args.annotated_clusters is None:
    annotatedClusterFile = IMlibs.getAnnotatedFilename(reducedCPsignalFile)
else:
    annotatedClusterFile = args.annotated_clusters
    
if os.path.exists(annotatedClusterFile):
    print 'CPannot file exists "%s". Skipping...'%annotatedClusterFile
else:
    seqMap = findSeqDistribution.findSeqMap(args.library_characterization,
                reducedCPsignalFile,
                uniqueBarcodesFile=args.unique_barcodes,
                reverseComplement=not args.noReverseComplement,
                seqCol=args.seqCol,
                barcodeCol=args.barcodeCol)
    
parameters = fittingParameters.Parameters()
barcode_col = parameters.barcode_col   # columns of CPseq that contains barcodes
sequence_col = parameters.sequence_col # columns of CPseq that contain sequences

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
    IMlibs.sortConcatenateCPsignal(reducedSignalNamesByTileDict, barcode_col, sortedAllCPsignalFile)

# map barcodes to consensus if unique barcode file isn't given in arguments
if args.unique_barcode is None:
    
    # use either the sortedAllCPsignal file or the CPseq file in arguments
    if args.CPseq is None:
        currCPseqfile = sortedAllCPsignalFile
    else:
        print 'Sorting given CPseq file %s...'%args.CPseq
        currCPseqfile = IMlibs.getSortedFilename(args.CPseq)
        IMlibs.sortCPsignal(args.CPseq, currCPseqfile, parameters.barcode_col)
        
    compressedBarcodeFile = IMlibs.getCompressedBarcodeFilename(currCPseqfile)
else:
    compressedBarcodeFile = args.unique_barcode
    
# make compressed barcode file if it doesn't exist
if os.path.isfile(compressedBarcodeFile):
    print 'Compressed barcode file exists "%s". Skipping...'%compressedBarcodeFile
else:
    print 'Making compressed barcode file "%s from %s"...'%(compressedBarcodeFile, currCPseqfile)
    IMlibs.compressBarcodes(currCPseqfile, barcode_col, sequence_col, compressedBarcodeFile)

    

# map barcode to sequence
barcodeToSequenceFilename = IMlibs.getBarcodeMapFilename(sortedAllCPsignalFile)
if os.path.isfile(barcodeToSequenceFilename):
    print 'barcode to sequence map file exists "%s". Skipping...'%barcodeToSequenceFilename
else:
    print 'Making Barcode to sequence map "%s"...'%barcodeToSequenceFilename
    IMlibs.barcodeToSequenceMap(compressedBarcodeFile, args.library_characterization, barcodeToSequenceFilename)

# map barcode to CP signal
annotatedSignalFilename = IMlibs.getAnnotatedSignalFilename(sortedAllCPsignalFile)
if os.path.isfile(annotatedSignalFilename):
    print 'annotated CPsignal file already exists "%s". Skipping...'%annotatedSignalFilename
else:
    print 'Making annotated CPsignal file "%s"...'%annotatedSignalFilename
    IMlibs.matchCPsignalToLibrary(barcodeToSequenceFilename, sortedAllCPsignalFile, annotatedSignalFilename)

sys.exit()

################ Reduce into Variant Table ################
variantFittedFilename = IMlibs.getPerVariantFilename(fittedBindingFilename)
if os.path.isfile(variantFittedFilename):
    print 'per variant fitted CPsignal file exists "%s". Skipping...'%variantFittedFilename
else:
    print 'Making per variant table from %s...'%fittedBindingFilename
    table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
    variant_table = IMlibs.findVariantTable(table, concentrations=concentrations)
    IMlibs.saveDataFrame(variant_table, variantFittedFilename, float_format='%4.3f', index=True)

# now generate boostrapped errors
variantBootstrappedFilename = IMlibs.getPerVariantBootstrappedFilename(variantFittedFilename)
if os.path.isfile(variantBootstrappedFilename):
    print 'bootstrapped error file exists "%s". Skipping...'%variantBootstrappedFilename
else:
    print 'Making boostrapped errors from %s...'%variantFittedFilename
    variant_table = pd.read_table(variantFittedFilename, index_col=0)
    table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
    parameters = fittingParameters.Parameters(concentrations, table=table)
    plt.savefig(os.path.join(os.path.dirname(fittedBindingFilename), 'stde_fmax.at_different_n.pdf'))
    
    results = IMlibs.getBootstrappedErrors(table, parameters, numCores)
    variant_table_final = IMlibs.matchTogetherResults(variant_table, results)
    IMlibs.saveDataFrame(variant_table_final, variantBootstrappedFilename, float_format='%4.3f', index=True)
    IMlibs.saveDataFrame(results, os.path.splitext(variantBootstrappedFilename)[0]+
                         '.fitParameters', float_format='%4.3f', index=True)

# do single cluster fits on subset that fit well earlier


sys.exit()

# load background binding curves
checkBackground = False
fittedBackgroundFilename = '%s.background%s'%(os.path.splitext(fittedBindingFilename))
variantBackgroundFilename = '%s.background%s'%(os.path.splitext(variantBootstrappedFilename))
if checkBackground:
    
    # load all cluster signal in positive clsuters, trim it as before, and use it to estimate binding Series of 'null' clusters
    tmp, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(annotatedSignalFilename, concentrations)
    del tmp
    IMlibs.boundFluorescence(allClusterSignal, plot=True)   # try to reduce noise by limiting how big/small you divide by
    
    # load background clusters
    bindingSeries = IMlibs.loadNullScores(signalNamesByTileDict, filterSet, tile='003', return_binding_series=True, concentrations=concentrations)
    bindingSeriesNorm = np.divide(bindingSeries, allClusterSignal.median())
    ecdf = ECDF(pd.Series(bindingSeries.iloc[:, -1]).dropna())
    qvalues = pd.Series(1-ecdf(bindingSeries.iloc[:, -1].dropna()), index=bindingSeries.iloc[:, -1].dropna().index)

    # find most recent fp
    filenames = glob.glob(os.path.join(os.path.dirname(fittedBindingFilename), "*.fp"))
    initial_file = filenames[0]
    initial_date = datetime.datetime.strptime(os.path.basename(initial_file.split('.')[1]), "%Y-%m-%d_%H-%M-%S")
    final_file = initial_file
    final_date = initial_date
    for filename in filenames:
        new_date = datetime.datetime.strptime(os.path.basename(filename.split('.')[1]), "%Y-%m-%d_%H-%M-%S")
        if new_date > final_date:
            final_date = new_date
            final_file = filename
    fitParameters = pd.read_table(final_file, index_col=0)
    
    # fit 10,000 background clusters
    index_all = bindingSeriesNorm.dropna(axis=0, thresh=4).index
    fitBackground = IMlibs.splitAndFit(bindingSeriesNorm, concentrations,
                                                       parameters.fitParameters, numCores, index=index_all)
    fitBackground.loc[:, 'qvalue'] = qvalues.loc[index_all]
    tableBackground = pd.concat([pd.DataFrame(index=bindingSeries.index, columns=['variant_number']),
                                 bindingSeriesNorm,
                                 pd.DataFrame(index=bindingSeries.index, columns=['all_cluster_signal'], data=allClusterSignal.median()),
                                 fitBackground], axis=1)
    tableBackground.to_csv(fittedBackgroundFilename, sep='\t')
    # do subsets
    table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
    variant_table = pd.read_table(variantBootstrappedFilename, index_col=0)
    parameters = fittingParameters.Parameters(concentrations, table=table)
    
    
    tableBackground.loc[:, 'variant_number'] = np.nan
    
    num_iterations = 1000
    remaining_indexes = pd.Series(index_all.values)
    # assign some random variant numbers   
    for i in np.arange(num_iterations):
        
        n_clusters = int(random.choice(variant_table.numClusters.dropna().values))
        clusters = random.sample(remaining_indexes, n_clusters)
        remaining_indexes = remaining_indexes[~np.in1d(remaining_indexes, clusters)]
        if i%100==0: print i
        #print len(remaining_indexes)
        tableBackground.loc[clusters, 'variant_number'] = i
    results = IMlibs.getBootstrappedErrors(tableBackground.dropna(subset=['variant_number'], axis=0), parameters, numCores)
    results.to_csv(variantBackgroundFilename, sep='\t', index=True, header=True, float_format='%.3e')
    tableBackground.to_csv(fittedBackgroundFilename, sep='\t', float_format='%.3e')
    
# plot some key stuff
IMlibs.plotFractionFit(variant_table, binedges=None)
