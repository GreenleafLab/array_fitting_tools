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
import IMlibs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import glob
from statsmodels.distributions.empirical_distribution import ECDF               
sns.set_style('white', )

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='master script for the fitting '+
                                 'clusters to binding curves pipeline')
parser.add_argument('filteredCPseqs', 
                    help='directory that holds the filtered sequence data (CPseq)')
parser.add_argument('mapCPfluors',
                    help='map file giving the dir names to look for CPfluor files')
parser.add_argument('-od','--output_dir', default="binding_curves",
                    help='save output files to here. default = ./binding_curves')
parser.add_argument('-fp','--filterPos', help='set of filters (semicolon separated) '
                    'that designate clusters to fit. If not set, use all.')                        
parser.add_argument('-fn','--filterNeg', help='set of filters (semicolon separated) '
                     'that designate "background" clusters. If not set, assume '
                     'complement to filterPos')
parser.add_argument('-n','--num_cores', type=int, default=1,
                    help='maximum number of cores to use. default=1')
parser.add_argument('-gv','--fitting_parameters_path',
                    help='path to the directory in which the "fittingParameters.py" '
                    'parameter file for the run can be found')
parser.add_argument('-nc', '--null_column', type=int, default=-1,
                    help='point in binding series to use for null scores (Default is '
                    'last concentration. -2 for second to last concentration)', )

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
print 'Finding CPseq files in directory "%s"...'%args.filtered_tile_dir
filteredCPseqFilenameDict = IMlibs.findTileFilesInDirectory(args.filtered_tile_dir,
                                                            ['CPseq'])
tileList = filteredCPseqFilenameDict.keys()

# import directory names to analyze
print 'Finding CPfluor files in directories given in "%s"...'%args.mapCPfluors
fluorDirsAll, fluorDirsSignal, concentrations = IMlibs.loadMapFile(args.mapCPfluors)
fluorNamesByTileDict = IMlibs.getFluorFileNames(fluorDirsSignal, tileList)
fluorNamesByTileRedDict = IMlibs.getFluorFileNames(fluorDirAll, tileList)

# make output base directory
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

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
    print 'not all signal files successfully generated'
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
if os.path.isfile(fittedBindingFilename):
    print 'CPfitted file exists "%s". Skipping...'%fittedBindingFilename
else:
    print 'Fitting single cluster fits "%s"...'%fittedBindingFilename
    # get binding series
    print '\tLoading binding series and all RNA signal:'
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(reducedCPsignalFile, concentrations)
    
    # make normalized binding series
    IMlibs.boundFluorescence(allClusterSignal, plot=True)   # try to reduce noise by limiting how big/small you divide by
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # find null scores and max signal
    fabs_green_max = bindingSeriesNorm.iloc[:, args.null_column]
    null_scores = IMlibs.loadNullScores(signalNamesByTileDict, filterPos=filterPos, filterNeg=filterNeg, binding_point=args.null_column)
    
    # get binding estimation and choose 10000 that pass filter
    ecdf = ECDF(pd.Series(null_scores).dropna())
    qvalues = pd.Series(1-ecdf(bindingSeries.iloc[:, args.null_column].dropna()), index=bindingSeries.iloc[:, args.null_column].dropna().index)
    qvalues.sort()
    index = qvalues.iloc[:1E4].index # take top 10K binders
    
    # fit first round
    print '\tFitting best binders with no constraints...'
    parameters = fittingParameters.Parameters(concentrations, fabs_green_max.loc[index])
    fitUnconstrained = IMlibs.splitAndFit(bindingSeriesNorm, 
                                          concentrations, parameters.fitParameters, numCores, index=index, mod_fmin=True)

    # reset fitting parameters based on results
    maxdG = parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.9, concentrations[args.null_column])) # 90% bound at 
    param = 'fmax'
    parameters.fitParameters.loc[:, param] = IMlibs.plotFitFmaxs(fitUnconstrained, maxdG=maxdG, param=param)
    plt.savefig(os.path.join(os.path.dirname(fittedBindingFilename), 'constrained_%s.pdf'%param))
    
    param = 'fmin'
    parameters.fitParameters.loc[:, param] = IMlibs.findProbableFmin(bindingSeriesNorm, qvalues)
    plt.savefig(os.path.join(os.path.dirname(fittedBindingFilename), 'constrained_%s.pdf'%param))
        
    # now refit all remaining clusters
    print 'Fitting all with constraints on fmax (%4.2f, %4.2f, %4.2f)'%(parameters.fitParameters.loc['lowerbound', 'fmax'], parameters.fitParameters.loc['initial', 'fmax'], parameters.fitParameters.loc['upperbound', 'fmax'])
    print 'Fitting all with constraints on fmin (%4.4f, %4.4f, %4.4f)'%(parameters.fitParameters.loc['lowerbound', 'fmin'], parameters.fitParameters.loc['initial', 'fmin'], parameters.fitParameters.loc['upperbound', 'fmin'])
    
    # save fit parameters
    fitParametersFilename = os.path.join(os.path.dirname(fittedBindingFilename),
                                         'bindingParameters.%s.fp'%datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    parameters.fitParameters.to_csv(fitParametersFilename, sep='\t')
    fitConstrained = pd.DataFrame(index=bindingSeriesNorm.index, columns=fitUnconstrained.columns)
    
    # sort by qvalue to try to get groups of equal distributions of binders/nonbinders
    index = pd.concat([bindingSeriesNorm, pd.DataFrame(qvalues, columns=['qvalue'])], axis=1).sort('qvalue').index
    index_all = bindingSeriesNorm.loc[index].dropna(axis=0, thresh=4).index
    fitConstrained.loc[index_all] = IMlibs.splitAndFit(bindingSeriesNorm, concentrations,
                                                       parameters.fitParameters, numCores, index=index_all)
    fitConstrained.loc[:, 'qvalue'] = qvalues

    # save fittedBindingFilename
    #fitParametersFilename = IMlibs.getFitParametersFilename(annotatedSignalFilename)
    #IMlibs.saveDataFrame(fitConstrained, fitParametersFilename, index=False, float_format='%4.3f')
    table = IMlibs.makeFittedCPsignalFile(fitConstrained,annotatedSignalFilename, fittedBindingFilename, bindingSeriesNorm, allClusterSignal)
    
    # save fit Parameters?
    # save Normalized Binding Series?



################ Map to variants ################
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
