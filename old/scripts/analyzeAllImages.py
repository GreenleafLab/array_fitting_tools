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
                   help='file containing tileIds and indexed variant number. '
                   'If not given, script will attempt to find variants with '
                   'library characterization file')
group.add_argument('-lc', '--library_characterization',
                   help='file that lists unique variant sequences')
group.add_argument('-bar', '--unique_barcodes',
                   help='barcode map file. if given, the variant sequences are '
                   'mapped to the barcode rather than directly on to the sequence data')

group = parser.add_argument_group('additional optional arguments to map variants')
group.add_argument('--barcodeCol', default='index1_seq',
                   help='if using barcode map, this indicates the column of CPsignal'
                   'file giving the barcode. Default is "index1_seq"')
group.add_argument('--seqCol', default='read2_seq',
                   help='when looking for variants, look within this sequence. '
                   'Default is "read2_seq"')
group.add_argument('--noReverseComplement', default=False, action="store_true",
                   help='when looking for variants, default is to look for the '
                   'reverse complement. Flag if you want to look for forward sequence')

group = parser.add_argument_group('other settings')
group.add_argument('-n','--num_cores', type=int, default=1,
                    help='maximum number of cores to use. default=1')
group.add_argument('-off','--off_rates', default=False, action="store_true",
                    help='flag if you wish to do off rates')
group.add_argument('-on','--on_rates', default=False, action="store_true",
                    help='flag if you wish to do on rates')

if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()

if args.off_rates and args.on_rates:
    print "Error: only one option [-off | -on] can be flagged"
    sys.exit()

# import CPseq filtered files split by tile
print 'Finding CPseq files in directory "%s"...'%args.filtered_CPseqs
filteredCPseqFilenameDict = IMlibs.findTileFilesInDirectory(args.filtered_CPseqs,
                                                            ['CPseq'])
tileList = filteredCPseqFilenameDict.keys()

# import directory names to analyze
print 'Finding CPfluor files in directories given in "%s"...'%args.map_CPfluors
fluorDirsAll, fluorDirsSignal, concentrations = IMlibs.loadMapFile(args.map_CPfluors)
if args.off_rates or args.on_rates:
    fluorNamesByTileDict, timeDelta = IMlibs.getFluorFileNamesOffrates(fluorDirsSignal, tileList)
else:
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
reducedSignalDirectory = os.path.join(args.output_dir, 'CPfitted')
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
        

    
################ Fit ################
fittedBindingFilename = IMlibs.getFittedFilename(reducedCPsignalFile) + '.pkl'
fitParametersFilename = os.path.splitext(reducedCPsignalFile)[0] + '.fitParameters'
bindingCurveFilename = os.path.splitext(reducedCPsignalFile)[0] + '.bindingCurve.pkl'
timesFilename = os.path.splitext(reducedCPsignalFile)[0] + '.times.pkl'

if not args.on_rates and not args.off_rates:
    # only fit single clusters if you are looking at binding curves (i.e. not
    # on rates or off rates)
    print 'Fitting single clusters...'

    
    # define background file: i.e CPsignal file that contains some null clusters
    backgroundTileFile = signalNamesByTileDict['003']
    
    # save figures in dated directory
    figDirectory = os.path.join(os.path.dirname(fittedBindingFilename),
                                'figs_%s'%str(datetime.date.today()))
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
        
    if os.path.isfile(fittedBindingFilename):
        print 'CPfitted file exists "%s". Skipping...'%fittedBindingFilename
    else:
        print 'Fitting single cluster fits "%s"...'%fittedBindingFilename
        
        # do single cluster fits on binding curve in reducedCPsignalFile
        fitConstrained, fitParameters, bindingSeriesNorm = singleClusterFits.bindingSeriesByCluster(
            pickleCPsignalFilename, concentrations, args.binding_point, numCores=numCores,
            backgroundTileFile=backgroundTileFile,
            filterPos=args.filterPos, filterNeg=args.filterNeg, num_clusters=None,
            subset=True)
        
        # try saving Figs
        try:
            plt.savefig(os.path.join(figDirectory, 'constrained_fmax.pdf')); plt.close()
        except: pass
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
        #fitConstrained.to_csv(fittedBindingFilename, index=True, header=True, sep='\t')
        fitConstrained.to_pickle(fittedBindingFilename)
        
        # save fitParameters
        fitParameters.to_csv(fitParametersFilename, index=True, header=True, sep='\t')
        
        # save binding series
        bindingSeriesNorm.to_pickle(bindingCurveFilename) 
else:
    print 'Skipping fitting single clusters because fitting onrates or offrates'
    if os.path.exists(bindingCurveFilename):
        print 'Binding curve file already exists. Skipping...'
    else:
        print 'Loading binding curves...'
        
        bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
                pickleCPsignalFilename)
        
        allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   
        bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
        # now find times and do the best you can to bin times and effectively
        # fill in data from different tiles measured at slightly different times
        
        # read tiles
        tiles = pd.read_pickle(pickleCPsignalFilename).loc[:, 'tile']
        
        # make array of universal times: i.e. bins in which you will test binding
        min_time_delta = np.min([(np.array(times[1:]) - np.array(times[:-1])).min()
                                 for times in timeDelta.values()])
        
        universalTimes = np.arange(0, np.hstack(timeDelta.values()).max()+min_time_delta,
                                   min_time_delta)
        
        # find which times in universal times have variants
        cols = []
        for tile, times in timeDelta.items():
            cols.append(np.searchsorted(universalTimes, times))
        finalCols = np.unique(np.hstack(cols))
            
        # remake binding series file
        timeSeriesNorm = pd.DataFrame(index=bindingSeriesNorm.index,
                                      columns=finalCols)
        for tile, times in timeDelta.items():
            print 'adding tile %s'%tile
            index = tiles == tile
            old_cols = np.arange(len(times))
            cols = np.searchsorted(universalTimes, times)
            timeSeriesNorm.loc[index, cols] = bindingSeriesNorm.loc[index,old_cols].values
        
        finalTimes = pd.Series(universalTimes[finalCols], index=finalCols)
        timeSeriesNorm.to_pickle(bindingCurveFilename)
        finalTimes.to_pickle(timesFilename)

    
################ Map to variants ################

# resulting file is a CPannotated file, which contains the clusterID, variant number,
# and bools for barcode_good if barcode was used

# first check if CPannot already exists
pickled = True
if args.annotated_clusters is None:
    annotatedClusterFile = IMlibs.getAnnotatedFilename(reducedCPsignalFile, pickled=pickled)
else:
    annotatedClusterFile = args.annotated_clusters

if os.path.exists(annotatedClusterFile):
    print 'CPannot file exists "%s". Skipping...'%annotatedClusterFile
else:
    seqMap = findSeqDistribution.findSeqMap(args.library_characterization,
                pickleCPsignalFilename,
                uniqueBarcodesFile=args.unique_barcodes,
                reverseComplement=not args.noReverseComplement,
                seqCol=args.seqCol,
                barcodeCol=args.barcodeCol)
    if len(seqMap.variant_number.dropna()) == 0:
        print ('Error: no variants found. Is reverse seq not reverse complemented?'
               ' are barcodeCol and seqCol correct? ')
        sys.exit()

    # save output
    if pickled:
        seqMap.to_pickle(annotatedClusterFile)
    else:
        seqMap.to_csv(annotatedClusterFile, sep='\t', header=True, index=True)
sys.exit()

################ Reduce into Variant Table ################
if not args.on_rates and not args.off_rates:
    # if doing binding curves, do biding curve method
    variantFittedFilename = IMlibs.getPerVariantFilename(reducedCPsignalFile)
    if os.path.isfile(variantFittedFilename):
        print 'per variant fitted CPsignal file exists "%s". Skipping...'%variantFittedFilename
    else:
        print 'Making per variant table from %s...'%fittedBindingFilename
        variant_table = bootStrapFits.fitBindingCurves(fittedBindingFilename,
                                                       annotatedClusterFile,
                                                        bindingCurveFilename,
                                                        concentrations,
                                                        pickled=pickled,
                                                        numCores=numCores,
                                                       )
        IMlibs.saveDataFrame(variant_table, variantBootstrappedFilename,
                             float_format='%4.3f', index=True)

elif args.on_rates:
    # if doing on rates, do on rate method
    
    pass
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
    bindingSeries = IMlibs.loadNullScores(backgroundTileFile, filterPos=filterPos,
                                          return_binding_series=True,
                                          concentrations=concentrations)
    bindingSeriesNorm = np.divide(bindingSeries, allClusterSignal.median())
    ecdf = ECDF(pd.Series(bindingSeries.iloc[:, -1]).dropna())
    qvalues = pd.Series(1-ecdf(bindingSeries.iloc[:, -1].dropna()),
                        index=bindingSeries.iloc[:, -1].dropna().index)

    # find most recent fp
    fitParameters = pd.read_table(fitParametersFilename, index_col=0)
    
    # fit 10,000 background clusters
    index_all = bindingSeriesNorm.dropna(axis=0, thresh=4).index
    fitBackground = IMlibs.splitAndFit(bindingSeriesNorm, concentrations,
                                       fitParameters, numCores, index=index_all)
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
