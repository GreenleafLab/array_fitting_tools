import os
import sys
import time
import re
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.stats as st
import warnings
import collections
import pickle
import seaborn as sns
import itertools
import fittinglibs.fileio as fileio

def filenameMatchesAListOfExtensions(filename, extensionList=None):
    """Return filenames that have a spcified set of extensions."""
    if extensionList is not None:
        for currExt in extensionList:
            if filename.lower().endswith(currExt.lower()):
                return True
    return False

def findTileFilesInDirectory(dirPath, extensionList, excludedExtensionList=None):
    """Return set of files in a directory as a dictionary indexed by tile."""
    dirList = os.listdir(dirPath)

    filenameDict = collections.defaultdict(list)
    for currFilename in dirList:
        if (filenameMatchesAListOfExtensions(currFilename,extensionList) and not
            filenameMatchesAListOfExtensions(currFilename,excludedExtensionList)):
            currTileNum = getTileNumberFromFilename(currFilename)
            if currTileNum != '':
                filenameDict[currTileNum].append(os.path.join(dirPath,currFilename))
    if len(filenameDict) == 0:
        print '      NONE FOUND'
    else:
        for currTile,currFilename in filenameDict.items():
            filenameDict[currTile] = np.sort(currFilename)

    return filenameDict

def addIndexToDir(perTileDict, index=None):
    """Take in a dictionary with keys=tile and entries=lest of filenames. Return the entries of the dict as a series indexed by index."""
    d = collections.defaultdict(list)
    for key, values in perTileDict.items():
        d[key] = pd.Series(values, index=index)
    return d
    
def getTileNumberFromFilename(inFilename):
    """Return the tile from the filename.
    # from CPlibs
    """
    (path,filename) = os.path.split(inFilename) #split the file into parts
    (basename,ext) = os.path.splitext(filename)
    matches = re.findall('tile[0-9]{1,3}',basename.lower())
    tileNumber = ''
    if matches != []:
        tileNumber = '{:03}'.format(int(matches[-1][4:]))
    return tileNumber

def loadMapFile(mapFilename):
    """Load map file. first line gives the root directory
    next line is the all cluster image.
    The remainder are filenames to compare with associated concentrations.
    """
    with open(mapFilename) as f:
        # first line is the root directory
        root_dir = f.readline().strip()
            
        # second line is the all cluster image directory
        red_dir = [os.path.join(root_dir, f.readline().strip())]
        
        # the rest are the test images and associated concetrnations
        remainder = f.readlines()
        num_dirs = len(remainder)
        directories = [os.path.join(root_dir, line.strip()) for line in remainder]

    return pd.Series(red_dir), pd.Series(directories, index=np.arange(len(directories)))

# Target for deletion
#def makeSignalFileName(directory, fluor_filename):
#    return os.path.join(directory, '%s.signal'%os.path.splitext(os.path.basename(fluor_filename))[0])

def getFluorFileNames(directories):
    """Return dict with keys tile numbers and entries list of CPfluor files."""
    d = collections.defaultdict(list)
    for idx, directory in directories.iteritems():
        # find dict with keys=tile and values=filenames with extension 'CPfluor' in directory
        filenames = findTileFilesInDirectory(directory, ['CPfluor'])
        if np.any([len(val)>1 for val in filenames.itervalues()]):
            print 'Error: more than one CPfluor file exists in directory: ',directory
            print 'for tiles:'
            for key, val in filenames.iteritems():
                if len(val) > 1:
                    print '\t', key
            print 'using the newest file per tile.'
            for key, val in filenames.iteritems():
                if len(val) > 1:
                    filenames[key] = np.sort(val)[-1]
        #if length is greater than 1, throw error
        new_dict = addIndexToDir(filenames, index=[idx])
        for tile, filenames in new_dict.items():
            d[tile].append(filenames)
    # consolidate tiles
    newdict = {}
    for tile, files in d.items():
        newdict[tile] = pd.concat(files)
    return newdict 

def parseTimeStampFromFilename(CPfluorFilename):
    """Return time stamp from a standardized filename."""
    try: 
        timestamp=CPfluorFilename.strip('.CPfluor').split('_')[-1]
        date, time = timestamp.split('-')
        year, month, day = np.array(date.split('.'), dtype=int)
        hour, minute, second, ms = np.array(time.split('.'), dtype=int)
        timestamp_object = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=ms*1000)
    except ValueError:
        timestamp_object = datetime.datetime.now()
    return timestamp_object
    

def getFluorFileNamesOffrates(directories):
    """Return dict whose keys are tile numbers and entries are a list of CPfluor files by concentration"""
    filenameDict = collections.defaultdict(list)
    for idx, directory in directories.iteritems():
        new_dict = addIndexToDir((findTileFilesInDirectory(directory, ['CPfluor'])))
        for tile, filenames in new_dict.items():
            filenameDict[tile].append(filenames)
    
    # make sure is flat
    for tile, filenames in filenameDict.items():
        filenameDict[tile] = pd.concat(filenameDict[tile])
    
    # check time stamps
    timeStampDict = {}
    for tile in filenameDict.keys():
        timeStampDict[tile] = [parseTimeStampFromFilename(filename)
                               for idx, filename in filenameDict[tile].iteritems()]
    allTimeStamps = []
    for times in timeStampDict.values():
        for time in times:
            allTimeStamps.append(time)
    allTimeStamps.sort()

    # and return list of time deltas
    timeDeltaDict = {}
    for tile, timeStamps in timeStampDict.items():
        timeDeltaDict[tile] = [getTimeDelta(time, allTimeStamps[0]) for time in timeStamps]
    
    return filenameDict, timeDeltaDict

def getCPseriesDictfromFluorDict(fluorDict, signal_directory):
    """From dictionary of CPfluor files, return the CPseries filenames in dict."""
    signalNamesByTileDict = {}
    for tiles, cpFluorFilenames in fluorDict.items():
        signalNamesByTileDict[tiles] = getCPseriesFileFromCPfluorTimeStamped(cpFluorFilenames.iloc[0], signal_directory)
    return signalNamesByTileDict

def getCPseriesFileFromCPfluorTimeStamped(cpFluorFilename, directory=None):
    """For a given cpfluor filename (i.e. per tile), return a new filename with the timestamp removed."""
    if directory is None:
        directory = os.path.dirname(cpFluorFilename)
    ext = '.CPseries.pkl'
    basename = os.path.basename(cpFluorFilename)
    match = re.search(r'\d{4}.\d{2}.\d{2}-\d{2}.\d{2}', basename)
    if match is None:
        index = len(os.path.splitext(basename)[0]) 
    else:
        index = match.start() - 1
    return os.path.join(directory, os.path.basename(cpFluorFilename)[:index] + ext)


def getCPseriesFileFromCPseq(cpSeqFilename, directory=None):
    """For a given cpfluor filename (i.e. per tile), return a new filename with CPseries extension."""
    if directory is None:
        directory = os.path.dirname(cpSeqFilename)
    return os.path.join(directory,  os.path.splitext(os.path.basename(cpSeqFilename))[0] + '.CPseries.pkl')

def getReducedCPsignalFilename(signalFilesByTile, directory, suffix=None):
    """From CPseries dict, indexed per tile, return a single filename that will represent all tiles."""
    if suffix is None:
        postScript = '_reduced.CPseries.pkl'
    else:
        postScript = '_reduced_%s.CPseries.pkl'%suffix
    startFile = os.path.basename(signalFilesByTile.values()[0])
    noTileFile = startFile[:startFile.find('tile')] + startFile[startFile.find('tile')+8:]
    return os.path.join(directory, os.path.splitext(noTileFile[:noTileFile.find('.pkl')])[0] + postScript)

def getTileOutputFilename(signalFilesByTile, directory, suffix=None):
    """From CPseries dict, indexed per tile, return a single filename that will represent all tiles."""
    if suffix is None:
        postScript = '_reduced.CPtiles.pkl'
    else:
        postScript = '_reduced_%s.CPtiles.pkl'%suffix
    startFile = os.path.basename(signalFilesByTile.values()[0])
    noTileFile = startFile[:startFile.find('tile')] + startFile[startFile.find('tile')+8:]
    return os.path.join(directory, os.path.splitext(noTileFile[:noTileFile.find('.pkl')])[0] + postScript)


def getSignalFromCPFluor(CPfluorfilename):
    """Starting from CPfluor, determine integrated signal."""
    fitResults = fileio.loadFile(CPfluorfilename)
    signal = 2*np.pi*fitResults.amplitude*fitResults.sigma*fitResults.sigma
    signal.loc[~fitResults.success.astype(bool)] = np.nan
    return signal

def makeCPseriesFile(cpseriesfilename, fluorFiles):
    """Starting with CPfluor files, combine to get CPseries."""
    signals = []
    signal_file_exists = [False]*len(fluorFiles)
    
    # find signal for each fluor file
    for i, (idx, filename) in enumerate(fluorFiles.iteritems()):
        if os.path.exists(filename):
            signal_file_exists[i] = True
            signals.append(pd.Series(getSignalFromCPFluor(filename), name=idx))
    
    # check any were made
    if not np.any(signal_file_exists):
        print 'Error: no CPfluor files found! Are directories in map file correct?'
        return
    
    # save
    signal = pd.concat(signals, axis=1)
    signal.to_pickle(cpseriesfilename)
    return
    
def reduceCPseriesFiles(outputFiles, reducedOutputFile, indices=None, tileOutputFile=None):
    """Concatenate the per-tile outputs and reduce to only include indices that are relevant."""
    # load all files in dict outputFiles
    allTiles = [fileio.loadFile(filename) for filename in outputFiles.values()]
    a = pd.concat(allTiles)
    a = a.groupby(level=0).first()
    
    if indices is None:    
        a.to_pickle(reducedOutputFile)
    else:
        a.loc[indices].to_pickle(reducedOutputFile)
    
    # find tile dict if tile output file is given
    if tileOutputFile is not None:
        tiles = pd.concat([pd.Series(index=s.index, data=tile)
                           for s, tile in itertools.izip(allTiles, outputFiles.keys())])
        tiles = tiles.groupby(level=0).first()
        if indices is None:
            tiles.to_pickle(tileOutputFile)
        else:
            tiles.loc[indices].to_pickle(tileOutputFile)

    return

def saveTimeDeltaDict(filename, timeDeltaDict):
    """Save the time delta dict.
    
    TODO: move to fileio.
    """
    with open(filename, "wb") as f:
        pickle.dump(timeDeltaDict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return   

def getTimeDelta(timestamp_final, timestamp_initial):
    """Return the difference between two time stamps in seconds."""
    return (timestamp_final - timestamp_initial).seconds + (timestamp_final - timestamp_initial).microseconds/1E6 

def loadCompressedBarcodeFile(filename):
    """Load the unique_barcodes file."""
    cols = ['sequence', 'barcode', 'clusters_per_barcode', 'fraction_consensus']
    mydata = pd.read_table(filename)
    return mydata

def filterFitParameters(table):
    """Filter the fit parameters. """
    index = (table.rsq > 0.5)&(table.dG_stde.astype(float) < 1)&(table.fmax_stde.astype(float)<table.fmax.astype(float))
    return table.loc[index]

def findVariantTable(table, parameter='dG', min_fraction_fit=0.25, filterFunction=filterFitParameters):
    """ Find per-variant information from single cluster fits. """
    
    # define columns as all the ones between variant number and fraction consensus
    test_stats = ['fmax', parameter, 'fmin']
    test_stats_init = ['%s_init'%param for param in ['fmax', parameter, 'fmin']]
    other_cols = ['numTests', 'fitFraction', 'pvalue', 'numClusters',
                  'fmax_lb','fmax', 'fmax_ub',
                  '%s_lb'%parameter, parameter, '%s_ub'%parameter,
                  'fmin_lb', 'fmin', 'fmin_ub', 'rsq', 'numIter', 'flag']
    
    table.dropna(subset=['variant_number'], axis=0, inplace=True)
    grouped = table.groupby('variant_number')
    variant_table = pd.DataFrame(index=grouped.first().index,
                                 columns=test_stats_init+other_cols)
    
    # filter for nan, barcode, and fit
    variant_table.loc[:, 'numTests'] = grouped.count().loc[:, parameter]
    
    fitFilteredTable = filterFunction(table)
    fitFilterGrouped = fitFilteredTable.groupby('variant_number')
    index = variant_table.loc[:, 'numTests'] > 0
    
    variant_table.loc[index, 'fitFraction'] = (fitFilterGrouped.count().loc[index, parameter]/
                                           variant_table.loc[index, 'numTests'])
    variant_table.loc[index, 'fitFraction'].fillna(0)
    # then save parameters
    old_test_stats = grouped.median().loc[:, test_stats]
    old_test_stats.columns = test_stats_init
    variant_table.loc[:, test_stats_init] = old_test_stats
    
    # null model is that all the fits are bad. 
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*
             variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n].dropna().astype(float)
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, min_fraction_fit)
    
    return variant_table
    
    

  