"""
useful functions for analyzig CP fluor files
"""

import os
import sys
import time
import re
import uuid
import subprocess
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scikits.bootstrap import bootstrap
import functools
import datetime
import lmfit
import scipy.stats as st
from statsmodels.sandbox.stats.multicomp import multipletests
from joblib import Parallel, delayed
import warnings
import pickle
import seqfun
import seaborn as sns
import itertools
import fitFun
import fitBindingCurve
import findSeqDistribution


def filenameMatchesAListOfExtensions(filename, extensionList=None):
    # from CPlibs
    if extensionList is not None:
        for currExt in extensionList:
            if filename.lower().endswith(currExt.lower()):
                return True
    return False

def findTileFilesInDirectory(dirPath, extensionList, excludedExtensionList=None):
    # from CPlibs
    dirList = os.listdir(dirPath)

    filenameDict = {}
    for currFilename in dirList:
        if filenameMatchesAListOfExtensions(currFilename,extensionList) and not filenameMatchesAListOfExtensions(currFilename,excludedExtensionList):
            currTileNum = getTileNumberFromFilename(currFilename)
            if currTileNum != '':
                filenameDict[currTileNum] = os.path.join(dirPath,currFilename)
    if len(filenameDict) == 0:
        print '      NONE FOUND'
    else:
        for currTile,currFilename in filenameDict.items():
            print '      found tile ' + currTile + ': "' + currFilename + '"'
    return filenameDict

def getTileNumberFromFilename(inFilename):
    # from CPlibs
    (path,filename) = os.path.split(inFilename) #split the file into parts
    (basename,ext) = os.path.splitext(filename)
    matches = re.findall('tile[0-9]{1,3}',basename.lower())
    tileNumber = ''
    if matches != []:
        tileNumber = '{:03}'.format(int(matches[-1][4:]))
    return tileNumber

def loadMapFile(mapFilename):

    #load map file. first line gives the root directory
    #next line is the all cluster image.
    #The remainder are filenames to compare with associated concentrations

    with open(mapFilename) as f:
        # first line is the root directory
        root_dir = f.readline().strip()
            
        # second line is the all cluster image directory
        red_dir = [os.path.join(root_dir, f.readline().strip())]
        
        # the rest are the test images and associated concetrnations
        remainder = f.readlines()
        num_dirs = len(remainder)
        directories = [os.path.join(root_dir, line.strip()) for line in remainder]

    return red_dir, np.array(directories)

def makeSignalFileName(directory, fluor_filename):
    return os.path.join(directory, '%s.signal'%os.path.splitext(os.path.basename(fluor_filename))[0])

def getFluorFileNames(directories, tileNames):
    """
    return dict whose keys are tile numbers and entries are a list of CPfluor files by concentration
    """
    filenameDict = {}
    filesPerTile = ['']*len(directories)
    for tile in tileNames:
        filenameDict[tile] = ['']*len(directories)
        for i, directory in enumerate(directories):
            try:
                filename = subprocess.check_output('find %s -maxdepth 1 -name "*CPfluor" -type f | grep tile%s'%(directory, tile), shell=True).strip()
            except: filename = ''
            filenameDict[tile][i] = filename
    return filenameDict

def parseTimeStampFromFilename(CPfluorFilename):
    try: 
        timestamp=CPfluorFilename.strip('.CPfluor').split('_')[-1]
        date, time = timestamp.split('-')
        year, month, day = np.array(date.split('.'), dtype=int)
        hour, minute, second, ms = np.array(time.split('.'), dtype=int)
        timestamp_object = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=ms*1000)
    except ValueError:
        timestamp_object = datetime.datetime.now()
    return timestamp_object
    

def getFluorFileNamesOffrates(directories, tileNames):
    """
    return dict whose keys are tile numbers and entries are a list of CPfluor files by concentration
    """
    filenameDict = {}
    
    for tile in tileNames:
        filenameDict[tile] = []
        for i, directory in enumerate(directories):
            try:
                filenames = subprocess.check_output(('find %s -maxdepth 1 -name "*CPfluor" | '+
                                                     'grep tile%s | sort')%(directory, tile),
                                                 shell=True).split()
                for filename in filenames: 
                    filenameDict[tile].append(filename)
            except:
                pass
    
    # check time stamps
    timeStampDict = {}
    for tile in tileNames:
        timeStampDict[tile] = [parseTimeStampFromFilename(filename)
                               for filename in filenameDict[tile]]
    allTimeStamps = []
    for times in timeStampDict.values():
        for time in times:
            allTimeStamps.append(time)
    allTimeStamps.sort()
    
    ## add a blank directory for every missing time stamp
    #allFilenameDict = {}
    #for tile in tileNames:
    #    filenames = filenameDict[tile] 
    #    timestamps = timeStampDict[tile]
    #    
    #    allfilename = ['']*len(allTimeStamps)
    #    for i, idx in enumerate(np.searchsorted(allTimeStamps, timestamps)):
    #        allfilename[idx] = filenames[i]
    #    allFilenameDict[tile] = list(allfilename)
    
    # and return list of time deltas
    timeDeltaDict = {}
    for tile, timeStamps in timeStampDict.items():
        timeDeltaDict[tile] = [getTimeDelta(time, allTimeStamps[0]) for time in timeStamps]
    
    return filenameDict, timeDeltaDict

def calculateSignal(df):
    return 2*np.pi*df['amplitude']*df['sigma']*df['sigma']

def getCPsignalDictfromCPseqDict(filteredCPseqFilenameDict, signal_directory):
    signalNamesByTileDict = {}
    for tiles, cpSeqFilename in filteredCPseqFilenameDict.items():
        signalNamesByTileDict[tiles] = getCPsignalFileFromCPseq(cpSeqFilename, signal_directory)
    return signalNamesByTileDict
    
def getCPsignalFileFromCPseq(cpSeqFilename, directory=None):
    if directory is None:
        directory = os.path.dirname(cpSeqFilename)
    return os.path.join(directory,  os.path.splitext(os.path.basename(cpSeqFilename))[0] + '.CPsignal')

def getReducedCPsignalDict(signalNamesByTileDict, filterSet = None, directory = None):
    if filterSet is None:
        filterSet = 'reduced'
    if directory is None:
        find_dir = True
    else: find_dir = False
    reducedSignalNamesByTileDict = {}
    for tiles, cpSignalFilename in signalNamesByTileDict.items():
        if find_dir:
            directory = os.path.dirname(cpSignalFilename)
        reducedSignalNamesByTileDict[tiles] = os.path.join(directory, os.path.splitext('__'+os.path.basename(cpSignalFilename))[0] + '_%s.CPsignal'%filterSet)
    return reducedSignalNamesByTileDict

def getReducedCPsignalFilename(reducedSignalNamesByTileDict):
    filename = reducedSignalNamesByTileDict.values()[0]
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename).lstrip('_')

    removeIndex = basename.find('tile')
    newBasename = basename[:removeIndex] + basename[removeIndex+8:]
    return os.path.join(dirname, newBasename)

def getSignalFromCPFluor(CPfluorfilename):
    fitResults = pd.read_csv(CPfluorfilename, usecols=range(7, 12), sep=':', header=None, names=['success', 'amplitude', 'sigma', 'fit_X', 'fit_Y'] )
    signal = np.array(2*np.pi*fitResults['amplitude']*fitResults['sigma']*fitResults['sigma'])
    signal[np.array(fitResults['success']==0)] = np.nan
    return signal

def getBSfromCPsignal(signalNamesByTileDict):
    bindingSeriesFilenameDict = {}
    for tiles, cpSignalFilename in signalNamesByTileDict.items():
        bindingSeriesFilenameDict[tiles] = os.path.splitext(cpSignalFilename)[0] + '.binding_series'
    return bindingSeriesFilenameDict

def getFPfromCPsignal(signalNamesByTileDict):
    bindingSeriesFilenameDict = {}
    for tiles, cpSignalFilename in signalNamesByTileDict.items():
        bindingSeriesFilenameDict[tiles] = os.path.splitext(cpSignalFilename)[0] + '.fit_parameters'
    return bindingSeriesFilenameDict

def findCPsignalFile(cpSeqFilename, redFluors, greenFluors, cpSignalFilename, tile=None):
 
    # find signal in red
    for filename in redFluors + greenFluors:
        if os.path.exists(filename):
            num_lines = int(subprocess.check_output(('wc -l %s | '+
                                                     'awk \'{print $1}\'')
                %filename, shell=True).strip())
            # assume all files have same number of lines. May want to check that here
            break
            
    if os.path.exists(redFluors[0]):        
        signal = getSignalFromCPFluor(redFluors[0]) #just take first red fluor image
    else:
        signal = np.ones(num_lines)*np.nan
        
    # find signal in green
    signal_green = np.zeros((num_lines, len(greenFluors)))
    
    # cycle through fit green images and get the final signal
    for i, currCPfluor in enumerate(greenFluors):
        if currCPfluor == '' or not os.path.exists(currCPfluor):
            signal_green[:,i] = np.ones(num_lines)*np.nan
        else:
            signal_green[:,i] = getSignalFromCPFluor(currCPfluor)
            
    # combine signal in both
    if tile is None:
        signal_comma_format = np.array(['\t'.join([signal[i].astype(str),
                                                   ','.join(signal_green[i].astype(str))])
                                        for i in range(num_lines)])
    else:
        signal_comma_format = np.array(['\t'.join([signal[i].astype(str),
                                                   ','.join(signal_green[i].astype(str)),
                                                   str(tile)])
                                        for i in range(num_lines)])        
    cp_seq = np.ravel(pd.read_table(cpSeqFilename, delimiter='\n', header=None))
    np.savetxt(cpSignalFilename, np.transpose(np.vstack((cp_seq, signal_comma_format))), fmt='%s', delimiter='\t')      
    return signal_green

def reduceCPsignalFile(cpSignalFilename, reducedCPsignalFilename, filterPos=None):
    
    if filterPos is None:
        # use all clusters
        to_run = "awk '{print $0}' %s > %s"%(cpSignalFilename, reducedCPsignalFilename)
    else:
        # filterPos is list of filters to keep
        awkFilterText = ' || '.join(['(a[i]==\"%s\")'%s for s in filterPos])  
        # take only lines that contain the filterSet
        to_run = "awk '{n=split($2, a,\":\"); b=0; for (i=1; i<=n; i++) if (%s) b=1; if (b==1) print $0}' %s > %s"%(awkFilterText, cpSignalFilename, reducedCPsignalFilename)
        
    print to_run
    os.system(to_run)
    return

def saveTimeDeltaDict(filename, timeDeltaDict):
    with open(filename, "wb") as f:
        pickle.dump(timeDeltaDict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def loadTimeDeltaDict(filename):
    with open(filename, "rb") as f:
        timeDeltaDict = pickle.load(f)
    return timeDeltaDict
   

########## FILENAME CHANGERS ##########

def getAllSortedCPsignalFilename(reducedSignalNamesByTileDict, directory):
    filename = [value for value in reducedSignalNamesByTileDict.itervalues()][0]
    newfilename = os.path.basename(filename[:filename.find('tile')] + filename[filename.find('filtered'):])
    return os.path.join(directory, os.path.splitext(newfilename)[0] + '_sorted.CPsignal')

def getCompressedBarcodeFilename(sortedAllCPsignalFile):
    return os.path.splitext(sortedAllCPsignalFile)[0] + '.unique_barcodes'


def getFittedFilename(sequenceToLibraryFilename):
    return os.path.splitext(sequenceToLibraryFilename)[0] + '.CPfitted'

def getAnnotatedFilename(sequenceToLibraryFilename, pickled=None):
    if pickled is None:
        pickled = False
    if pickled:
        return os.path.splitext(sequenceToLibraryFilename)[0] + '.CPannot.pkl'
    else:
        return os.path.splitext(sequenceToLibraryFilename)[0] + '.CPannot'


def getFitParametersFilename(sequenceToLibraryFilename):
    return os.path.splitext(sequenceToLibraryFilename)[0] + '.fitParameters'

def getTimesFilename(annotatedSignalFilename):
    return os.path.splitext(annotatedSignalFilename)[0] + '.times'

def getPerVariantFilename(fittedBindingFilename):
    return os.path.splitext(fittedBindingFilename)[0] + '.CPvariant'

def getPerVariantBootstrappedFilename(variantFittedFilename):
    return os.path.splitext(variantFittedFilename)[0] + '.bootStrapped.CPfitted'

########## ACTUAL FUNCTIONS ##########
def sortCPsignal(cpSeqFile, sortedAllCPsignalFile, barcode_col):
    # make sure file is sorted
    mydata = pd.read_table(cpSeqFile, index_col=False, header=None)
    mydata.sort(barcode_col-1, inplace=True)
    mydata.to_csv(sortedAllCPsignalFile, sep='\t', na_rep='nan', header=False, index=False)
    return

def uniqueCPsignal(reducedCPsignalFile, outputFile,  index_col=None, pickled_output=None):
    if index_col is None:
        index_col = 'tileID'
    print '\tloading CPsignal file...'
    mydata = loadCPseqSignal(reducedCPsignalFile)
    print '\tuniqueing column %s...'%index_col
    mydata = mydata.groupby(index_col).first()
    print '\tsaving uniqued file as pickle...'
    mydata.to_csv(outputFile, sep='\t', na_rep='nan', header=False, index=True)
    if pickled_output is not None:
        mydata.to_pickle(pickled_output)
    return

def sortConcatenateCPsignal(reducedSignalNamesByTileDict, sortedAllCPsignalFile, pickled_output=None):
    # save concatenated and sorted CPsignal file
    print "Concatenating CPsignal files"
    allCPsignals = ' '.join([value for value in reducedSignalNamesByTileDict.itervalues()])
    os.system("cat %s > %s"%(allCPsignals, sortedAllCPsignalFile))
    
    print "Making sure tileIDs are unique"
    uniqueCPsignal(sortedAllCPsignalFile, sortedAllCPsignalFile,
                   pickled_output=pickled_output)
    return

def compressBarcodes(sortedAllCPsignalFile, barcode_col, seq_col, compressedBarcodeFile):
    script = 'compressBarcodes'
    to_run = "python -m %s -i %s -o %s -c %d -C %d"%(script, sortedAllCPsignalFile, compressedBarcodeFile, barcode_col, seq_col)
    print to_run
    os.system(to_run)
    return


def splitAndFit(bindingSeries, concentrations, fitParameters, numCores, index=None, mod_fmin=None, split=None):
    if index is None:
        index = bindingSeries.index
    if split is None:
        split = True
    
    if split:  
        # split into parts
        print 'Splitting clusters into %d groups:'%numCores
        # assume that list is sorted somehow
        indicesSplit = [index[i::numCores] for i in range(numCores)]
        bindingSeriesSplit = [bindingSeries.loc[indices] for indices in indicesSplit]
        printBools = [True] + [False]*(numCores-1)
        print 'Fitting binding curves:'
        fits = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(fitSetKds)(subBindingSeries, concentrations, fitParameters, mod_fmin, print_bool) for
                 subBindingSeries, print_bool in itertools.izip(bindingSeriesSplit, printBools)))
    else:
        # fit
        print 'Fitting binding curves:'
        fits = (Parallel(n_jobs=numCores, verbose=10, batch_size=4)
                (delayed(fitKds)(bindingSeries.loc[idx], concentrations, fitParameters, mod_fmin) for idx in index))


    return pd.concat(fits)

def fitSetKds(subBindingSeries, concentrations, fitParameters, mod_fmin=None, print_bool=None):
    if print_bool is None: print_bool = True
    if mod_fmin is None: mod_fmin = False
    #print print_bool
    singles = []
    for i, idx in enumerate(subBindingSeries.index):
        if print_bool:
            num_steps = max(min(100, (int(len(subBindingSeries)/100.))), 1)
            if (i+1)%num_steps == 0:
                print 'working on %d out of %d iterations (%d%%)'%(i+1,
                                                                   len(subBindingSeries.index),
                                                                   100*(i+1)/float(len(subBindingSeries.index)))
        fluorescence = subBindingSeries.loc[idx]
        singles.append(fitKds(fluorescence, concentrations, fitParameters, mod_fmin=mod_fmin))

    return pd.concat(singles)

def fitKds(fluorescence, concentrations, fitParameters, mod_fmin=None):
    if mod_fmin is None:
        mod_fmin = False
    
    fitParametersNew = fitParameters.copy()
    if mod_fmin:
        if not np.isnan(fluorescence.iloc[0]):
            # as long as the first point is measured, further constrain the fmin
            fitParametersNew.loc['upperbound', 'fmin'] = 2*fluorescence.min()   
    return pd.DataFrame(columns=[fluorescence.name], data=fitBindingCurve.fitSingleBindingCurve(concentrations, fluorescence, fitParametersNew, plot=False)).transpose()


def saveDataFrame(dataframe, filename, index=None, float_format=None):
    if index is None:
        index = False
    if float_format is None:
        float_format='%4.3f'
    dataframe.to_csv(filename, sep='\t', na_rep="NaN", index=index, float_format=float_format)
    return


def makeFittedCPsignalFile(fitParameters,annotatedSignalFilename, fittedBindingFilename, bindingSeriesNorm=None, allClusterSignal=None):
    # start at the 'barcode' column- i.e. skip all sequence info
    f = open(annotatedSignalFilename); header = np.array(f.readline().split()); f.close()
    index_start = np.where(header=='barcode')[0][0]
    index_end = len(header)
    
    # load annotated signal
    table =  pd.read_table(annotatedSignalFilename, usecols=tuple([0]+range(index_start,index_end)), index_col=0)
    
    # append fit info, binding series information if given
    table = pd.concat([table, fitParameters], axis=1)
    if bindingSeriesNorm is not None:
        table = pd.concat([table, bindingSeriesNorm], axis=1)
    if allClusterSignal is not None:
        table = pd.concat([table, pd.DataFrame(allClusterSignal, columns=['all_cluster_signal'])], axis=1)
    table.to_csv(fittedBindingFilename, index=True, header=True, sep='\t')
    return table


def loadLibraryCharacterization(filename, use_index=None):
    if use_index is not None:
        mydata = pd.read_table(filename, index_col=0)
        mydata = mydata.loc[:, 'sequence']
    else:
        mydata = pd.read_table(filename, usecols=['sequence'])
    return mydata

def loadCompressedBarcodeFile(filename):
    cols = ['sequence', 'barcode', 'clusters_per_barcode', 'fraction_consensus']
    mydata = pd.read_table(filename)
    return mydata

def loadCPseqSignal(filename, concentrations=None, index_col=None, usecols=None):
    #function to load CPseqsignal file with the appropriate headers
    # will load pickled file if extension is 'pkl'
    if os.path.splitext(filename)[1] == '.pkl':
        print 'Reading pickled file...'
        table = pd.read_pickle(filename)
        if usecols is not None:
            table = table.loc[:, usecols]
    else:
        print 'Reading from text file...'
        cols = ['tileID','filter','read1_seq','read1_quality','read2_seq',
                'read2_quality','index1_seq','index1_quality','index2_seq',
                'index2_quality','all_cluster_signal','binding_series', 'tile']
        dtypes = {}
        for col in cols:
            if col == 'all_cluster_signal':
                dtypes[col] = float
            else:
                dtypes[col] = str
        
        table = pd.read_csv(filename, sep='\t', header=None, names=cols,
                            index_col=index_col, usecols=usecols, dtype=dtypes)
    return table

def loadNullScores(backgroundTileFile, filterPos=None, filterNeg=None,
                   binding_point=None, return_binding_series=None,
                   concentrations=None):
    # find one CPsignal file before reduction. Find subset of rows that don't
    # contain filterSet
    if return_binding_series is None:
        return_binding_series = False
    if binding_point is None:
        binding_point = -1
    if filterPos is None:
        if filterNeg is None:
            print "Error: need to define either filterPos or filterNeg"
            return

    # Now load the file, specifically the signal specifed
    # by index.
    table = loadCPseqSignal(backgroundTileFile)
    table.dropna(subset=['filter'], axis=0, inplace=True)
    
    # if filterNeg is given, use only this to find background clusters
    # otherwise, use clusters that don't have any of filterPos
    if filterNeg is None:
        # if any of the positive filters are found, don't use these clusters
        subset = np.logical_not(np.asarray([[str(s).find(filterSet) > -1
                                             for s in table.loc[:, 'filter']]
            for filterSet in filterPos]).any(axis=0))
    else:
        if filterPos is None:
            # if any of the negative filters are found, use these clusters
            subset = np.asarray([[str(s).find(filterSet) > -1
                                  for s in table.loc[:, 'filter']]
                for filterSet in filterNeg]).any(axis=0)
        else:
            subset = np.asarray([[str(s).find(negOne) > -1 and not(str(s).find(posOne) > -1)
                                  for s in table.loc[:, 'filter']]
                for negOne, posOne in itertools.product(filterNeg, filterPos)]).any(axis=0)            

    binding_series = pd.DataFrame([s.split(',') for s in table.loc[subset].binding_series],
        dtype=float, index=table.loc[subset].index)
    
    if concentrations is not None:
        binding_series.columns = formatConcentrations(concentrations)
    if return_binding_series:
        return binding_series.dropna(axis=0)
    else: 
        return binding_series.iloc[:, binding_point].dropna()

def tileIntToString(currTile):
    if currTile < 10:
        tile = '00%d'%currTile
    else:
        tile = '0%d'%currTile
    return tile
    

def getTile(clusterID):
    flowcellSN,currSide,currSwath,currTile=CPlibs.parseClusterID(clusterID)
    return tileIntToString(currTile)

def getTimeDelta(timestamp_final, timestamp_initial):
    return (timestamp_final - timestamp_initial).seconds + (timestamp_final - timestamp_initial).microseconds/1E6 

def getTimeDeltas(timestamps):
    maxNumFiles = len(timestamps)
    return np.array([getTimeDelta(timestamps[i], timestamps[0]) for i in range(maxNumFiles)])
    
def getTimeDeltaBetweenTiles(timeStampDict, tile):
    allTiles = np.array(timeStampDict.keys(), dtype=int)
    minTile = tileIntToString(np.min(allTiles))
    return getTimeDelta(timeStampDict[tile][0], timeStampDict[minTile][0])

def getTimeDeltaDict(timeStampDict):

    
    timeDeltas = {}
    for tile, timeStamps in timeStampDict.items():
        timeDeltas[tile] = getTimeDeltas(timeStamps) + getTimeDeltaBetweenTiles(timeStampDict, tile)
    return timeDeltas

def splitBindingCurve(table):
    return table.binding_series.str.split(',', expand=True)


def loadBindingCurveFromCPsignal(filename, concentrations=None, subset=None, index_col=None):
    """
    open file after being reduced to the clusters you are interested in.
    find the all cluster signal and the binding series (comma separated),
    then return the binding series.
    """
    # assume index_col is the tileID
    if index_col is None:
        index_col = 'tileID'
        
    # headers will be formatted concentrations if given
    if concentrations is not None:
        formatted_concentrations = formatConcentrations(concentrations)
    else:
        formatted_concentrations = None
        
    # if extension is 'pkl', load pickled table
    print 'Loading table...'
    table = loadCPseqSignal(filename, index_col=index_col,
                            usecols=[index_col, 'binding_series'])
    if subset is not None:
        table = table.loc[subset]
        
    print 'Splitting binding series...'
    tmpFile = filename+'.tmp'
    np.savetxt(tmpFile, table.binding_series.values, fmt='%s')
    binding_series = pd.read_csv(tmpFile, header=None, names=formatted_concentrations)

    for col in binding_series:
        binding_series.loc[:, col] = binding_series.loc[:, col].astype(float)
    binding_series.index = table.index

    os.remove(tmpFile)       
    
    #tableSplit = [table.loc[index] for index in np.array_split(table.index,
    #                                                           np.ceil(len(table)/10000.))]
    #binding_series = pd.concat([splitBindingCurve(subtable)
    #                            for subtable in tableSplit])   

    all_cluster_signal = loadCPseqSignal(filename,
                                         index_col=index_col,
                                         usecols=[index_col, 'all_cluster_signal'])
    return binding_series, all_cluster_signal.all_cluster_signal

def boundFluorescence(signal, plot=None):
    # take i.e. all cluster signal and bound it 
    if plot is None: plot=False
    
    signal = signal.copy()
    
    # check if at least one element of signal is not nan
    if np.isfinite(signal).sum() > 0:    
        lowerbound = np.percentile(signal.dropna(), 1)
        upperbound = signal.median() + 5*signal.std()
        
        if plot:
            binwidth = (upperbound - lowerbound)/50.
            plt.figure(figsize=(4,3))
            sns.distplot(signal.dropna(), bins = np.arange(signal.min(), signal.max()+binwidth, binwidth), color='seagreen')
            ax = plt.gca()
            ax.tick_params(right='off', top='off')
            ylim = ax.get_ylim()
            plt.plot([lowerbound]*2, ylim, 'k:')
            plt.plot([upperbound]*2, ylim, 'k:')
            plt.xlim(0, upperbound + 2*signal.std())
            plt.tight_layout()
        signal.loc[signal < lowerbound] = lowerbound
        signal.loc[signal > upperbound] = upperbound
    
    else:
        #if they are all nan, set to 1 for division
        signal.loc[:] = 1
    return signal
    
def loadOffRatesCurveFromCPsignal(filename, timeStampDict, numCores=None):
    """
    open file after being reduced to the clusters you are interested in.
    find the all cluster signal and the binding series (comma separated),
    then return the binding series, normalized by all cluster image.
    """
    if numCores is None: numCores = 20
    print 'loading annotated signal..'
    table = pd.read_table(filename, usecols=(0,10,11))
    binding_series = np.array([np.array(series.split(','), dtype=float) for series in table['binding_series']])
    
    # get tile ids
    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    tiles = np.array(workerPool.map(getTile, np.array(table.loc[:, 'tileID'])))
    workerPool.close(); workerPool.join()

    # make an array of timeStamps   
    timeDeltas = getTimeDeltaDict(timeStampDict)
    maxNumFiles = np.max([len(timeStamp) for timeStamp in timeStampDict.values()])
    xvalues = np.ones((len(table), maxNumFiles))*np.nan
    for tile, timeDelta in timeDeltas.items():
        xvalues[tiles==tile] = timeDelta
    return binding_series, np.array(table['all_cluster_signal']), xvalues, tiles  

def loadFittedCPsignal(fittedBindingFilename, annotatedClusterFile=None,
                       bindingCurveFilename=None, pickled=None):
    if pickled is None:
        pickled = False
    
    if pickled:
        table = pd.read_pickle(fittedBindingFilename)
    else:
        table = pd.read_table(fittedBindingFilename, index_col=0)
        
    if annotatedClusterFile is not None:
        if pickled:
            a = pd.read_pickle(annotatedClusterFile)
        else:
            a = pd.read_table(annotatedClusterFile, index_col=0)
        table = pd.concat([a, table], axis=1)
        
    if bindingCurveFilename is not None:
        if pickled:
            c = pd.read_pickle(bindingCurveFilename)
        else:
            c = pd.read_table(bindingCurveFilename, index_col=0)
        table = pd.concat([table, c], axis=1)
        
    table.sort('variant_number', inplace=True)
    for param in table:
        try:
            table.loc[:, param] = table.param.astype(float)
        except:
            pass
    return table

def bindingCurve(concentrations, kd=None, dG=None, fmax=None, fmin=None):
    if fmax is None:
        fmax = 1
    if fmin is None:
        fmin = 0
    if kd is None and dG is None:
        print "Error: must define either Kd or dG"
        sys.exit()
    if kd is not None:
        return (fmax - fmin)*concentrations/(concentrations + kd)+fmin

    else:
        return (fmax - fmin)*concentrations/(concentrations + np.exp(dG / 0.582)/1e-9)+fmin

def formatConcentrations(concentrations):
    return [('%.2E'%x) for x in concentrations]

def plotSingleClusterfit(bindingSeries, fitConstrained, cluster, concentrations):
    
    params = lmfit.Parameters()
    for param in ['dG', 'fmax', 'fmin']:
        params.add(param, value=fitConstrained.loc[cluster, param])
        
        
    plt.figure(figsize=(4,4));
    plt.plot(concentrations, bindingSeries.loc[cluster], 'ko', )
    more_concentrations =  np.logspace(-2, 4, 50)  
    fit = fitFun.bindingCurveObjectiveFunction(params, more_concentrations)
    plt.plot(more_concentrations, fit, 'r')
    ax = plt.gca()
    ax.set_xscale('log')
    ax.tick_params(right='off', top='off')
    plt.xlabel('concentration (nM)')
    plt.ylabel('normalized fluorescence')
    plt.tight_layout()
    
    
def plotSingleVariantFits(table, results, variant, concentrations, plot_init=None ):
    if plot_init is None:
        plot_init = False

    params = lmfit.Parameters()
    for param in ['dG', 'fmax', 'fmin']:
        params.add(param, value=results.loc[variant, param])
    
    concentrationCols = formatConcentrations(concentrations)
    filteredTable = filterStandardParameters(table, concentrations)
    bindingSeries = filteredTable.loc[table.variant_number==variant,
                                      concentrationCols]
    # get error
    try:
        eminus, eplus = fitFun.findErrorBarsBindingCurve(bindingSeries)
    except NameError:
        eminus, eplus = [np.ones(len(concentrations))*np.nan]*2

    plt.figure(figsize=(4,4));
    plt.errorbar(concentrations, bindingSeries.median(),
                 yerr=[eminus, eplus], fmt='.', elinewidth=1,
                 capsize=2, capthick=1, color='k', linewidth=1)
    
    # plot fit
    more_concentrations =  np.logspace(-2, 4, 50)
    fit = fitFun.bindingCurveObjectiveFunction(params, more_concentrations)
    plt.plot(more_concentrations, fit, 'r')
    
    
    try:
        # find upper bound
        params_ub = lmfit.Parameters()
        for param in ['dG_lb', 'fmax_ub', 'fmin']:
            name = param.split('_')[0]
            params_ub.add(name, value=results.loc[variant, param])
        ub = fitFun.bindingCurveObjectiveFunction(params_ub, more_concentrations)
    
        # find lower bound
        params_lb = lmfit.Parameters()
        for param in ['dG_ub', 'fmax_lb', 'fmin']:
            name = param.split('_')[0]
            params_lb.add(name, value=results.loc[variant, param])
        lb = fitFun.bindingCurveObjectiveFunction(params_lb, more_concentrations)

        plt.fill_between(more_concentrations, lb, ub, color='0.5', label='95% conf int', alpha=0.5)
    except:
        pass
    if plot_init:
        params_init = lmfit.Parameters()
        for param in ['dG_init', 'fmax_init', 'fmin_init']:
            name = param.split('_')[0]
            params_init.add(name, value=results.loc[variant, param])
        init = fitFun.bindingCurveObjectiveFunction(params_init, more_concentrations)
        plt.plot(more_concentrations, init, sns.xkcd_rgb['purplish'], linestyle=':')
        
    ax = plt.gca()
    ax.set_xscale('log')
    ax.tick_params(right='off', top='off')
    plt.xlabel('concentration (nM)')
    plt.ylabel('normalized fluorescence')
    plt.tight_layout()



def getPvalueFitFilter(variant_table, p):
    variant_table.loc[:, 'pvalue'] = np.nan
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n]
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, p)
    return      


def filterStandardParameters(table, concentrations=None):

    #not_nan_binding_points = formatConcentrations([0.91, 2.73])# if first two binding points are NaN, filter
    #not_nan_binding_points = formatConcentrations(concentrations)
    #nan_filter = ~np.isnan(table.loc[:, not_nan_binding_points]).all(axis=1)
    if 'barcode_good' in table.columns:
        if len(table.barcode_good.dropna()) > 0:
            
            barcode_filter = table.loc[:, 'barcode_good']
            barcode_filter.fillna(False, inplace=True)
            print ('only using %d (%4.2f%%) clusters with good barcodes'
                   %(barcode_filter.sum(),
                   barcode_filter.sum()/float(len(barcode_filter))*100))
        return table.loc[barcode_filter]
    else:
        return table

def filterFitParameters(table):
    # fit explains at least 50% of the variance, and the variance in dG is less than 1 kcal/mol
    #index = (table.dG_var < 1)&(table.rsq > 0.5)&(table.exit_flag>0)&(table.fmax_var<table.fmax), # kcal/mol
    index = (table.rsq > 0.5)&(table.dG_stde.astype(float) < 1)&(table.fmax_stde.astype(float)<table.fmax.astype(float))
    return table.loc[index]


    

def perVariantError(measurements):
    # find subset of table that has variant number equal to variant
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bounds = bootstrap.ci(data=measurements, statfunction=np.median)
    except IndexError:
        bounds = [np.nan]*2           
    return bounds

def plotFractionFit(variant_table, binedges=None):
    # plot
    binwidth=0.01
    bins=np.arange(0,1+binwidth, binwidth)
    plt.figure(figsize=(4, 3.5))
    plt.hist(variant_table.loc[variant_table.pvalue <= 0.05].fitFraction.values, alpha=0.5, color='red', bins=bins)
    plt.hist(variant_table.loc[variant_table.pvalue > 0.05].fitFraction.values, alpha=0.5, color='grey', bins=bins)
    plt.ylabel('number of variants')
    plt.xlabel('fraction fit')
    plt.tight_layout()
    
    if binedges is None:
        binedges = np.arange(-12, -6, 0.5)
    subtable = pd.DataFrame(index=variant_table.index,
                            columns=['binned_dGs', 'pvalueFilter'],
                            data=np.column_stack([np.digitize(variant_table.dG, binedges),
                                                  variant_table.pvalue <= 0.05]))
    g = sns.factorplot(x="binned_dGs", y="pvalueFilter", data=subtable,
                order=np.unique(subtable.binned_dGs),
                color="r", kind='bar');
    g.set(ylim=(0, 1.1), );
    g.set_xticklabels(['%3.1f:%3.1f'%(i, j) for i, j in itertools.izip(binedges[:-1], binedges[1:])]+['>%4.1f'%binedges[-1]], rotation=90)
    g.set(xticks=np.arange(len(binedges)))
    g.fig.subplots_adjust(hspace=.2, bottom=0.35)
    
    
    
def plotNumber(variant_table, binedges=None):
    # plot
    if binedges is None:
        binedges = np.arange(-12, -6, 0.5)
    
    plt.figure(figsize=(4,3.5));
    plt.hist((variant_table.numTests*
              variant_table.fitFraction).values, np.arange(0, 150),
            facecolor='r', edgecolor='w', histtype='stepfilled')
    plt.xlabel('number of good clusters per variant')
    plt.ylabel('number of variants')
    plt.tight_layout()
    return
    
    
def plotErrorInBins(variant_table, binedges=None, count_binedges=None):
    if binedges is None:
        binedges = np.arange(-12, -4, 0.5)
    subtable = pd.DataFrame(index=variant_table.index,
                            columns=['binned_dGs', 'confInt', 'numTests'],
                            data=np.column_stack([np.digitize(variant_table.dG.astype(float), binedges),
                                                  (variant_table.dG_ub - variant_table.dG_lb).astype(float),
                                                  np.around(variant_table.numClusters.astype(float))]))
    subtable.loc[:, 'dG (kcal/mol)'] = ['%3.1f:%3.1f'%(binedges[bins-1], binedges[bins]-1) if bins!=len(binedges) else '>%3.1f'%binedges[-1] for bins in subtable.binned_dGs.astype(int)]
    order = ['%3.1f:%3.1f'%(binedges[bins-1], binedges[bins]-1) if bins!=len(binedges) else '>%3.1f'%binedges[-1] for bins in np.arange(1, len(binedges)+1)]
    
    
    if count_binedges is None:
        count_binedges = [0, 5, 10, 20, 40, 80]
    count_order = ['%d-%d'%(count_binedges[bins-1], count_binedges[bins]-1) if bins!=len(count_binedges) else '>%d'%count_binedges[-1] for bins in np.arange(1, len(count_binedges)+1)]
    subtable.loc[:, 'binned_counts'] = np.digitize(subtable.numTests, count_binedges) 
    subtable.loc[:, '# tests'] = ['%d-%d'%(count_binedges[bins-1], count_binedges[bins]-1) if bins!=len(count_binedges) else '>%d'%count_binedges[-1] for bins in subtable.binned_counts.astype(int)]
    with sns.axes_style('darkgrid'):
        g = sns.FacetGrid(subtable, size=2.5, aspect=1.5, col="# tests", col_wrap=3,
                          col_order=count_order )
        g.map(sns.barplot, "dG (kcal/mol)", "confInt",
                    order=order,
                    color="r")
        g.set(ylim=(0, 2), );
        g.set_xticklabels(rotation=90)
        g.fig.subplots_adjust(hspace=.2, bottom=0.25)
    
def plotFitFmaxs(fitUnconstrained, remove_outliers=None, maxdG=None, index=None, param=None):
    if param is None: param='fmax'
    if remove_outliers is None:
        remove_outliers=True
    if maxdG is None:
        maxdG = -9
    if index is None:
        index = ((fitUnconstrained.dG < maxdG)&
                 (fitUnconstrained.dG_stde < 1)&
                 (fitUnconstrained.fmax_stde < fitUnconstrained.fmax))

    fmaxUnconstrainedBest = fitUnconstrained.loc[index, param]
    if remove_outliers:
        subset = pd.Series(index=fmaxUnconstrainedBest.index, data=np.logical_not(seqfun.is_outlier(fmaxUnconstrainedBest)))
        
        print 'removed %d out of %d fit fmaxes (%4.2f%%)'%(len(subset)-subset.sum(), len(subset), (len(subset)-subset.sum())/float(len(subset)))
        fmaxUnconstrainedBest = fmaxUnconstrainedBest.loc[subset]
        
    fmax_lb, fmax_initial, fmax_upperbound = np.percentile(fmaxUnconstrainedBest, [0, 50, 100])
    
    # find maximum of probability distribution
    counts, binedges = np.histogram(fmaxUnconstrainedBest, bins=np.linspace(fmaxUnconstrainedBest.min(), fmaxUnconstrainedBest.max(), 50))
    counts = counts[1:]; binedges=binedges[1:] # ignore first bin
    idx_max = np.argmax(counts)
    if idx_max != 0 and idx_max != len(counts)-1:
        fmax_initial = binedges[idx_max+1]
        
    with sns.axes_style('white'):
        fig = plt.figure(figsize=(4,3));
        ax = fig.add_subplot(111)
        sns.distplot(fmaxUnconstrainedBest, color='r', hist_kws={'histtype':'stepfilled'}, ax=ax)
        ylim = [0, ax.get_ylim()[1]*1.1]
        ax.plot([fmax_lb]*2, ylim, 'k--')
        ax.plot([fmax_initial]*2, ylim, 'k:')
        ax.plot([fmax_upperbound]*2, ylim, 'k--')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.annotate('sigma/mu=%4.2f\n(95th-5th)/50th=%4.2f'%(
            fmaxUnconstrainedBest.std()/fmaxUnconstrainedBest.mean(),
            (np.percentile(fmaxUnconstrainedBest, 95) - np.percentile(fmaxUnconstrainedBest, 5))/np.percentile(fmaxUnconstrainedBest, 50)),
            xy=(0.90, 0.95),
                        xycoords='axes fraction',
                        horizontalalignment='right', verticalalignment='top',
                        fontsize=12)
        plt.xlim(0, np.percentile(fmaxUnconstrainedBest, 100)*1.05)
        plt.ylim(ylim)
        plt.tight_layout()
    return [fmax_lb, fmax_initial, fmax_upperbound]

def findProbableFmin(bindingSeriesNorm, qvalues, remove_outliers=None, min_q=None):

    if remove_outliers is None:
        remove_outliers=True
    if min_q is None:
        min_q = 0.75
    fminProbable = bindingSeriesNorm.loc[qvalues.index].loc[qvalues>min_q].iloc[:, 0].dropna()
    if remove_outliers:
        subset = pd.Series(index=fminProbable.index, data=np.logical_not(seqfun.is_outlier(fminProbable)))
        print 'removed %d out of %d fit fmins (%4.2f%%)'%(len(subset)-subset.sum(), len(subset), (len(subset)-subset.sum())/float(len(subset)))
        fminProbable = fminProbable.loc[subset]
    
    # initial bounds by percentile    
    fmax_lb, fmax_initial, fmax_upperbound = np.percentile(fminProbable, [0, 50, 100])
    
    # find maximum of probability distribution
    counts, binedges = np.histogram(fminProbable, bins=np.linspace(fminProbable.min(), fminProbable.max(), 50))
    counts = counts[1:]; binedges=binedges[1:] # ignore first bin
    idx_max = np.argmax(counts)
    if idx_max != 0 and idx_max != len(counts)-1:
        fmax_initial = binedges[idx_max+1]

    # only let fmin go up to ~2 std devs from initial
    #std = fminProbable.loc[fminProbable>binedges[0]].std()
    #fmax_upperbound = fmax_initial + std*2

    # plot
    with sns.axes_style('white'):
        fig = plt.figure(figsize=(4,3));
        ax = fig.add_subplot(111)
        sns.distplot(fminProbable, color='r', hist_kws={'histtype':'stepfilled'}, ax=ax)
        ylim = [0, ax.get_ylim()[1]*1.1]
        ax.plot([fmax_lb]*2, ylim, 'k--')
        ax.plot([fmax_initial]*2, ylim, 'k:')
        ax.plot([fmax_upperbound]*2, ylim, 'k--')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.annotate('sigma/mu=%4.2f\n(95th-5th)/50th=%4.2f'%(
            fminProbable.std()/fminProbable.mean(),
            (np.percentile(fminProbable, 95) - np.percentile(fminProbable, 5))/np.percentile(fminProbable, 50)),
            xy=(0.90, 0.95),
                        xycoords='axes fraction',
                        horizontalalignment='right', verticalalignment='top',
                        fontsize=12)
        plt.xlim(0, np.percentile(fminProbable, 100)*1.05)
        plt.ylim(ylim)
        plt.tight_layout()

    return [fmax_lb, fmax_initial, fmax_upperbound]