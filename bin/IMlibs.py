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
import fileFun

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
            if len(currFilename) == 1:
                print '      found tile ' + currTile + ': "' + currFilename[0] + '"'
                filenameDict[currTile] = currFilename[0]
            elif len(currFilename) > 1:
                filenameDict[currTile] = np.sort(currFilename)

    return filenameDict

def addIndexToDir(perTileDict, index=None):
    d = collections.defaultdict(list)
    for key, values in perTileDict.items():
        d[key] = pd.Series(values, index=index)
    return d
    
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

    return pd.Series(red_dir), pd.Series(directories, index=np.arange(len(directories))+1)

def makeSignalFileName(directory, fluor_filename):
    return os.path.join(directory, '%s.signal'%os.path.splitext(os.path.basename(fluor_filename))[0])

def getFluorFileNames(directories, tileNames):
    """ Return dict with keys tile numbers and entries list of CPfluor files."""
    d = collections.defaultdict(list)
    for idx, directory in directories.iteritems():
        new_dict = addIndexToDir((findTileFilesInDirectory(directory, ['CPfluor'])), index=[idx])
        for tile, filenames in new_dict.items():
            d[tile].append(filenames)
    # consolidate tiles
    newdict = {}
    for tile, files in d.items():
        newdict[tile] = pd.concat(files)
    return newdict 

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
    for tile in tileNames:
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

def calculateSignal(df):
    return 2*np.pi*df['amplitude']*df['sigma']*df['sigma']

def getCPsignalDictfromCPseqDict(filteredCPseqFilenameDict, signal_directory):
    signalNamesByTileDict = {}
    for tiles, cpSeqFilename in filteredCPseqFilenameDict.items():
        signalNamesByTileDict[tiles] = getCPsignalFileFromCPseq(cpSeqFilename, signal_directory)
    return signalNamesByTileDict

def getCPseriesDictfromCPseqDict(filteredCPseqFilenameDict, signal_directory):
    signalNamesByTileDict = {}
    for tiles, cpSeqFilename in filteredCPseqFilenameDict.items():
        signalNamesByTileDict[tiles] = getCPseriesFileFromCPseq(cpSeqFilename, signal_directory)
    return signalNamesByTileDict

def getCPseriesFileFromCPseq(cpSeqFilename, directory=None):
    if directory is None:
        directory = os.path.dirname(cpSeqFilename)
    return os.path.join(directory,  os.path.splitext(os.path.basename(cpSeqFilename))[0] + '.CPseries.pkl')
    
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

def getReducedCPsignalFilename(signalFilesByTile, directory, suffix=None):
    if suffix is None:
        postScript = '_reduced.CPseries.pkl'
    else:
        postScript = '_reduced_%s.CPseries.pkl'%suffix
    startFile = os.path.basename(signalFilesByTile.values()[0])
    noTileFile = startFile[:startFile.find('tile')] + startFile[startFile.find('tile')+8:]
    return os.path.join(directory, os.path.splitext(noTileFile[:noTileFile.find('.pkl')])[0] + postScript)

def getTileOutputFilename(signalFilesByTile, directory, suffix=None):
    if suffix is None:
        postScript = '_reduced.CPtiles.pkl'
    else:
        postScript = '_reduced_%s.CPtiles.pkl'%suffix
    startFile = os.path.basename(signalFilesByTile.values()[0])
    noTileFile = startFile[:startFile.find('tile')] + startFile[startFile.find('tile')+8:]
    return os.path.join(directory, os.path.splitext(noTileFile[:noTileFile.find('.pkl')])[0] + postScript)


def getSignalFromCPFluor(CPfluorfilename):
    fitResults = fileFun.loadFile(CPfluorfilename)
    signal = 2*np.pi*fitResults.amplitude*fitResults.sigma*fitResults.sigma
    signal.loc[~fitResults.success.astype(bool)] = np.nan
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
        print 'Error: no CPfluor files found! Are directories in map file correct?'
        return
            
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

def makeCPseriesFile(cpseriesfilename, fluorFiles):
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
    

def reduceCPsignalFile(cpSignalFilename, reducedCPsignalFilename, filterPos=None):
    
    if filterPos is None:
        # use all clusters
        to_run = "awk '{print $0}' %s > %s"%(cpSignalFilename, reducedCPsignalFilename)
    else:
        # filterPos is list of filters to keep
        awkFilterText = ' || '.join(['(a[i]==\"%s\")'%s for s in filterPos])  
        # take only lines that contain the filterSet
        to_run = ("awk '{n=split($2, a,\":\"); b=0; for (i=1; i<=n; i++) "
                  "if (%s) b=1; if (b==1) print $0}' %s > %s")%(awkFilterText,
                                                                cpSignalFilename,
                                                                reducedCPsignalFilename)
        
    print to_run
    os.system(to_run)
    return

def makeIndexFileNoGrep(filteredCPseqFilenameDict, filterPos, outputFile):
    # takes about 61 seconds to run on ~2 million indices
    awkFilterText = ' || '.join(['(a[i]==\"%s\")'%s for s in filterPos]) 
    call = ("cat %s | "
            "awk '{n=split($2, a,\":\"); b=0; for (i=1; i<=n; i++) if (%s) b=1; "
            "if (b==1) print $1}' > %s")%(' '.join(filteredCPseqFilenameDict.values()),
                                          awkFilterText, outputFile)
    print call
    os.system(call)
    return
    
def makeIndexFile(filteredCPseqFilenameDict, filterPos, outputFile):
    # takes about 35 seconds to run on ~2 million indices
    call = "cat %s | grep -w '%s' | cut -f1 > %s"%(' '.join(filteredCPseqFilenameDict.values()),
                                                    '\|'.join(filterPos),
                                                    outputFile)
    print call
    os.system(call)
    return
    

def reduceCPseriesFiles(outputFiles, reducedOutputFile, indices=None, tileOutputFile=None):
    # load all files in dict outputFiles
    allTiles = [fileFun.loadFile(filename) for filename in outputFiles.values()]
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
        if indices is None:
            tiles.to_pickle(tileOutputFile)
        else:
            tiles.loc[indices].to_pickle(tileOutputFile)

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
    try:
        # not sure why this error is happeneing and that should probably be fixed.
        # but hopefully this try/except clause gets around it!
        tmpFile = filename+'.tmp'
        np.savetxt(tmpFile, table.binding_series.values, fmt='%s')
        binding_series = pd.read_csv(tmpFile, header=None, names=formatted_concentrations)
        os.remove(tmpFile)
        binding_series.index = table.index
        
    except:
        # this may be slower
        binding_series = splitBindingCurve(table)
        
    for col in binding_series:
        binding_series.loc[:, col] = binding_series.loc[:, col].astype(float)
    
    all_cluster_signal = loadCPseqSignal(filename,
                                         index_col=index_col,
                                         usecols=[index_col, 'all_cluster_signal'])
    return binding_series, all_cluster_signal.all_cluster_signal


    
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

def formatConcentrations(concentrations):
    return [('%.2E'%x) for x in concentrations]  

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

def loadLibCharVariantTable(libCharFile, variantTableFile, old=None):
    if old is None:
        old = False
    
    if not old:
        libChar = pd.read_table(libCharFile).loc[:, :'ss_correct']
    else:
        libChar = pd.read_table(libCharFile)
    variant_table = pd.read_table(variantTableFile, index_col=0)
    
    final = pd.DataFrame(index=libChar.index, columns=libChar.columns.tolist() +
                         variant_table.columns.tolist())
    final.loc[:, libChar.columns] = libChar
    final.loc[variant_table.index, variant_table.columns] = variant_table
    return final
    
    
    
    

  