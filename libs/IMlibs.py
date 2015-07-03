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
import CPlibs
import scipy.stats as st
from statsmodels.sandbox.stats.multicomp import multipletests
from joblib import Parallel, delayed
import warnings
import seqfun
import seaborn as sns
import itertools
import fitBindingCurve
sys.path.insert(0, '/home/sarah/array_image_tools_SKD')
import fittingParameters

def spawnMatlabJob(matlabFunctionCallString):
    # from CPlibs
    try:
        #construct the command-line matlab call 
        functionCallString =                      "try,"
        functionCallString = functionCallString +     matlabFunctionCallString + ';'
        functionCallString = functionCallString + "catch e,"
        functionCallString = functionCallString +     "disp(getReport(e,'extended'));"
        functionCallString = functionCallString + "end,"
        functionCallString = functionCallString + "quit;"
    
        logFilename = 'matlabProcess_' + str(uuid.uuid4()) + str(time.time()) + '.tempLog' #timestamped logfile filename
    
        cmdString ='matlab -nodesktop -nosplash -singleCompThread -r "{0}"'.format(functionCallString)
        cmdString = cmdString + ' 1>> {0}'.format(logFilename)
        cmdString = cmdString + ' 2>> {0}'.format(logFilename)
       
        print 'issuing subprocess shell command: ' + cmdString
       
        returnCode = subprocess.call(cmdString,shell=True) #execute the command in the shell
        returnCode2 = subprocess.call('stty sane',shell=True) #matlab messes up the terminal in a weird way--this fixes it 
    
        #read log file into a string
        try:
            with open(logFilename) as logFilehandle:
                logString = logFilehandle.read()
            # delete logfile
            try:
                os.unlink(logFilename)
            except OSError:
                pass
        except IOError:
            logString = 'Log file not generated for command "' + functionCallString + '".'
    
        # return log
        return logString
    except Exception, e:
        return 'Python exception generated in spawnMatlabJob: ' + e.message

def filenameMatchesAListOfExtensions(filename, extensionList):
    # from CPlibs
    for currExt in extensionList:
        if filename.lower().endswith(currExt.lower()):
            return True
    return False

def findTileFilesInDirectory(dirPath, extensionList, excludedExtensionList):
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
    """
    load map file. header prefixed by '>' gives the root directory
    next line is the all cluster image.
    The remainder are filenames to compare with associated concentrations
    """
    f = open(mapFilename)
    # first line is the root directory
    root_dir = f.readline().strip()
        
    # second line is the all cluster image directory
    red_dir = [os.path.join(root_dir, f.readline().strip())]
    
    # the rest are the test images and associated concetrnations
    remainder = f.readlines()
    num_dirs = len(remainder)
    directories = ['']*num_dirs
    concentrations = np.zeros(num_dirs)
    for i, line in enumerate(remainder):
        directories[i] = os.path.join(root_dir, line.strip().split('\t')[0])
        concentrations[i] = line.strip().split('\t')[1]
    f.close()
    return red_dir, np.array(directories), concentrations

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
                filename = subprocess.check_output('find %s -maxdepth 2 -name "*CPfluor" -type f | grep tile%s'%(directory, tile), shell=True).strip()
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
        filenameDict[tile] = np.array([],dtype=str)
        for i, directory in enumerate(directories):
            filenameDict[tile] = np.append(filenameDict[tile],
                                           np.sort(subprocess.check_output('find %s -name "*CPfluor" | grep tile%s'%(directory, tile), shell=True).split()))
    # check time stamps
    timeStampDict = {}
    for tile in tileNames:
        timeStampDict[tile] = [parseTimeStampFromFilename(filename) for filename in filenameDict[tile]]
        
    # modiy such that all have the same length. Add 1 hour to last time stamp of those files that have no data associated
    maxNumFiles = np.max([len(filenameDict[tile]) for tile in tileNames])
    for tile in tileNames:
        if len(filenameDict[tile]) < maxNumFiles:
            filenameDict[tile] = np.append(filenameDict[tile], ['']*(maxNumFiles - len(filenameDict[tile])))
            timeStampDict[tile] = np.append(timeStampDict[tile], [timeStampDict[tile][-1] +  datetime.timedelta(hours=1)]*(maxNumFiles - len(timeStampDict[tile])))
    return filenameDict, timeStampDict

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

def getReducedCPsignalDictFromCPsignalDict(signalNamesByTileDict, filterSet = None, directory = None):
    if filterSet is None:
        filterSet = 'reduced'
    if directory is None:
        find_dir = True
    else: find_dir = False
    reducedSignalNamesByTileDict = {}
    for tiles, cpSignalFilename in signalNamesByTileDict.items():
        if find_dir:
            directory = os.path.dirname(cpSignalFilename)
        reducedSignalNamesByTileDict[tiles] = os.path.join(directory, os.path.splitext(os.path.basename(cpSignalFilename))[0] + '_%s.CPsignal'%filterSet)
    return reducedSignalNamesByTileDict


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

def findCPsignalFile(cpSeqFilename, redFluors, greenFluors, cpSignalFilename):
 
    # find signal in red
    for filename in redFluors + greenFluors:
        if os.path.exists(filename):
            num_lines = int(subprocess.check_output('wc -l %s | awk \'{print $1}\''%filename, shell=True).strip())
    
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
    signal_comma_format = np.array(['\t'.join([signal[i].astype(str), ','.join(signal_green[i].astype(str))]) for i in range(num_lines)])   
    cp_seq = np.ravel(pd.read_table(cpSeqFilename, delimiter='\n', header=None))
    np.savetxt(cpSignalFilename, np.transpose(np.vstack((cp_seq, signal_comma_format))), fmt='%s', delimiter='\t')      
    return signal_green

def reduceCPsignalFile(cpSignalFilename, filterSet, reducedCPsignalFilename):
    # take only lines that contain the filterSet
    to_run = "awk '{n=split($2, a,\":\"); b=0; for (i=1; i<=n; i++) if (a[i]==\"%s\") b=1; if (b==1) print $0}' %s > %s"%(filterSet, cpSignalFilename, reducedCPsignalFilename)
    #to_run = "awk '{i=index($2, \"%s\"); if (i>0) print}' %s > %s"%(filterSet, cpSignalFilename, reducedCPsignalFilename)
    print to_run
    os.system(to_run)
    #os.system("awk '{i=index($2, \"%s\"); if (i>0 && $9!=\"\" && $9!=\"nan\") print}' %s > %s"%(filterSet, cpSignalFilename, reducedCPsignalFilename))
    return

########## FILENAME CHANGERS ##########

def getAllSortedCPsignalFilename(reducedSignalNamesByTileDict, directory):
    filename = [value for value in reducedSignalNamesByTileDict.itervalues()][0]
    newfilename = os.path.basename(filename[:filename.find('tile')] + filename[filename.find('filtered'):])
    return os.path.join(directory, os.path.splitext(newfilename)[0] + '_sorted.CPsignal')

def getSortedFilename(filename):
    return os.path.splitext(filename)[0] + '.sorted' + os.path.splitext(filename)[1]

def getCompressedBarcodeFilename(sortedAllCPsignalFile):
    return os.path.splitext(sortedAllCPsignalFile)[0] + '.unique_barcodes'

def getBarcodeMapFilename(sortedAllCPsignalFile):
    return os.path.splitext(sortedAllCPsignalFile)[0] + '.barcode_to_seq'

def getAnnotatedSignalFilename(sortedAllCPsignalFile):
    return os.path.splitext(sortedAllCPsignalFile)[0] + '.annotated.CPsignal'

def getFittedFilename(sequenceToLibraryFilename):
    return os.path.splitext(sequenceToLibraryFilename)[0] + '.CPfitted'

def getBindingSeriesFilenameParts(sequenceToLibraryFilename, numCores):
    bindingSeriesFilenameDict = {}
    for i in range(numCores):
        bindingSeriesFilenameDict[i] = os.path.splitext(sequenceToLibraryFilename)[0] + '.%d.binding_series.mat'%i
    return bindingSeriesFilenameDict

def getfitParametersFilenameParts(bindingSeriesFilenameParts):
    fitParametersFilenameParts = {}
    for i, filename in bindingSeriesFilenameParts.items():
        fitParametersFilenameParts[i] = os.path.splitext(os.path.splitext(filename)[0])[0] + '.fitParameters.mat'
    return fitParametersFilenameParts

def getFitParametersFilename(sequenceToLibraryFilename):
    return os.path.splitext(sequenceToLibraryFilename)[0] + '.fitParameters'

def getTimesFilename(annotatedSignalFilename):
    return os.path.splitext(annotatedSignalFilename)[0] + '.times'

def getPerVariantFilename(fittedBindingFilename):
    return os.path.splitext(fittedBindingFilename)[0] + '.perVariant.CPfitted'

def getPerVariantBootstrappedFilename(variantFittedFilename):
    return os.path.splitext(variantFittedFilename)[0] + '.bootStrapped.CPfitted'

########## ACTUAL FUNCTIONS ##########
def sortCPsignal(cpSeqFile, sortedAllCPsignalFile, barcode_col):
    # make sure file is sorted
    mydata = pd.read_table(cpSeqFile, index_col=False, header=None)
    mydata.sort(barcode_col-1, inplace=True)
    mydata.to_csv(sortedAllCPsignalFile, sep='\t', na_rep='nan', header=False, index=False)
    return

def sortConcatenateCPsignal(reducedSignalNamesByTileDict, barcode_col, sortedAllCPsignalFile):
    # save concatenated and sorted CPsignal file
    allCPsignals = ' '.join([value for value in reducedSignalNamesByTileDict.itervalues()])
    os.system("cat %s > %s"%(allCPsignals, sortedAllCPsignalFile))
    sortCPsignal(sortedAllCPsignalFile, sortedAllCPsignalFile, barcode_col)
    return

def compressBarcodes(sortedAllCPsignalFile, barcode_col, seq_col, compressedBarcodeFile):
    script = 'compressBarcodes'
    to_run = "python -m %s -i %s -o %s -c %d -C %d"%(script, sortedAllCPsignalFile, compressedBarcodeFile, barcode_col, seq_col)
    print to_run
    os.system(to_run)
    return

def barcodeToSequenceMap(compressedBarcodeFile, libraryDesignFile, outFile):
    script = 'findSeqDistribution'
    to_run =  "python -m %s -b %s -l %s -o %s "%(script, compressedBarcodeFile, libraryDesignFile, outFile)
    print to_run
    os.system(to_run)
    return

def matchCPsignalToLibrary(barcodeToSequenceFilename, sortedAllCPsignalFile, sequenceToLibraryFilename):
    script = 'matchCPsignalLibrary'
    to_run =  "python -m %s -b %s -i %s -o %s "%(script, barcodeToSequenceFilename, sortedAllCPsignalFile, sequenceToLibraryFilename)
    print to_run
    os.system(to_run)
    return

def splitAndFit(bindingSeries, concentrations, fitParameters, numCores, index=None):
    if index is None:
        index = bindingSeries.index
        
    # split into parts
    print 'Splitting clusters into %d groups:'%numCores
    indicesSplit = np.array_split(index, numCores)
    bindingSeriesSplit = [bindingSeries.loc[indices] for indices in indicesSplit]
    printBools = [True] + [False]*(numCores-1)
    # fit
    print 'Fitting binding curves:'
    fits = Parallel(n_jobs=20, verbose=25)(delayed(fitSetKdsPython)(subBindingSeries, concentrations, fitParameters, print_bool) for subBindingSeries, print_bool in itertools.izip(bindingSeriesSplit, printBools))
    return pd.concat(fits)

def fitSetKdsPython(subBindingSeries, concentrations, fitParameters, print_bool=None):
    if print_bool is None: print_bool = True
    singles = {}
    for i, idx in enumerate(subBindingSeries.index):
        if print_bool:
            if (i+1)%(len(subBindingSeries)/20.) == 0: print 'working on %d out of %d iterations (%d%%)'%(i+1, len(subBindingSeries.index), 100*(i+1)/float(len(subBindingSeries.index)))
        fluorescence = subBindingSeries.loc[idx]
        fitParametersNew = fitParameters.copy()
        if not np.isnan(fluorescence[0]):
            # as long as the first point is measured, further constrain the fmin
            fitParametersNew.loc['upperbound', 'fmin'] = 2*fluorescence.min()
        singles[idx] = fitBindingCurve.fitSingleBindingCurve(concentrations, fluorescence, fitParametersNew, plot=False)
    return pd.concat(singles, axis=1).transpose()       


def fitSetKds(fitParametersFilenameParts, bindingSeriesFilenameParts, initialFitParameters, scale_factor, indices=None):
    workerPool = multiprocessing.Pool(processes=len(bindingSeriesFilenameParts)) #create a multiprocessing pool that uses at most the specified number of cores
    for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
        result = workerPool.apply_async(findKds, args=(bindingSeriesFilename, fitParametersFilenameParts[i],
                                                     initialFitParameters['fmax']['lowerbound'], initialFitParameters['fmax']['upperbound'], initialFitParameters['fmax']['initial'],
                                                     initialFitParameters['dG']['lowerbound'],   initialFitParameters['dG']['upperbound'],   initialFitParameters['dG']['initial'],
                                                     initialFitParameters['fmin']['lowerbound'], initialFitParameters['fmin']['upperbound'], initialFitParameters['fmin']['initial'],
                                                     scale_factor,)
                               )
    workerPool.close()
    workerPool.join()
    return 

def fitSetKoff(fitParametersFilenameParts, bindingSeriesFilenameParts, initialFitParameters, scale_factor, fittype=None):
    if fittype is None: fittype = 'offrate'
    if fittype == 'offrate': parameter = 'toff'
    if fittype == 'onrate': parameter = 'ton'
    workerPool = multiprocessing.Pool(processes=len(bindingSeriesFilenameParts)) #create a multiprocessing pool that uses at most the specified number of cores
    for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
        result = workerPool.apply_async(findKoff, args=(bindingSeriesFilename, fitParametersFilenameParts[i],
                                                     initialFitParameters['fmax']['lowerbound'], initialFitParameters['fmax']['upperbound'], initialFitParameters['fmax']['initial'],
                                                     initialFitParameters[parameter]['lowerbound'], initialFitParameters[parameter]['upperbound'], initialFitParameters[parameter]['initial'],
                                                     initialFitParameters['fmin']['lowerbound'], initialFitParameters['fmin']['upperbound'], initialFitParameters['fmin']['initial'],
                                                     scale_factor, fittype)
                               )
    workerPool.close()
    workerPool.join()
    fitParameters = joinTogetherFitParts(fitParametersFilenameParts, parameter=parameter)
    return fitParameters

def findKds(bindingSeriesFilename, outputFilename, fmax_min, fmax_max, fmax_initial, kd_min, kd_max, kd_initial, fmin_min, fmin_max, fmin_initial, scale_factor):
    matlabFunctionCallString = "fitBindingCurve('%s', [%4.2f, %4.2f, %4.2f], [%4.2f, %4.2f, %4.2f], '%s', [%4.2f, %4.2f, %4.2f], %4.2f );"%(bindingSeriesFilename,
                                                                                                                
                                                                                                                fmax_min, kd_min, fmin_min,
                                                                                                                fmax_max, kd_max, fmin_max,
                                                                                                                outputFilename,
                                                                                                                fmax_initial, kd_initial, fmin_initial,
                                                                                                                scale_factor)
    try:
        logString = spawnMatlabJob(matlabFunctionCallString)
        return (matlabFunctionCallString, logString)
    except Exception,e:
        return(matlabFunctionCallString,'Python excpetion generated in findKds: ' + e.message + e.stack)
    
def findKoff(bindingSeriesFilename, outputFilename, fmax_min, fmax_max, fmax_initial, kd_min, kd_max, kd_initial, fmin_min, fmin_max, fmin_initial, scale_factor, fittype):
    matlabFunctionCallString = "fitOffRateCurve('%s', [%4.2f, %4.2f, %4.2f], [%4.2f, %4.2f, %4.2f], '%s', [%4.2f, %4.2f, %4.2f], %4.2f, '%s' );"%(bindingSeriesFilename,
                                                                                                                
                                                                                                                fmax_min, kd_min, fmin_min,
                                                                                                                fmax_max, kd_max, fmin_max,
                                                                                                                outputFilename,
                                                                                                                fmax_initial, kd_initial, fmin_initial,
                                                                                                                scale_factor, fittype)
    try:
        logString = spawnMatlabJob(matlabFunctionCallString)
        return (matlabFunctionCallString, logString)
    except Exception,e:
        return(matlabFunctionCallString,'Python excpetion generated in findKds: ' + e.message + e.stack)  


def removeFilenameParts(annotatedSignalFilename, numCores):
    bindingSeriesFilenameParts = getBindingSeriesFilenameParts(annotatedSignalFilename, numCores)
    for i, filename in bindingSeriesFilenameParts.items():
        os.system("rm %s"%filename)
    fitParametersFilenameParts = getfitParametersFilenameParts(bindingSeriesFilenameParts)
    for i, filename in fitParametersFilenameParts.items():
        os.system("rm %s"%filename)    
    return

def saveDataFrame(dataframe, filename, index=None, float_format=None):
    if index is None:
        index = False
    if float_format is None:
        float_format='%4.3f'
    dataframe.to_csv(filename, sep='\t', na_rep="NaN", index=index, float_format=float_format)
    return

def joinTogetherFitParts(fitParametersFilenameParts=None, parameter=None, indices=None):
    if parameter is None: parameter='dG'
    # define columns coming from matlab file
    cols = ['params', 'exit_flag', 'rsq', 'rmse', 'qvalue', 'params_var']
    names = ['fmax', parameter, 'fmin', 'exit_flag', 'rsq', 'rmse', 'qvalue', 'fmax_var', parameter+'_var', 'fmin_var']
    
    # if no files are given, just return column names
    if fitParametersFilenameParts is None:
        return names

    all_fit_parameters = []
    for i, fitParametersFilename in fitParametersFilenameParts.items():
        fit_parameters = sio.loadmat(fitParametersFilename)
        if indices is None:
            index=None
        else:
            index = indices[i]
        # save as data frame
        all_fit_parameters.append(pd.DataFrame(index=index, columns=names, data= np.hstack([fit_parameters[col] for col in cols])) )
    
    return pd.concat(all_fit_parameters)

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

def loadLibraryCharacterization(filename, version=None):
    if version is None:
        version = 'v2'
    
    if version == 'v1':  # i.e. from september
        cols = ['sequence', 'topology', 'loop', 'receptor', 'helix_context',
                'junction_sequence', 'helix_sequence', 'helix_one_length',
                'helix_two_length', 'junction_length', 'total_length']
        mydata = pd.read_table(filename, header=0, names=cols, index_col=False)
    
    elif version == 'v2':
        mydata = pd.read_table(filename)
    return mydata

def loadCompressedBarcodeFile(filename):
    cols = ['sequence', 'barcode', 'clusters_per_barcode', 'fraction_consensus']
    mydata = pd.read_table(filename)
    return mydata

def findSequenceRepresentation(consensus_sequences, compare_to, exact_match=None):
    # initialize
    if exact_match is None:
        exact_match = False # default is to search for whether it contains the sequence .
                            # set to True if the sequences must match exactly
    num_bc_per_variant = np.zeros(len(compare_to), dtype=int) # number consensus sequences per designed sequence
    is_designed = np.zeros(len(consensus_sequences), dtype=int)-1 # -1 if no designed sequence is found that matches. else index
    
    # cycle through designed sequences. Find if they are in the actual sequences
    for i, sequence in enumerate(compare_to):
        if i%1000==0:
            print "checking %dth sequence"%i
        
        # whether the sequence (designed) is in the actual sequence, given by the fastq
        in_fastq = True
    
        # start count
        count = -1
        
        # first location in the sorted list that the sequence might match
        indx = np.searchsorted(consensus_sequences, sequence)
        
        # starting from the first index given by searching the sorted list,
        # cycle until the seuqnece is no longer found
        while in_fastq:
            count += 1
            if indx+count < len(consensus_sequences):
                if exact_match:
                    in_fastq = consensus_sequences[indx+count] == sequence
                else:
                    in_fastq = consensus_sequences[indx+count].find(sequence)==0
            else:
                in_fastq = False
            # if the designed sequence is in the most probable location of the
            # sorted consensus sequences, give 'is_designed' at that location to the
            # indx of the matching sequence in 'compare_to'
            if in_fastq:
                is_designed[indx+count] = i
        num_bc_per_variant[i] = count
    return num_bc_per_variant, is_designed

def loadCPseqSignal(filename, concentrations=None):
    """
    function to load CPseqsignal file with the appropriate headers
    """
    cols = ['tileID','filter','read1_seq','read1_quality','read2_seq','read2_quality','index1_seq','index1_quality','index2_seq', 'index2_quality','all_cluster_signal','binding_series']
    table = pd.read_csv(filename, sep='\t', header=None, names=cols, index_col=False)
    return table

def loadNullScores(signalNamesByTileDict, filterSet, tile=None, index=None):
    # find one CPsignal file before reduction. Find subset of rows that don't
    # contain filterSet
    if tile is None: tile = '003'   # Default is third tile
    filename = signalNamesByTileDict[tile]

    # Now load the file, specifically the signal specifed
    # by index.
    table = loadCPseqSignal(filename)
    table.dropna(subset=['filter'], axis=0, inplace=True)
    subset = [str(s).find(filterSet) == -1 for s in table.loc[:, 'filter']]
    
    if index is None: index = -1
    null_scores = np.array([float(s.split(',')[index]) for s in table.loc[subset].binding_series])
    
    return null_scores[np.isfinite(null_scores)]

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

def loadBindingCurveFromCPsignal(filename, concentrations):
    """
    open file after being reduced to the clusters you are interested in.
    find the all cluster signal and the binding series (comma separated),
    then return the binding series, normalized by all cluster image.
    """
    formatted_concentrations = formatConcentrations(concentrations)
    table = pd.read_table(filename, usecols=(0, 10,11), index_col=0)
    binding_series = pd.DataFrame([s.split(',') for s in table.binding_series], dtype=float, index=table.index, columns=formatted_concentrations)
    all_cluster_signal = pd.Series(table.all_cluster_signal, index=table.index, dtype=float)
    return binding_series, all_cluster_signal

def boundFluorescence(signal, plot=None):
    if plot is None: plot=False
    
    lowerbound = np.percentile(signal.dropna(), 1)
    upperbound = signal.median() + 5*signal.std()
    
    if plot:
        binwidth = (upperbound - lowerbound)/50.
        plt.figure()
        sns.distplot(signal.dropna(), bins = np.arange(signal.min(), signal.max()+binwidth, binwidth), color='seagreen')
        ax = plt.gca()
        ylim = ax.get_ylim()
        plt.plot([lowerbound]*2, ylim, 'k:')
        plt.plot([upperbound]*2, ylim, 'k:')
        plt.xlim(0, upperbound + 2*signal.std())
    signal.loc[signal < lowerbound] = lowerbound
    signal.loc[signal > upperbound] = upperbound
    
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

def loadFittedCPsignal(filename):
    
    table = pd.read_table(filename, index_col=0)
    for param in table:
        try:
            table.loc[:, param] = table.param.astype(float)
        except:
            pass
    return table

def bindingCurve(concentrations, kd, fmax=None, fmin=None):
    if fmax is None:
        fmax = 1
    if fmin is None:
        fmin = 0
    return fmax*concentrations/(concentrations + kd) + fmin


def plotVariant(sub_table, concentrations):
    # reduce to fits that were successful
    sub_table = sub_table[sub_table['fit_success']==1]
    sub_table = sub_table[sub_table['rsq']>0.5]
    print 'testing %d variants'%len(sub_table)
    
    # plot  
    num_concentrations = len(concentrations)
    binding_curves = np.array([np.array(sub_table[i]) for i in range(num_concentrations)])
    frac_bound = np.median(binding_curves, axis=1)
    [percentile25, percentile75] = np.percentile(binding_curves, [25,75],axis=1)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.errorbar(concentrations, frac_bound, yerr=[frac_bound-percentile25, percentile75-frac_bound], fmt='o')
    ax.plot(np.logspace(-1, 4, 50), bindingCurve(np.logspace(-1, 4, 50), np.median(sub_table['kd']), np.median(sub_table['fmax']), np.median(sub_table['fmin'])),'k')
    ax.set_xscale('log')
    ax.set_xlabel('concentration (nM)')
    ax.set_ylabel('normalized fluorescence')
    #plt.title('%d variants tested'%len(sub_table))
    plt.tight_layout()
    return

def findBarcodeFilter(table):
    cols = ['barcode', 'clusters_per_barcode', 'fraction_consensus']
    table_test = pd.DataFrame(index = table.index, columns = cols + ['max_confidence', 'outside_max', 'barcode_length', 'barcode_good'] )
    table_test.loc[:, cols] = table.loc[:, cols]
    
    index = table_test.loc[:, 'clusters_per_barcode'].dropna().index
    table_test.loc[:, 'max_confidence'] = pd.Series(st.binom.interval(0.95, table_test.loc[index, 'clusters_per_barcode'].astype(int).values, 0.5)[1],
                  index = index)
    table_test.loc[:, 'outside_max'] = False
    table_test.loc[:, 'outside_max'] = table_test.loc[index, 'clusters_per_barcode']*table_test.loc[index, 'fraction_consensus']/100 > table_test.loc[index, 'max_confidence']
    
    index = table_test.loc[:, 'barcode'].dropna().index
    table_test.loc[:, 'barcode_length'] = 0
    table_test.loc[index, 'barcode_length'] = [len(seq) for seq in table_test.loc[index, 'barcode'].astype(str).values]
    
    table.loc[:, 'barcode_good'] = table_test.loc[:, 'outside_max']&(table_test.loc[:, 'barcode_length'] > 12)

    return table

def formatConcentrations(concentrations):
    return [('%4.1f'%x).lstrip().rstrip('0').rstrip('.') for x in concentrations]

def findFitParameters(table):
    fitFilteredTable = IMlibs.filterFitParameters(IMlibs.filterStandardParameters(table))

def findPerVariantFits(table, concentrations, fitParameters):
    concentrationCols = IMlibs.formatConcentrations(concentrations)
    fitFilteredTable = IMlibs.filterFitParameters(IMlibs.filterStandardParameters(table))
    fitFilteredTable.loc[:, 'fmax'] = fitFilteredTable.fmax.astype(float)
    grouped = fitFilteredTable.groupby('variant_number')
    
    fmaxes = pd.concat([grouped['fmax'].count(), grouped['fmax'].median()], axis=1, keys=['number', 'fmax'])
    fmax_std = fmaxes.groupby('number').std()
    weights = np.sqrt(fmaxes.groupby('number').count())
    index = fmax_std.dropna().index
    x = np.array(fmax_std.loc[index].index).astype(float)
    y = fmax_std.loc[index, 'fmax'].astype(float).values
    s = np.ravel(1/np.sqrt(weights.loc[index]))
 
    # plot data
    plt.figure(figsize=(4,3));
    plt.plot(x, y,  'ko');
    
    # fit exponential function
    popt, pcov = curve_fit(IMlibs.fitFunction2, x, y, p0=[1, 1, 0], sigma=s)
    plt.plot(x, IMlibs.fitFunction2(x, *popt), 'c', label='%4.2f*exp(-%4.2f*n)+%4.2f'%(popt[0],popt[1], popt[2]))
    
    # fit 1/sqrt(n)
    popt, pcov = curve_fit(IMlibs.fitFunction, x, y, p0=[1, 0], sigma=s)
    plt.plot(x, IMlibs.fitFunction(x, *popt), 'r', label='%4.2f/sqrt(n)+%4.2f'%(popt[0], popt[1]));

    plt.xlabel('number of tests')
    plt.ylabel('std of fit fmaxes in bin')
    plt.legend()
    plt.tight_layout()
    
    fitParameters.loc['lowerbound', 'fmax'] = fmaxes.fmax.min()
    fitParameters.loc['initial',    'fmax'] = fmaxes.fmax.mean()
    fitParameters.loc['upperbound', 'fmax'] = fmaxes.fmax.max()
    
    fmins = grouped['fmin'].median()
    
    return

def fitFunction(xdata, a, b):
    return float(a)/np.sqrt(xdata)+b

def fitFunction2(xdata, a, b, c):
    return  a * np.exp(-b * xdata) + c

def findVariantTable(table, parameter=None, name=None, concentrations=None):
    # define defaults
    if parameter is None: parameter = 'dG'
    if name is None:
        name = 'ss_correct'   # default for lib 2
    
    # define columns as all the ones between variant number and fraction consensus
    columns = [name for name in table.loc[:,'variant_number':name]]
    columns.remove('variant_number')
    test_stats = [parameter, 'fmin', 'fmax', 'qvalue']
    
    grouped = table.groupby('variant_number')
    variant_table = pd.concat([grouped[columns].first(),
                               pd.DataFrame(index=grouped[columns].first().index,
                                            columns=test_stats+[parameter+'_%s'%s for s in ['lb', 'ub']])],
                              axis=1)
    
    # filter for nan, barcode, and fit
    firstFilterGrouped = filterStandardParameters(table).groupby('variant_number')
    variant_table.loc[:, 'numTests'] = firstFilterGrouped[columns[0]].count()
    variant_table.loc[:, 'numRejects'] = grouped[columns[0]].count() - variant_table.loc[:, 'numTests'] 
    
    fitFilteredTable = filterFitParameters(filterStandardParameters(table))
    fitFilterGrouped = fitFilteredTable.groupby('variant_number')
    variant_table.loc[:, 'fitFraction'] = fitFilterGrouped[columns[0]].count()/variant_table.loc[:, 'numTests']
    
    # then save parameters
    variant_table.loc[:, test_stats] = fitFilterGrouped[test_stats].median()
    
    # null model is that all the fits are bad. bad fits happen ~15% of the time
    #p = 1-variant_table.loc[(variant_table.numTests>=5)&(variant_table.dG < -10), 'fitFraction'].mean()
    p = 0.25
    variant_table.loc[:, 'pvalue'] = np.nan
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n]
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, p)
    
    # get normalized binding series
    concentrationCols = formatConcentrations(concentrations)
    bindingSeries = fitFilteredTable.loc[:, concentrationCols]
    bindingSeries.loc[:, 'variant_number'] = fitFilteredTable.variant_number
    bindingSeriesNorm = np.divide(fitFilteredTable.loc[:, concentrationCols], np.vstack(fitFilteredTable.all_cluster_signal))
    bindingSeriesNorm.loc[:, 'variant_number'] = fitFilteredTable.variant_number
    allClusterSignal = fitFilteredTable.loc[:, ['all_cluster_signal']]
    allClusterSignal.loc[:, 'variant_number'] = fitFilteredTable.variant_number
    variant_table = pd.concat([variant_table, bindingSeriesNorm.groupby('variant_number').median()], axis=1)
    return variant_table, bindingSeries.groupby('variant_number').median(), allClusterSignal.groupby('variant_number').median(),  bindingSeriesNorm.groupby('variant_number').median()

def getPvalueFitFilter(variant_table, p):
    variant_table.loc[:, 'pvalue'] = np.nan
    for n in np.unique(variant_table.loc[:, 'numTests'].dropna()):
        # do one tailed t test
        x = (variant_table.loc[:, 'fitFraction']*variant_table.loc[:, 'numTests']).loc[variant_table.numTests==n]
        variant_table.loc[x.index, 'pvalue'] = st.binom.sf(x-1, n, p)
    return      

def getBootstrappedErrors(variant_table, table, numCores, parameter=None, variants=None):
    if parameter is None: parameter = 'dG'
    if variants is None:
        variants = variant_table.loc[(variant_table.numTests >=3)&(variant_table.pvalue <= 0.05)].index
        
    subtable = filterFitParameters(filterStandardParameters(table)).loc[:, ['variant_number', parameter]]
    grouped = subtable.groupby('variant_number')
    
    print '\tDividing table into groups...'
    actual_variants = []
    groups = []
    for name, group in grouped:
        if name in variants:
            actual_variants.append(name)
            groups.append(group)
    
    print '\tMultiprocessing bootstrapping...'
    bounds = Parallel(n_jobs=20, verbose=5)(delayed(perVariantError)(group.dG) for group in groups)
    
    variant_table.loc[actual_variants, [parameter+'_%s'%s for s in ['lb', 'ub']]] = bounds
    return variant_table

def filterStandardParameters(table):

    not_nan_binding_points = formatConcentrations([0.91, 2.73])# if first two binding points are NaN, filter
    nan_filter = ~np.isnan(table.loc[:, not_nan_binding_points]).all(axis=1)        
    barcode_filter = table.loc[:, 'barcode_good']
    return table.loc[nan_filter&barcode_filter]

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
        binedges = np.arange(-12, -6, 0.5)
    subtable = pd.DataFrame(index=variant_table.index,
                            columns=['binned_dGs', 'confInt', 'numTests'],
                            data=np.column_stack([np.digitize(variant_table.dG, binedges),
                                                  variant_table.dG_ub - variant_table.dG_lb,
                                                  np.around(variant_table.numTests*variant_table.fitFraction)]))
    subtable.loc[:, 'dG (kcal/mol)'] = ['%3.1f:%3.1f'%(binedges[bins-1], binedges[bins]-1) if bins!=len(binedges) else '>%3.1f'%binedges[-1] for bins in subtable.binned_dGs.astype(int)]
    order = ['%3.1f:%3.1f'%(binedges[bins-1], binedges[bins]-1) if bins!=len(binedges) else '>%3.1f'%binedges[-1] for bins in np.arange(1, len(binedges)+1)]
    
    
    if count_binedges is None:
        count_binedges = [0, 5, 10, 20, 40, 80]
    count_order = ['%d-%d'%(count_binedges[bins-1], count_binedges[bins]-1) if bins!=len(count_binedges) else '>%d'%count_binedges[-1] for bins in np.arange(1, len(count_binedges)+1)]
    subtable.loc[:, 'binned_counts'] = np.digitize(subtable.numTests, count_binedges) 
    subtable.loc[:, '# tests'] = ['%d-%d'%(count_binedges[bins-1], count_binedges[bins]-1) if bins!=len(count_binedges) else '>%d'%count_binedges[-1] for bins in subtable.binned_counts.astype(int)]
    g = sns.FacetGrid(subtable, size=2.5, aspect=1.5, col="# tests", col_wrap=3,
                      col_order=count_order )
    g.map(sns.boxplot, "dG (kcal/mol)", "confInt",
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
    


