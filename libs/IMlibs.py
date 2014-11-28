"""
useful functions for analyzig CP fluor files
"""

import os
import sys
import time
import re
import uuid
import subprocess
import numpy as np
import pandas as pd
import CPlibs

def spawnMatlabJob(matlabFunctionCallString):
    try:
        #construct the command-line matlab call 
        functionCallString =                      "try,"
        functionCallString = functionCallString +      "addpath('{0}', '{1}');".format('/home/sarah/array_image_tools_SKD/libs', '/home/sarah/array_image_tools_SKD/scripts') #placeholder TEMP DEBUG CHANGE
        functionCallString = functionCallString +     matlabFunctionCallString + ';'
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

def loadMapFile(mapFilename):
    """
    load map file. header prefixed by '>' gives the root directory
    next line is the all cluster image.
    The remainder are filenames to compare with associated concentrations
    """
    f = open(mapFilename)
    # first line is the root directory
    root_dir = f.readline().strip()
    
    # second line is the filename to store second column
    xValueFilename = f.readline().strip()
    
    # third line is the all cluster image directory
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
    return xValueFilename, red_dir, np.array(directories), concentrations

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
                filename = subprocess.check_output('find %s -name "*CPfluor" | grep tile%s'%(directory, tile), shell=True).strip()
            except: filename = ''
            filenameDict[tile][i] = filename
    return filenameDict

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

def pasteTogetherSignal(cpSeqFilename, signal, outfilename, delimiter=None):
    if delimiter is None:
        cmd_string = 'paste'
    else:
        cmd_string = 'paste -d %s'%delimiter
    tmpfilename = 'signal_'+str(time.time())
    np.savetxt(tmpfilename, signal)
    os.system("%s %s %s > %s"%(cmd_string, cpSeqFilename, tmpfilename, outfilename+'.tmp'))
    os.system("mv %s %s"%(outfilename+'.tmp', outfilename))
    os.system("rm %s"%tmpfilename)
    return

def findCPsignalFile(cpSeqFilename, redFluors, greenFluors, cpSignalFilename):
 
    # find signal in red
    signal = getSignalFromCPFluor(redFluors[0]) #just take first red fluor image
    pasteTogetherSignal(cpSeqFilename, signal, cpSignalFilename)
    
    # cycle through fit green images and get the final signal
    for i, currCPfluor in enumerate(greenFluors):
        if currCPfluor == '':
            num_lines = int(subprocess.check_output("wc -l %s | awk '{print $1}'"%cpSeqFilename, shell=True).strip())
            signal = np.ones(num_lines)*np.nan
        else:
            signal = getSignalFromCPFluor(currCPfluor)
            
        # if its the first in the list, append with tab delimit, else with comma
        if i==0: pasteTogetherSignal(cpSignalFilename, signal, cpSignalFilename)
        else: pasteTogetherSignal(cpSignalFilename, signal, cpSignalFilename, delimiter = ',')
           
    return

def reduceCPsignalFile(cpSignalFilename, filterSet, reducedCPsignalFilename):
    os.system("awk '{i=index($2, \"%s\"); if (i>0 && $9!=\"\" && $9!=\"nan\") print}' %s > %s"%(filterSet, cpSignalFilename, reducedCPsignalFilename))
    return

def getAllSortedCPsignalFilename(reducedSignalNamesByTileDict, directory):
    filename = [value for value in reducedSignalNamesByTileDict.itervalues()][0]
    newfilename = os.path.basename(filename[:filename.find('tile')] + filename[filename.find('filtered'):])
    return os.path.join(directory, os.path.splitext(newfilename)[0] + '_sorted.CPsignal')
    
def sortConcatenateCPsignal(reducedSignalNamesByTileDict, barcode_col, outFilename):
    # save concatenated and sorted CPsignal file
    allCPsignals = ' '.join([value for value in reducedSignalNamesByTileDict.itervalues()])
    os.system("cat %s | sort -dk%d > %s"%(allCPsignals, barcode_col, outFilename))
    return

def findKds(bindingSeriesFilename, xvalueFilename, outputFilename, fmax_min, fmax_max, fmax_initial, kd_min, kd_max, kd_initial):
    matlabFunctionCallString = "fitBindingCurve('%s', '%s', [%4.2f, %4.2f], [%4.2f, %4.2f], '%s', [%4.2f, %4.2f] );"%(bindingSeriesFilename,
                                                                                                                xvalueFilename,
                                                                                                                fmax_min, kd_min,
                                                                                                                fmax_max, kd_max,
                                                                                                                outputFilename,
                                                                                                                fmax_initial, kd_initial )
    try:
        logString = spawnMatlabJob(matlabFunctionCallString)
        return (matlabFunctionCallString)
    except Exception,e:
        return(matlabFunctionCallString,'Python excpetion generated in findKds: ' + e.message + e.stack)   


def loadCPseqSignal(filename):
    """
    function to load CPseqsignal file with the appropriate headers
    """
    cols = ['tileID','filter','read1_seq','read1_quality','read2_seq','read2_quality','index1_seq','index1_quality','index2_seq', 'index2_quality','allClusterSignal','bindingSeries']
    mydata = pd.read_csv(filename, sep='\t', header=None, names=cols, index_col=False)

    return mydata

def loadBindingCurveFromCPsignal(filename):
    """
    open file after being reduced to the clusters you are interested in.
    find the all cluster signal and the binding series (comma separated),
    then return the binding series, normalized by all cluster image.
    """
    f = open(filename)
    alllines = f.readlines()
    num_clusters = len(alllines)
    num_concentrations = len(alllines[0].split('\t')[-1].split(','))
    binding_series = np.zeros((num_clusters, num_concentrations))
    all_cluster_signal = np.zeros(num_clusters)
    for i, line in enumerate(alllines):
        line_split = line.split('\t')
        all_cluster_signal[i] = (line_split[-2])
        binding_series[i] = line_split[-1].split(',')
    f.close()
    binding_series_norm = binding_series/np.vstack(all_cluster_signal)
    return binding_series_norm, all_cluster_signal

