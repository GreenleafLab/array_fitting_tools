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
import scipy.io as sio
import matplotlib.pyplot as plt
#import CPlibs

def spawnMatlabJob(matlabFunctionCallString):
    try:
        #construct the command-line matlab call 
        functionCallString =                      "try,"
        functionCallString = functionCallString +      "addpath('{0}', '{1}');".format('/home/namita/Code/array_image_tools_SKD/libs', '/home/namita/Code/array_image_tools_SKD/scripts') #placeholder TEMP DEBUG CHANGE
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

def pasteTogetherSignal(cpSeqFilename, signal, outfilename):
    output_directory = os.path.dirname(outfilename)
    tmpfilename = os.path.join(output_directory, 'signal_'+str(time.time()))
    np.savetxt(tmpfilename, signal, fmt='%s')
    os.system("paste %s %s > %s"%(cpSeqFilename, tmpfilename, outfilename+'.tmp'))
    os.system("mv %s %s"%(outfilename+'.tmp', outfilename))
    os.system("rm %s"%tmpfilename)
    return

def findCPsignalFile(cpSeqFilename, redFluors, greenFluors, cpSignalFilename):
 
    # find signal in red
    signal = getSignalFromCPFluor(redFluors[0]) #just take first red fluor image

    # find signal in green
    num_lines = len(signal)
    signal_green = np.zeros((num_lines, len(greenFluors)))
    
    # cycle through fit green images and get the final signal
    for i, currCPfluor in enumerate(greenFluors):
        if currCPfluor == '':
            signal_green[:,i] = np.ones(num_lines)*np.nan
        else:
            signal_green[:,i] = getSignalFromCPFluor(currCPfluor)
            
    # combine signal in both
    signal_comma_format = np.array(['\t'.join([signal[i].astype(str), ','.join(signal_green[i].astype(str))]) for i in range(num_lines)])   
    pasteTogetherSignal(cpSeqFilename, signal_comma_format, cpSignalFilename)       
    return

def reduceCPsignalFile(cpSignalFilename, filterSet, reducedCPsignalFilename):
    os.system("awk '{i=index($2, \"%s\"); if (i>0 && $9!=\"\" && $9!=\"nan\") print}' %s > %s"%(filterSet, cpSignalFilename, reducedCPsignalFilename))
    return

def getAllSortedCPsignalFilename(reducedSignalNamesByTileDict, directory):
    filename = [value for value in reducedSignalNamesByTileDict.itervalues()][0]
    newfilename = os.path.basename(filename[:filename.find('tile')] + filename[filename.find('filtered'):])
    return os.path.join(directory, os.path.splitext(newfilename)[0] + '_sorted.CPsignal')

def getCompressedBarcodeFilename(sortedAllCPsignalFile):
    return os.path.splitext(sortedAllCPsignalFile)[0] + '.unique_barcodes'

def getBarcodeMapFilename(compressedBarcodeFile):
    return os.path.splitext(compressedBarcodeFile)[0] + '.barcode_to_seq'

def getAnnotatedSignalFilename(barcodeToSequenceFilename):
    return os.path.splitext(barcodeToSequenceFilename)[0] + '.annotated.CPsignal'

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
 
def sortConcatenateCPsignal(reducedSignalNamesByTileDict, barcode_col, sortedAllCPsignalFile):
    # save concatenated and sorted CPsignal file
    allCPsignals = ' '.join([value for value in reducedSignalNamesByTileDict.itervalues()])
    os.system("cat %s > %s"%(allCPsignals, sortedAllCPsignalFile))

    # make sure file is sorted
    mydata = pd.read_table(sortedAllCPsignalFile, index_col=False, header=None)
    mydata.sort(barcode_col-1, inplace=True)
    mydata.to_csv(sortedAllCPsignalFile, sep='\t', na_rep='nan', header=False, index=False)
    return

def compressBarcodes(sortedAllCPsignalFile, barcode_col, seq_col, compressedBarcodeFile):
    script = '/home/namita/Code/array_image_tools_SKD/scripts/compressBarcodes.py'
    print "python %s -i %s -o %s -c %d -C %d"%(script, sortedAllCPsignalFile, compressedBarcodeFile, barcode_col, seq_col)
    os.system("python %s -i %s -o %s -c %d -C %d"%(script, sortedAllCPsignalFile, compressedBarcodeFile, barcode_col, seq_col))
    return

def barcodeToSequenceMap(compressedBarcodeFile, libraryDesignFile, outFile):
    script = '/home/namita/Code/array_image_tools_SKD/scripts/findSeqDistribution.py'
    print "python %s -b %s -l %s -o %s "%(script, compressedBarcodeFile, libraryDesignFile, outFile)
    os.system("python %s -b %s -l %s -o %s "%(script, compressedBarcodeFile, libraryDesignFile, outFile))
    return

def matchCPsignalToLibrary(barcodeToSequenceFilename, sortedAllCPsignalFile, sequenceToLibraryFilename):
    script = '/home/namita/Code/array_image_tools_SKD/scripts/matchCPsignalLibrary.py'
    print "python %s -b %s -i %s -o %s "%(script, barcodeToSequenceFilename, sortedAllCPsignalFile, sequenceToLibraryFilename)
    os.system("python %s -b %s -i %s -o %s "%(script, barcodeToSequenceFilename, sortedAllCPsignalFile, sequenceToLibraryFilename))
    return

def findKds(bindingSeriesFilename, outputFilename, fmax_min, fmax_max, fmax_initial, kd_min, kd_max, kd_initial, fmin_min, fmin_max, fmin_initial):
    matlabFunctionCallString = "fitBindingCurve('%s', [%4.2f, %4.2f, %4.2f], [%4.2f, %4.2f, %4.2f], '%s', [%4.2f, %4.2f, %4.2f] );"%(bindingSeriesFilename,
                                                                                                                
                                                                                                                fmax_min, kd_min, fmin_min,
                                                                                                                fmax_max, kd_max, fmin_max,
                                                                                                                outputFilename,
                                                                                                                fmax_initial, kd_initial, fmin_initial )
    try:
        logString = spawnMatlabJob(matlabFunctionCallString)
        return (matlabFunctionCallString, logString)
    except Exception,e:
        return(matlabFunctionCallString,'Python excpetion generated in findKds: ' + e.message + e.stack)   

def removeFilenameParts(filenameParts):
    for i, filename in filenameParts.items():
        os.system("rm %s"%filename)
    return

def saveDataFrame(dataframe, filename, index=None, float_format=None):
    if index is None:
        index = False
    if float_format is None:
        float_format='%4.3f'
    dataframe.to_csv(filename, sep='\t', na_rep="NaN", index=index, float_format=float_format)
    return

def joinTogetherFitParts(fitParametersFilenameParts,  ):
    all_fit_parameters = np.empty((0,10))
    for i, fitParametersFilename in fitParametersFilenameParts.items():
        fit_parameters = sio.loadmat(fitParametersFilename)
        all_fit_parameters = np.vstack((all_fit_parameters, np.hstack((fit_parameters['params'],
                                                   fit_parameters['exit_flag'],
                                                   fit_parameters['rsq'],
                                                   fit_parameters['rmse'],
                                                   fit_parameters['qvalue'],
                                                   fit_parameters['params_var']))
                                        ))
    all_fit_parameters = pd.DataFrame(data=all_fit_parameters, columns=['fmax', 'dG', 'fmin', 'exit_flag', 'rsq', 'rmse', 'qvalue', 'fmax_var', 'dG_var', 'fmin_var'])
    return all_fit_parameters

def makeFittedCPsignalFile(fitParametersFilename,annotatedSignalFilename, fittedBindingFilename):
    # paste together annotated signal and fitParameters
    os.system("paste %s %s > %s"%(annotatedSignalFilename, fitParametersFilename, fittedBindingFilename))
    return

def loadLibraryCharacterization(filename):
    cols = ['sequence', 'topology', 'loop', 'receptor', 'helix_context',
            'junction_sequence', 'helix_sequence', 'helix_one_length',
            'helix_two_length', 'junction_length', 'total_length']
    mydata = pd.read_table(filename, header=0, names=cols, index_col=False)
    return mydata

def loadCompressedBarcodeFile(filename):
    cols = ['sequence', 'barcode', 'clusters_per_barcode', 'fraction_consensus']
    mydata = pd.read_table(filename, usecols=(0, 1, 2, 3), header=None, names = cols)
    return mydata

def findSequenceRepresentation(consensus_sequences, compare_to, exact_match=None):
    if exact_match is None:
        exact_match = False # default is to search for whether it contains the sequence .
                            # set to True if the sequences must match exactly
    num_bc_per_variant = np.zeros(len(compare_to), dtype=int) # number barcodes per variant
    is_designed = np.zeros(len(consensus_sequences), dtype=int)-1 # -1 if no designed sequence is found that matches. else index
    # cycle through designed sequences. Find if they are in the
    # actual sequences
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

def loadCPseqSignal(filename):
    """
    function to load CPseqsignal file with the appropriate headers
    """
    cols = ['tileID','filter','read1_seq','read1_quality','read2_seq','read2_quality','index1_seq','index1_quality','index2_seq', 'index2_quality','all_cluster_signal','binding_series']
    table = pd.read_csv(filename, sep='\t', header=None, names=cols, index_col=False)
    #we ended here!!
    #count = 0
    #    
    #binding_series = np.zeros((len(table),8))
    #for series in table['binding_series']:
    #    if (len(np.array(series.split(','), dtype=float)) ==8):
    #        binding_series[count] = np.array(series.split(','), dtype=float)
    #        count += 1
    #import pdb; pdb.set_trace()
    binding_series = np.array([np.array(series.split(','), dtype=float) for series in table['binding_series']])
    
    for col in range(binding_series.shape[1]):
        print('newversion')
        table[col] = binding_series[:, col]
  
    return table

def loadNullScores(signalNamesByTileDict, filterSet):
    tile = '003'
    filename = signalNamesByTileDict[tile]
    tmp_filename = signalNamesByTileDict[tile] + 'tmp'
    os.system("grep -v %s %s > %s"%(filterSet, filename, tmp_filename))
    table = loadCPseqSignal(tmp_filename)
    null_scores = table[7][np.isfinite(table[7])]
    os.system("rm %s"%(tmp_filename))
    return np.array(null_scores)
    

def loadBindingCurveFromCPsignal(filename):
    """
    open file after being reduced to the clusters you are interested in.
    find the all cluster signal and the binding series (comma separated),
    then return the binding series, normalized by all cluster image.
    """
    table = pd.read_table(filename, usecols=(10,11))
    count = 0
    binding_series = np.zeros((len(table),8))
    for series in table['binding_series']:
        if (len(np.array(series.split(','), dtype=float)) ==8):
            binding_series[count] = np.array(series.split(','), dtype=float)
            count += 1
     
    #binding_series = np.array([np.array(series.split(','), dtype=float) for series in table['binding_series']])
    
    return binding_series, np.array(table['all_cluster_signal'])

def loadFittedCPsignal(filename):

    table =  pd.read_table(filename, usecols=tuple(range(21,45)))
    binding_series, all_cluster_image = loadBindingCurveFromCPsignal(filename)
    for col in range(binding_series.shape[1]):
        table[col] = binding_series[:, col]
    table['all_cluster_signal'] = all_cluster_image
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

def reduceByVariant(table):

    variants = np.arange(0, np.max(table['variant_number']))
    columns = [name for name in table][:12] + ['numTests', 'numRejects', 'kd', 'dG', 'fmax', 'fmin']
    newtable = pd.DataFrame(columns=columns, index=np.arange(len(variants)))
    for i, variant in enumerate(variants):
        if i%1000==0: print 'computing iteration %d'%i
        sub_table = table[table['variant_number']==variant]
        if len(sub_table) > 0:
            sub_table_filtered = filterFitParameters(sub_table)[['kd', 'dG', 'fmax', 'fmin']]
            newtable.iloc[i]['numTests'] = len(sub_table_filtered)
            newtable.iloc[i]['numRejects'] = len(sub_table) - len(sub_table_filtered)
            newtable.iloc[i]['kd':] = np.median(sub_table_filtered, 0)
            newtable.iloc[i][:'total_length'] = sub_table.iloc[0][:'total_length']
    return newtable
    