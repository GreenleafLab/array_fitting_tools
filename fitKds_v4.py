#!/usr/bin/env python

'''
Written by Lauren Chircus
Stanford Univeristy
July 23, 2014

'''

''' Import Modules '''
import os, sys
import numpy as np
from scipy import optimize
import pandas as pd
#import matplotlib.pyplot as plt
from optparse import OptionParser
import datetime
import subprocess
import multiprocessing
import NLS
import globalvars
import filefun

''' Functions '''

def bindingCurve(params, concentration):
    '''
    formula for a binding curve.
    input: a vector [float(maxSignal), float(dG)], float(concentation)
    output: calculated signal at given concentration
    '''
    return params[0]*concentration/(np.exp(params[1])*(10**9)+concentration)

def offRate(params, time):
    '''
    formula for an offrate.
    input: a vector [float(amplitude), float(lifetime)], float(time)
    output: calculated signal at given time point
    '''
    return params[0]*np.exp(-time/params[1])

def offRateResiduals(params, x, y):
    '''
    formula for an offrate.
    input: a vector [float(amplitude), float(lifetime)], float(time)
    output: calculated signal at given time point
    '''
    return y - offRate(params, x)

def bindingCurveResiduals(params, concentration, signal):
    '''
    calculates residuals for binding curve
    input: a vector [float(maxSignal), float(dG)], float(concentation), float(signal)
    output: given signal - calculated signal at given concentration
    '''
    return signal - bindingCurve(params, concentration)


def fitBindingCurve(concentration, signal, fmax0, dG0):
    '''
    # fits binding curve equation
    input:  np.array(concentation), np.array(signal), initial guess float(maxSignal), initial guess float(dG)
    output: fit parameters as vector [maxSignalFit, dGFit]
    '''

    fit, cov, infodict, mesg, ier = optimize.leastsq(bindingCurveResiduals, [fmax0, dG0], args=(concentration, signal), full_output=True)
    fmax = fit[0]
    dG = fit[1]
    RSS = np.sum(infodict['fvec']**2)
    degreesOfFreedom = len(signal)-len(fit)
    MSE = RSS/degreesOfFreedom

    return fmax, dG, RSS, degreesOfFreedom, MSE


def fitKdAndMaxSignal(dataFrame, xvalues):
    '''
    function used to map binding curve fit over all rows in table
    input:  data frame with column 'bindingSeries' that contains fit values for 11 concentrations of
    output: fit parameters as vector [maxSignalFit, dGFit]
    '''

    # concentration series used in dCas9 experiment
    #xvalues = [ 0, 90./(3**9), 90./(3**8), 90./(3**7), 90./(3**6), 90./(3**5), 90./(3**4), 90./(3**3), 90./(3**2), 90./(3**1), 90.]
    #concentration = np.array(concentration)
    

                   
    # make signal series useful
    signal = np.array(dataFrame['bindingSeries'].replace('NA','NaN').split(',')).astype(float)
    #signal = np.array(map(float,signal))
    # normalize to all cluster image
    signal = signal/float(dataFrame['allClusterSignal'])

    # figure out which indicies have real numbers and which aren't quantified
    useIndex = np.all((np.ones(len(xvalues), dtype=bool) - np.isnan(signal),
                       np.isfinite(signal)),
                       axis=0)

    # reset the first index to false so it's not included in fit
    #useIndex[0] = False
    pNames = ['amplitude', 'lifetime']
    
    # fit a binding curve if you have data for at least 5 points in the binding series
    if sum(useIndex)>4:
        #fmax, dG, RSS, degreesOfFreedom, MSE = fitBindingCurve(concentration[useIndex], signal[useIndex], np.nanmax(signal), 0.2)

        #p0 = {'p0':np.nanmax(signal), 'dG':np.log(.2e-9)}
        p0 = [np.nanmax(signal), 1]
        
        fitModel = NLS.NLS(offRateResiduals, p0, xvalues[useIndex], signal[useIndex], pNames)

        p1 = fitModel.parmEsts
        if np.any((not fitModel.jacobian_exists, np.any(fitModel.parmEsts > fitParameters.maxAnyFitParameter)), axis=0):
            p1_upper = np.ones(2)*np.Inf
            p1_lower = -np.ones(2)*np.Inf
        else:
            p1_upper = fitModel.parmEsts + 1.96*fitModel.parmSE
            p1_lower = fitModel.parmEsts - 1.96*fitModel.parmSE

        RSS = fitModel.RSS
        MSE = fitModel.MSE
        degreesOfFreedom = fitModel.df

    # otherwise, return a vector saying you didn't fit it
    else:
        p1 = np.array(['NaN', 'NaN']).astype(float)
        p1_upper = p1_lower =  p1
        dG = float('NaN')
        RSS = float('NaN')
        MSE = float('NaN')
        degreesOfFreedom = float('NaN')
    return pd.Series({pNames[0]: p1[0],
                      pNames[1]: p1[1],
                      pNames[0]+'_upper':p1_upper[0], pNames[0]+'_lower':p1_lower[0],
                      pNames[1]+'_upper':p1_upper[1], pNames[1]+'_lower':p1_lower[1],
                      'RSS': RSS, 'MSE': MSE, "df": degreesOfFreedom})
    #return pd.Series({'fmax': fmax, 'dG': dG, 'RSS': RSS, 'MSE': MSE, "df": degreesOfFreedom, "fmax_pval": fmax_pval, "dG_pval": dG_pval, "fmax_se": fmax_se, "dG_se": dG_se})



def nameOutputFileFromInputFile(inFile, outFolder):
    """
    name file in a reasonable way, with the extension '.kdFit'
    """
    outFile_name = os.path.splitext(os.path.basename(inFile))[0]+'.kdFit'
    return os.path.join(outFolder, outFile_name)



def fitSubFile(myfile, outFolder, xvalues):
        
    # load subFile
    mydata = filefun.loadCPseqSignal(myfile)
    
    # fit subFile
    data2 = mydata.apply(fitKdAndMaxSignal, axis=1, args=(xvalues,))
    
    # quality filter fits
    firstDataPoint = np.array([row.replace('NA', 'nan').split(',')[0] for row in mydata['bindingSeries']]).astype(float)
    allCluster = np.array(mydata['allClusterSignal'])
    data2['amplitude_upper'][data2['amplitude_upper'] > fitParameters.maxAmplitude(np.divide(firstDataPoint, allCluster))] = np.Inf
    data2['amplitude_lower'][data2['amplitude_lower'] > fitParameters.maxAmplitude(np.divide(firstDataPoint, allCluster))] = np.Inf
    
    print "setting lifetimes of more than %d minutes to infinity"%fitParameters.maxLifetime
    data2['lifetime_upper'][data2['lifetime_upper'] > fitParameters.maxLifetime] = np.Inf
    data2['lifetime_lower'][data2['lifetime_lower'] < fitParameters.minLifetime] = 0
    
    # join everything together
    mydata = mydata.join(data2)
    outFile_name = nameOutputFileFromInputFile(myfile, outFolder)
    mydata.to_csv(outFile_name, index=False, sep='\t', na_rep='NA', header=True)

    #subprocess.call(['python ~/Workspace/array_project/src/fitKds_v4.py -i '+file+' -o '+outFolder], shell=True)
    return

def main():

    ## Get options and arguments from command line
    parser = OptionParser()

    parser.add_option('-i', dest="inFile", help="Data file to be analyzed")
    parser.add_option('-x', dest="xvalueFile", help="File")
    parser.add_option("-s", dest='splitFile', help="number of lines to split input file into for faster processing", action='store', type='int', default=0)
    parser.add_option('-o', dest="outFolder", help="directory in which to write output")
    parser.add_option('-n','--num_cores', help='maximum number of cores to use', action='store', type='int', default=1)

    options, arguments = parser.parse_args()

    # return usage information if no argvs given
    if len(sys.argv)==1:
        os.system(sys.argv[0]+" --help")
        sys.exit()

    # Making sure all mandatory options appeared.
    mandatories = ['inFile']
    for m in mandatories:
        if not options.__dict__[m]:
            print "mandatory option is missing\n"
            parser.print_help()
            exit(-1)

    # make output directory if one does not exist:
    if not options.outFolder:
        options.outFolder = os.path.dirname(options.inFile)
        
    # load xvalues from file
    xvalues = np.loadtxt(options.xvalueFile)
    
    # split the input file into smaller chunks for faster processing
    if options.splitFile > 0:

        print "Splitting the file into smaller files for parallel processing..."

        # make temporary directories
        tempInputFolder = 'tempInputBindingData'
        tempOutputFolder =  'tempOutputBindingData'
        '''
        subprocess.call(['mkdir '+ tempInputFolder], shell=True)
        subprocess.call(['mkdir '+ tempOutputFolder], shell=True)

        # split files and put them in tempInputFolder
        subprocess.call(['split -a 4 -l '+str(options.splitFile)+' '+options.inFile+' '+tempInputFolder+'/temp'],  shell=True)
        '''

        # get list of files in tempInputFolder
        fileList = subprocess.check_output(["ls -v "+tempInputFolder],shell=True).rstrip().split("\n")

        print fileList

        # assign the number of cores we will use for multicore processing based on the command-line parameter that was passed in and the number of files
        numCores = int(options.num_cores)
        numFiles = len(fileList)
        if (numFiles < numCores): #no need to use more cores than we actually have files to process
            numCores = numFiles
        resultList = {}
        workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores

        print "Fitting kd's to clusters in subfiles..."

        # call this function for each file in tempInputFolder and wait for all the jobs to finish
        for myfile in fileList:
            filePath = tempInputFolder+'/'+myfile
            print "  ...processing subfile " + myfile
            workerPool.apply_async(fitSubFile, args=(filePath, tempOutputFolder, xvalues)) #send out the jobs
        workerPool.close()
        workerPool.join()

        # check to make sure that output folder and input folder have the same number of files
        outputFileList = subprocess.check_output(["ls -v "+tempOutputFolder],shell=True).rstrip().split("\n")
        if len(fileList) == len(outputFileList):
            print "Merging temporary output files into single file..."

            # concatenate all output files into single file
            outFile_name = nameOutputFileFromInputFile(options.inFile, options.outFolder)
            subprocess.call(['cat '+tempOutputFolder+'/* > '+outFile_name], shell=True)

            print "Deleting temporary files and folders..."
            # delete temp folders and files
            subprocess.call(['rm -r '+ tempInputFolder], shell=True)
            subprocess.call(['rm -r '+ tempOutputFolder], shell=True)
        else:
            print "The number of output files does not match the number of input files."
            sys.exit()

        print "Kd fits completed at " + str(datetime.datetime.now())

    # if you're not splitting the file into pieces, open the files and fit kd's
    else:
        fitSubFile(options.inFile, options.outFolder, xvalues)
        # import data




if __name__ == '__main__':
    # load fit parameters
    fitParameters = globalvars.fitParameters()
    main()
