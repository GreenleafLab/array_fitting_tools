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

def fitKdAndMaxSignal(dataFrame, xvalues):
    '''
    function used to map binding curve fit over all rows in table
    input:  data frame with column 'bindingSeries' that contains fit values for 11 concentrations of
    output: fit parameters as vector [maxSignalFit, dGFit]
    '''
    # only fit if filter is present
    
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
    pNames = ['fmax', 'delta_g']
    
    # fit a binding curve if you have data for at least 5 points in the binding series
    if sum(useIndex)>4 and dataFrame['filter']==fitParameters.filterSet:
        #fmax, dG, RSS, degreesOfFreedom, MSE = fitBindingCurve(concentration[useIndex], signal[useIndex], np.nanmax(signal), 0.2)

        #p0 = {'p0':np.nanmax(signal), 'dG':np.log(.2e-9)}
        p0 = [np.nanmax(signal), np.log(20)]
        
        fitModel = NLS.NLS(bindingCurveResiduals, p0, xvalues[useIndex], signal[useIndex], pNames)

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

def makePlots(mydata):
    indx = np.all((mydata['filter']==fitParameters.filterSet,
                    np.isfinite(mydata['allClusterSignal'])),
                    axis=0)
    binding_signal = np.array([np.array(mydata['bindingSeries'].loc[i].split(',')).astype(float) for i in range(len(mydata)) ])
    histogram.compare(np.transpose((binding_signal/np.vstack(mydata['allClusterSignal']))[indx]))
    histogram.compare([mydata['allClusterSignal'][indx],
                       last_point[indx]], labels=['signal in all RNA', 'signal in green in last point'])
    
    return

def fitSubFile(myfile, outFolder, xvalues):
        
    # load subFile
    mydata = IMlibs.loadCPseqSignal(myfile)
    
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

if __name__ == '__main__':
    
    # load fit parameters
    fitParameters = globalvars.Parameters()
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

    # if you're not splitting the file into pieces, open the files and fit kd's
    fitSubFile(options.inFile, options.outFolder, xvalues)
        # import data

