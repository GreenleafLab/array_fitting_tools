"""
Date: 9/9/14
Author: Sarah Denny

This module is intended to analyze the data gotten in an experiment on 8/19/14.
The data had four filter sets
"""

# import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from optparse import OptionParser
import filefun
import plotfun

# Get command-line options
opts = OptionParser()
usage = "usage: %prog [options] [inputs]"
opts = OptionParser(usage=usage)
opts.add_option("-g", "--binding", help="accepts directory containing all CPfluor files of BINDING images")
opts.add_option("-r", "--background", help="accepts directory containing all CPfluor files of BACKGROUND images (i.e. all RNA images)")
opts.add_option("-f", "--filter", help="gives the name of the filter, i.e. 'rigid:'")
opts.add_option("-s", "--seq", help="gives the filename of the CPseq file")
options, arguments = opts.parse_args()

# functions
def getFluorescence(wholeDir):
    cpfluorFiles = subprocess.check_output('find %s -name "*.CPfluor" | sort'%wholeDir, shell=True).split()
    
    numImages = len(cpfluorFiles)
    numLines = int(subprocess.check_output("wc -l %s | awk '{print $1}'"%cpfluorFiles[0], shell=True))
    fluorArray = np.zeros((numLines, numImages))
    fitSuccess = np.zeros((numLines, numImages), dtype=bool)
    
    for i, cpfluorFile in enumerate(cpfluorFiles):
        cpfluor = filefun.fluorFile(cpfluorFile)
        fluorArray[:, i] = cpfluor.volume
        fitSuccess[:, i] = cpfluor.success
    
    return fluorArray, fitSuccess

# load the filters from the cpseq file
cpseqFile = options.seq
cpseq = filefun.seqFile(cpseqFile)

# load the images to analyze
greenArray, greenSuccess = getFluorescence(options.binding)
redArray, redSuccess     = getFluorescence(options.background)


# what is the fluorescence of each?
myfilter = options.filter

myfilters = ['rigid:', 'rigidLoop:', 'wc:', 'notRigid:']
labels = ['0nM', '4nM', '20nM', '100nM', '500nM', '2uM']
numConcentrations = len(labels)

for myfilter in myfilters:
    criteria = np.all((np.all(greenSuccess, 1),
                       np.all(redSuccess, 1),
                       cpseq.getIndx(myfilter)),
                       axis = 0)
    
    # plot all cluster histograms
    plotfun.plotClustersNew(redArray, greenArray, criteria, labels)
    plt.savefig('%s.all_images_histogram.pdf'%myfilter.strip(':'))
    
# find fmax
criteria = np.zeros(redArray.shape, dtype=bool)
filterIndx = np.any((cpseq.getIndx('rigid:'),
                    cpseq.getIndx('wc:'),
                    cpseq.getIndx('notRigid:')),
                    axis=0)
criteria[:, -1] = np.all((np.all(greenSuccess, 1),
                    np.all(redSuccess, 1),
                    filterIndx),
                    axis=0)
fmax = plotfun.findFmax(redArray, greenArray, criteria)

# plot binding curves
for myfilter in myfilters:
    criteria = np.all((np.all(greenSuccess, 1),
                       np.all(redSuccess, 1),
                       cpseq.getIndx(myfilter)),
                       axis = 0)
    
    # plot binding curve
    concentrations = np.array([0, 4, 20, 100, 500, 2000])
    
    yvalues = np.array([])
    xvalues = np.array([])
    for i in range(numConcentrations):
        yvalues = np.hstack((yvalues, np.divide(greenArray[criteria, i], redArray[criteria, i])))
        xvalues = np.hstack((xvalues, [concentrations[i]]*np.sum(criteria)))
    
    #fmax = np.median(yvalues[xvalues==concentrations[-1]])
    plotfun.plotBindingCurve(xvalues, yvalues/fmax, concentrations)
    plt.savefig('%s.binding_curve.fmax_isconstant.pdf'%myfilter.strip(':'))
    
# offrates are slightly different
mydata = filefun.loadKdFits(filename)
bindingSeries = np.divide(filefun.findBindingSeriesAsArray(mydata['bindingSeries']), np.vstack(mydata['allClusterSignal']))
for myfilter in myfilters:
    criteria = np.all((np.logical_not(np.any(np.isnan(bindingSeries), axis=1)),
                        cpseq.getIndx(myfilter)),
                        axis = 0)
    plotfun.plotOffrates(bindingSeries, xvalues, criteria, mydata['amplitude'], mydata['lifetime'])
    