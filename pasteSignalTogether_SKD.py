                             #!/usr/bin/env python

# Mer
# ---------------------------------------------
#
# This script merges sequence files with quantified fluoresence data line-by-line
# across all files in a directory
#
# Inputs:
#   CPseq files split by tile in a directory
#   CPfluor files split by tile in a directory
#
# Outputs:
#   Combined sequence and fluoresence files (.LCseqfluor files; one file per tile)
#
# Adapted from Curtis Layton (curtis.layton@stanford.edu) & Peter McMahon (pmcmahon@stanford.edu)
# December 2013, January 2014
# Adapted by Lauren Chircus (lchircus@stanford.edu)
# July 2014

import sys
import os
import subprocess
import argparse
import multiprocessing
import datetime
import numpy as np

# make sure path is on
sys.path.append('/Users/greenleaflab/array_image_tools/')
import filefun 
#import CPlibs



# Writes a single line with a newline to the specified file
# assumes that file is already open
def writeLine(myfile, line):
    myfile.write(line+'\n')

# Prints a line to the console and writes it to the specified file
def printLine(myfile, line):
    print line
    writeLine(myfile, line)

def collectLogs(inLog): #multiprocessing callback function to collect the output of the worker processes
    logFilename = inLog[0]
    logText = inLog[1]
    resultList[logFilename] = logText

def zipFilesTogether(filenames, CPseqFile):
    separator = ","
    output_list = []

    for f in filenames:
        if f == 'NA':
            #add NA's to each line
            if len(output_list) == 0:
                numberOfLines = int(subprocess.check_output(['wc', '-l', CPseqFile]).split(' ')[0])
                output_list = [ "NA" for x in range(0, numberOfLines)]
            else:
                for i in range(len(output_list)):
                    output_list[i] = output_list[i].rstrip() + separator + 'NA\n'
        else:
            #add fluor values
            with open(f,'r') as inputfile:
                if len(output_list) == 0:
                    output_list = inputfile.readlines()
                else:
                    input_list = inputfile.readlines()
                    for i in range(len(output_list)):
                        output_list[i] = output_list[i].rstrip() + separator + input_list[i]
    return output_list

def makeSignalFileName(directory, filename):
    return os.path.join(directory, '%s.signal'%os.path.splitext(os.path.basename(filename))[0])
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description="Pastes together signal files")

#parser.add_argument('-n','--num_cores', help='maximum number of cores to use',required=True)
parser.add_argument('-s', '--CPseq', help="CPseq file")
parser.add_argument('-f', '--CPfluor_dir', help="Folder containing CPfluor files")
parser.add_argument('-b', '--CPfluor_red', help="CPfluor file of expected signal in red")
parser.add_argument('-o', '--output', help="Where to put pasted files")

if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()
cpfluorDir = args.CPfluor_dir
cpfluorRed = args.CPfluor_red
cpseqFile = args.CPseq
signalDir = os.path.join(cpfluorDir, 'quantifiedSignal')
if not os.path.exists(signalDir): os.mkdir(signalDir)
# make a outputput file if none is provided
if not args.output:
    outputFile = os.path.join(signalDir, cpfluorDir)
else:
    outputFile = args.output

################ Make log File ################

## Make log file
logFileName = outputFile + ".pasteFilesTogether.log"
logFile = open(logFileName,'w')

## Write to log
writeLine(logFile, 'Analysis performed: ' + str(datetime.datetime.now()))
writeLine(logFile, 'Pasted together quantified tiles from directory: ' + cpfluorDir)

################ Find signal files ################
print "Generating signal files in %s"%cpfluorDir
print "Saving to %s"%signalDir

# find CPfluor files
cpfluorFiles = subprocess.check_output('find %s -maxdepth 1 -name "*.CPfluor" | sort '%(cpfluorDir), shell=True).split()
cpfluorFiles.append(cpfluorRed)
pyScript = '~/LMCarray_tools/CPfluorToRawSignal.py'
for cpfluorFile in cpfluorFiles:
    os.system("python %s -q %s -o %s"%(pyScript, cpfluorFile, makeSignalFileName(signalDir, cpfluorFile)))

# Find column headings
redSignalFile = makeSignalFileName(signalDir, cpfluorRed)
greenSignalFiles = subprocess.check_output('find %s -maxdepth 1 -name "*.signal" | sort | grep -v %s'%(signalDir, redSignalFile), shell=True).split()
columns = filefun.cpseqFileCols()
columns.append(redSignalFile)
for sigfile in greenSignalFiles: columns.append(sigfile)

# make file names to save to
columnHeaderFile = outputFile+'.columnHeaders'
timeStampFile = outputFile+'.timeStamps'
cpseqSignalFile = outputFile+'.CPseqsignal'
cpseqSignalTempfile = outputFile+'tmp.allsignal'

# save CPseqsignal file
os.system("paste -d , %s > %s"%(' '.join(greenSignalFiles), cpseqSignalTempfile))
os.system("paste %s %s %s > %s"%(cpseqFile, redSignalFile, cpseqSignalTempfile, cpseqSignalFile))

# save all column headers
np.savetxt(columnHeaderFile, columns, fmt='%s')

# save time stamps
timeStamps = np.array([filefun.absoluteTime(filefun.getTimeStamp(filename)) for filename in greenSignalFiles])
np.savetxt(timeStampFile, timeStamps - timeStamps[0], fmt='%s')

logFile.close()

