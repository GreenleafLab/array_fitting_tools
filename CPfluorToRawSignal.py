'''
Created on Jul 21, 2014

@author: Lauren Chircus
'''


''' Import Modules'''

import os, sys
import numpy as np
from optparse import OptionParser


''' Functions '''

# Writes a single line with a newline to the specified file 
# assumes that file is already open
def writeLine(myfile, line):
    myfile.write(line+'\n')

# Prints a line to the console and writes it to the specified file
def printLine(myfile, line):
    print line
    writeLine(myfile, line)


def geOutputFilenameFromCPfluorFilename(CPseqFilename,directory):
    (path,filename) = os.path.split(CPseqFilename) #split the file into parts
    (basename,ext) = os.path.splitext(filename)
    return os.path.join(directory,basename + '.signal')


def calculateSignal(amplitude,sigma):
    return 2*np.pi*amplitude*sigma*sigma


def calculateSignalInFile(myfile, outFile):
    
    outFile = open(outFile, 'w')
    
    with open(myfile,"r") as quantificationFile:
    
        for line in quantificationFile:
            
            CPfluorOutput = line.rstrip().split(':')
            CPfluorOutput = CPfluorOutput[::-1][0:5]
            #print CPfluorOutput[4]
            
            if CPfluorOutput[4] == '1':
                signal = calculateSignal(float(CPfluorOutput[3]), float(CPfluorOutput[2]))
            else:
                signal = 'NA'
            
            outFile.write(str(signal)+'\n')
            
    # Close files
    quantificationFile.close()
    outFile.close()


def main():
    
    ## Get options and arguments from command line
    parser = OptionParser()
    parser.add_option('-q', dest="quantificationFile", help="CPfluor file with quantification data")
    parser.add_option('-o', dest="outFile", help="file in which to write output")
    options, arguments = parser.parse_args()
        
    # return usage information if no argvs given
    if len(sys.argv)==1:
        os.system(sys.argv[0]+" --help")
        sys.exit()
    
    # Making sure all mandatory options appeared.
    mandatories = ['quantificationFile', 'outFile']
    for m in mandatories:
        if not options.__dict__[m]:
            print "mandatory option is missing\n"
            parser.print_help()
            exit(-1)
    
    calculateSignalInFile(options.quantificationFile, options.outFile)


if __name__ == '__main__':
    main()