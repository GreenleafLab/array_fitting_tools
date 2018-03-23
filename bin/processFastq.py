#!/usr/bin/env python

# written by Curtis Layton Nov 2013 to merge multiple fastq files that came FROM THE SAME RUN (e.g. a paired-end run)
# this is intended to consolidate reads that pertain to the same cluster into one file, i.e. read1, read2, index read1, index read2 with the same cluster id.
# Files are written into a new format,".CPseq", which is a tab-delimited flat file of the format:
#
# <cluster ID> <filter IDs> <read 1> <phred quality read 1> <read 2> <phred quality read 2> <index read 1> <phred quality index read 1> <index read 2> <phred quality index read 2>
#
# where the format of <cluster ID> is:
# <machine id>:<run index>:<flowcell id>:<lane #>:<tile #>:<x coord>:<y coord>
#
# The program colates sequences by cluster id and does not require that they be in the same order
# All clusters are outputted, including those that are present in some files but missing in others

import os
import sys
import time
import argparse
import itertools
import numpy as np
import gzip
#from skbio.alignment import global_pairwise_align_nucleotide
#from skbio import DNA
from Bio import SeqIO
import nwalign as nw
from fittinglibs import fileio

nw_version = 'skbio'
filetype = 'fastq'
numLinesPerRecord = 4 #there are 4 lines for every record in a fastq file
rnap_init_seq = 'TTTATGCTATAATTATTTCATGTAGTAAGGAGGTTGTATGGAAGACGTTCCTGGATCC'
pvalue_cutoff = 1E-4
os.system('wget("ftp://ftp.ncbi.nih.gov/blast/matrices/NUC.4.4")')
gap_penalty = 8
gap_extension = 0
scoring_matrix = "NUC.4.4"

def phredStr(phredArray):
    if all(x == 0 for x in phredArray):
        return ''
    else:
        phredQuality = [chr(q+33) for q in phredArray]
        return ''.join(phredQuality)

def getNumLinesInFile(filename):
    n_lines = 0
    with open(filename) as f:
        for line in f: n_lines += 1
    return n_lines

class ClusterData:
    def __init__(self):
        self.filterID = ''
        self.read1 =  ''
        self.qRead1 = []
        self.read2 = ''
        self.qRead2 = []
        self.index1 = ''
        self.qIndex1 = []
        self.index2 = ''
        self.qIndex2 = []

def getFilehandle(filename, mode=None):
    """open normally if ext is not gz"""
    if os.path.splitext(filename)[-1] == '.gz':
        f = gzip.open(filename, mode)
    else:
        f = open(filename, mode)
    return f

def tempFilename(fullPath): #appends a prefix to the filename to indicate the the file is incomplete
    (path,filename) = os.path.split(fullPath) #split the file into parts
    return os.path.join(path,'__' + filename)

def getScorePvalue(nwScore, m, n, k=0.0097, l=0.5735, nwScoreScale=0.2773):
    """Ge pvalue of extreme value distribution which model's likelihood of achieveng a more extreme
    alignment score for two sequences of length m and n. K and l are empirically determined.
    nwScoreScale is a factor applied to scores of NUC4.4 matrices (from MATLAB nwalign)"""
    u = np.log(k*m*n)/l
    score = nwScore*nwScoreScale
    return (1 - np.exp(-np.exp(-l*(score-u))))

def getNumLinesInFile(filename):
    n_lines = 0
    with gzip.open(filename, 'rb') as f:
        for line in f: n_lines += 1
    return n_lines

### MAIN ###
if __name__=='__main__':

    #set up command line argument parser
    parser = argparse.ArgumentParser(description="Consolidates individual fastq files from a paired-end run (optionally with index reads, as well) into one file (.CPseq format)")

    parser.add_argument('-r1','--read1', help='read 1 fastq filename', )
    parser.add_argument('-r2','--read2', help='read 2 fastq filename', )
    parser.add_argument('-o','--output', help='output filename (.CPseq.gz)')

    #parse command line arguments
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        parser.print_help()


    #check output filename
    if args.output is not None:
        outputFilename = args.output
    else:
        outputFilename = fileio.stripExtension(args.read1).replace('_R1', '') + '.CPseq.gz' 
    print 'Merged sequences will be saved to : ' + outputFilename


    #read in sequence files and consolidate records
    read1Handle = getFilehandle(args.read1)
    read2Handle = getFilehandle(args.read2)

    print '    Colating read1 file "' + args.read1 + '" with:'
    print '        read2 file "' + args.read2 + '"'
    print '        line-by-line into file "' + outputFilename + '"...'

    print 'processing ' + str() + ' sequences...'
    i = 0

    # iterate through all files in lockstep
    # any files that were not found are just given the read1 filename as a "dummy" file which is then just iterated through redundantly but not read from more than once
    n_lines = getNumLinesInFile(args.read1)
    
    with gzip.open(tempFilename(outputFilename), 'ab') as outputFileHandle:
        for i, (currRead1Record,currRead2Record) in enumerate(itertools.izip(SeqIO.parse(read1Handle, filetype),SeqIO.parse(read2Handle, filetype))):

            # print status
            if i%100 == 0:
                sys.stdout.write('\r%d out of %d lines'%(i, n_lines))
                sys.stdout.flush()

            if currRead1Record.id == currRead2Record.id:
                cl = ClusterData() #create a new cluster
                currID = currRead1Record.id
                cl.read1 = currRead1Record.seq
                cl.qRead1 = currRead1Record.letter_annotations['phred_quality']
                cl.read2 = currRead2Record.seq
                cl.qRead2 = currRead2Record.letter_annotations['phred_quality']
                
                # do an nw alignment of the RNAP init sequence to the read1 sequence
                # if they align, then this sequence will be transcribed
                seq = str(currRead1Record.seq)
                seq1, seq2 = nw.global_align(seq, rnap_init_seq, gap_open=-gap_penalty, gap_extend=-gap_extension, matrix=scoring_matrix)
                score = nw.score_alignment(seq1, seq2, gap_open=-gap_penalty,  gap_extend=-gap_extension, matrix=scoring_matrix)
                pvalue = getScorePvalue(score, len(seq), len(rnap_init_seq))
                
                # add tag to filter column
                if pvalue < pvalue_cutoff:
                    cl.filterID = 'anyRNA'
    
                # add UMI to a new column assuming the UMI is the sequencee preceding the RNAP stall site
                if pvalue < pvalue_cutoff:
                    start_rnap = 0
                    while seq2[start_rnap]=='-':
                        start_rnap +=1
                    cl.index1 = seq1[:start_rnap].replace('-', '')
                    cl.qIndex1 = currRead1Record.letter_annotations['phred_quality'][:len(cl.index1)]
    
                #write to file
                outputFileHandle.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'.format(currID, cl.filterID, cl.read1, phredStr(cl.qRead1), cl.read2, phredStr(cl.qRead2), cl.index1,phredStr(cl.qIndex1), cl.index2,phredStr(cl.qIndex2)))
            else:
                sys.stderr('ERROR Cluster names not aligned!')
                sys.exit()
    #rename ouput file to final filename indicating it is complete
    os.rename(tempFilename(outputFilename), outputFilename)
    read1Handle.close()
    read2Handle.close()


