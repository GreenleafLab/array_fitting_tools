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
import subprocess
import numpy as np
from copy import copy
import gzip
#from skbio.alignment import global_pairwise_align_nucleotide
#from skbio import DNA
from Bio import SeqIO
import nwalign as nw
from fittinglibs import fileio

##### define parameters for alignment #####
numRecordsPerLine = 4 # for fastq files
rnapInitSeq = 'TTTATGCTATAATTATTTCATGTAGTAAGGAGGTTGTATGGAAGACGTTCCTGGATCC' #rnapInitSeq
pValueCutoff = 1E-6
gapPenalty = 8
gapExtension = 0
scoringMatrix = "NUC.4.4"
if not os.path.exists('NUC.4.4'):
    subprocess.check_call('wget "ftp://ftp.ncbi.nih.gov/blast/matrices/NUC.4.4"', shell=True)

def phredStr(phredArray):
    if all(x == 0 for x in phredArray):
        return ''
    else:
        phredQuality = [chr(q+33) for q in phredArray]
        return ''.join(phredQuality)

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

def getTempFilename(fullPath): #appends a prefix to the filename to indicate the the file is incomplete
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
    if filename.endswith('gz'):

        with gzip.open(filename, 'rb') as f:
            for line in f: n_lines += 1
    else:
        with open(filename,'r') as f:
            for line in f: n_lines +=1 
    return n_lines

def getAlignmentPvalue(seq):
    seq1, seq2 = nw.global_align(seq, rnapInitSeq, gap_open=-gapPenalty, gap_extend=-gapExtension, matrix=scoringMatrix)
    score = nw.score_alignment(seq1, seq2, gap_open=-gapPenalty,  gap_extend=-gapExtension, matrix=scoringMatrix)
    pvalue = getScorePvalue(score, len(seq), len(rnapInitSeq))
    return pvalue

def reverse_complement(seq):
    '''Return reverse complement (used here for filtering R2 to reference library)'''
    rev_dct = {'A':'T','T':'A','C':'G','G':'C'}

    return ''.join([rev_dct[x] for x in list(reversed(seq))])

### MAIN ###
if __name__=='__main__':

    #set up command line argument parser
    parser = argparse.ArgumentParser(description="Consolidates individual fastq files from a paired-end run (optionally with index reads, as well) into one file (.CPseq format)")

    parser.add_argument('-r1','--read1', help='read 1 fastq filename', )
    parser.add_argument('-r2','--read2', help='read 2 fastq filename', )
    parser.add_argument('-rd','--read_dir', help='path to folder containing read 1 and read2 fastq filenames', )
    parser.add_argument('-o','--output', help='output filename (.CPseq.gz)')
    parser.add_argument('--filterref', help='text file of reference library sequences to filter R2 on')
    parser.add_argument('--NoPrimerAlign', default=False, action="store_true", help='flag if you not wish to perform Needleman-wunsch alignment to look for primer site')
    parser.add_argument('--nosort', action='store_true', help='flag if you do not wish to sort alphabetically by barcode.')
    parser.add_argument('--primer', help='Primer to filter for. If not provided, RNAP initiation site / stall sequence will be used.')
    parser.add_argument('--barcode_start', type=int, help='Location where barcode starts in read 1. If provided, will override default barcode extraction (taking sequence before Primer)')
    parser.add_argument('--barcode_length', type=int, action="store", help='length of barcode following primer sequence.')

    #parse command line arguments
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        parser.print_help()

    if args.barcode_start is not None:
        if args.barcode_length is None:
            raise RuntimeError('If using --barcode_start, must also provide --barcode_length.')

    if args.barcode_length is not None:
        if args.barcode_start is None:
            raise RuntimeError('If using --barcode_length, must also provide --barcode_start.')

    #read in sequence files and consolidate records
    if args.read1 is not None and args.read2 is not None:
        read1Filenames = [args.read1]
        read2Filenames = [args.read2]
        outputFilename = fileio.stripExtension(args.read1).replace('_R1', '') + '.CPseq.gz'
    else:
        read1Filenames = subprocess.check_output(
            ('find %s -mindepth 1 -maxdepth 1 -type f -name "*R1*fastq.gz"')%
            args.read_dir, shell=True).strip().split()
        read2Filenames = [filename.replace('_R1_', '_R2_') for filename in read1Filenames]
        outputFilename = os.path.join(args.read_dir, 'allfastqs.CPseq.gz')

    #check output filename
    if args.output is not None:
        outputFilename = args.output

    if args.primer is not None:
        primerSeq = args.primer
    else:
        primerSeq = copy(rnapInitSeq)

    print('Filtering for %s' % primerSeq)

    # make list of RefSeqs:
    rev_dct = {'A':'T','T':'A','C':'G','G':'C','U':'A'}

    if args.filterref is not None:
        RefSeqList = []
        with open(args.filterref,'r') as f:
            for line in f.readlines():

                refseq = line.decode('utf-8').strip()
                rc=[]
                for char in refseq[::-1]:
                    if char in list('ACTGU'):
                        rc.append(rev_dct[char])
                        
                RefSeqList.append(''.join(rc))

    outputFilenames = {}
    for j, (read1Filename, read2Filename) in enumerate(zip(read1Filenames, read2Filenames)):
        # open input files
        read1Handle = getFilehandle(read1Filename, 'r') 
        read2Handle = getFilehandle(read2Filename, 'r')
        
        # define output files
        if read1Filenames[0].endswith('gz'):
            print('h')
            currOutputFilename = (fileio.stripExtension(outputFilename) + '_%03d.CPseq.gz'%j)
        else:
            currOutputFilename = (fileio.stripExtension(outputFilename) + '_%03d.CPseq'%j)
        currTempFilename = getTempFilename(currOutputFilename)
    
        # print log
        print ''
        print 'Processing %d file out of %d:'%(j+1, len(read1Filenames))
        print '    Read1 file: "' + read1Filename + '"'
        print '    Read2 file: "' + read2Filename + '"'
        print '    Output file: "' + currOutputFilename + '"...'
               
        # iterate through all entries in lockstep
        numRecords = getNumLinesInFile(read1Filename)/numRecordsPerLine
        
        with gzip.open(currTempFilename, 'ab') as outputFileHandle:
            for i, (currRead1Record,currRead2Record) in enumerate(itertools.izip(
                SeqIO.parse(read1Handle, 'fastq'),
                SeqIO.parse(read2Handle, 'fastq'))):
    
                # print status
                if (i+1)%100 == 0:
                    sys.stdout.write('\r%d out of %d records (%4.1f%%)'%(i+1, numRecords, 100*(i+1.)/numRecords))
                    sys.stdout.flush()
    
                if currRead1Record.id == currRead2Record.id:
                    cl = ClusterData() #create a new cluster
                    currID = currRead1Record.id
                    cl.read1 = currRead1Record.seq
                    cl.qRead1 = currRead1Record.letter_annotations['phred_quality']
                    cl.read2 = currRead2Record.seq
                    cl.qRead2 = currRead2Record.letter_annotations['phred_quality']
                    
                    if not args.NoPrimerAlign:
                        
                        # do an nw alignment of the RNAP init sequence to the read1 sequence
                        # if they align, then this sequence will be transcribed
                        seq = str(currRead1Record.seq)
                        seq1, seq2 = nw.global_align(seq, primerSeq, gap_open=-gapPenalty, gap_extend=-gapExtension, matrix=scoringMatrix)
                        score = nw.score_alignment(seq1, seq2, gap_open=-gapPenalty,  gap_extend=-gapExtension, matrix=scoringMatrix)
                        pvalue = getScorePvalue(score, len(seq), len(primerSeq))

                        print(seq, primerSeq, seq1, seq2, pvalue)

                        # add tag to filter column
                        if pvalue < pValueCutoff:
                            cl.filterID = 'anyRNA'

                            #get start of primer
                            start_rnap = 0
                            while seq2[start_rnap]=='-':
                                start_rnap +=1

                    else:
                        # use find routine to find the primer (no allowance for mutations in primer sequence.)

                        start_rnap = str(currRead1Record.seq).find(primerSeq)
                        if start_rnap > -1:
                            cl.filterID = 'anyRNA'

                            if args.barcode_start is not None:

                                barcode_start = start_rnap + len(primerSeq)+ args.barcode_start
                
                                # add UMI to a new column based on defined start/end locations.

                                cl.index1 = currRead1Record.seq[barcode_start:(barcode_start + args.barcode_length)]
                                cl.qIndex1 = currRead1Record.letter_annotations['phred_quality'][barcode_start:(barcode_start + args.barcode_length)]

                            else:
                                # write UMI, assuming that field prior to primerSeq is the UMI.
                                cl.index1 = seq1[:start_rnap].replace('-', '')
                                cl.qIndex1 = currRead1Record.letter_annotations['phred_quality'][:len(cl.index1)]

                    # compare reverse complement of Read2 to library and filter for if it includes a ref seq or not
                    if args.filterref is not None:
                        ContainsARefSeq = False
                        seq = str(currRead2Record.seq)

                        for ref_seq in RefSeqList:
                            if ref_seq in seq:
                                ContainsARefSeq = True
                                break

                        if ContainsARefSeq:
                            cl.filterID += '_IncludesRefSeq'
                            
                    #write to file
                    outputFileHandle.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'.format(currID, cl.filterID, cl.read1, phredStr(cl.qRead1), cl.read2, phredStr(cl.qRead2), cl.index1,phredStr(cl.qIndex1), cl.index2,phredStr(cl.qIndex2)))
                else:
                    sys.stderr('ERROR Cluster names not aligned!')
                    sys.exit()
        #rename ouput file to final filename indicating it is complete
        os.rename(currTempFilename, currOutputFilename)
        read1Handle.close()
        read2Handle.close()
        
        outputFilenames[j] = currOutputFilename

    # concatenate each of the output filenames

    print ''
    print 'Making final output file: "' + outputFilename + '"...'
    if len(read1Filenames)<=1:
        # don't unzip and rezip in this case (faster)
        os.rename(outputFilenames[0], outputFilename)

        # if the input wasn't zipped, unzip the output too
        if not read1Filenames[0].endswith('gz'):
            subprocess.check_call('gunzip %s' % (outputFilename),shell=True)

            print 'Sorting alphabetically by barcode: "' + outputFilename.replace('.gz','') + '"...'

            if not args.nosort:
                subprocess.check_call('sort -k7 %s > %s' % (outputFilename.replace('.gz',''), outputFilename.replace('.gz','').replace('CPseq','sort.CPseq')), shell=True)

    else:
        print read1Filenames
        call = 'zcat %s | gzip > %s'%(' '.join(outputFilenames.values()), outputFilename)
        subprocess.check_call('zcat %s | gzip > %s'%(' '.join(outputFilenames.values()), outputFilename),
                              shell=True)
        # remove individual outputs
        for filename in outputFilenames.values():
            subprocess.check_call(['rm', filename])

