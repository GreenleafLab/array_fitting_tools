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

def getFilehandle(filename, mode=None):
    """open normally if ext is not gz"""
    if os.path.splitext(filename)[-1] == '.gz':
        f = gzip.open(filename, mode)
    else:
        f = open(filename, mode)
    return f


### MAIN ###
if __name__=='__main__':

    #set up command line argument parser
    parser = argparse.ArgumentParser(description="Consolidates individual fastq files from a paired-end run (optionally with index reads, as well) into one file (.CPseq format)")

    parser.add_argument('-r1','--read1', help='read 1 fastq filename', default='greennas/151204_chip/seqData/Undetermined_S0_L001_R1_001.fastq.gz')
    parser.add_argument('-r2','--read2', help='read 2 fastq filename', default='greennas/151204_chip/seqData/Undetermined_S0_L001_R2_001.fastq.gz')
    parser.add_argument('-c','--subset_clusters', help='filename of clusters to keep', default='example_subset/clusters.txt')
    parser.add_argument('-o','--output', help='output basename (will have _R1.fastq.gz and _R2.fastq.gz appended)')

    #parse command line arguments
    args = parser.parse_args()

    clusters = np.loadtxt(args.subset_clusters, dtype=str)

    #read in sequence files and consolidate records
    read1Handle = getFilehandle(args.read1, 'rb') # input read1
    read2Handle = getFilehandle(args.read2, 'rb') # input read2
    outputRead1Handle = getFilehandle(args.output + '_R1.fastq.gz', 'wb')
    outputRead2Handle = getFilehandle(args.output + '_R2.fastq.gz', 'wb')

    filetype= 'fastq'


    # iterate through all files in lockstep
    for currRead1Record,currRead2Record in itertools.izip(SeqIO.parse(read1Handle, filetype),SeqIO.parse(read2Handle, filetype)):

        if currRead1Record.id == currRead2Record.id:
            currID = currRead1Record.id

            # check the id
            if currID in clusters:
                SeqIO.write(currRead1Record, outputRead1Handle, filetype)
                SeqIO.write(currRead2Record, outputRead2Handle, filetype)
            
    read1Handle.close()
    read2Handle.close()
    outputRead1Handle.close()
    outputRead2Handle.close()



