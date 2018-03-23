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

def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = iterator.next()
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch

def getNumLinesInFile(filename):
    n_lines = 0
    with gzip.open(filename, 'rb') as f:
        for line in f: n_lines += 1
    return n_lines

### MAIN ###
if __name__=='__main__':

    #set up command line argument parser
    parser = argparse.ArgumentParser(description="Consolidates individual fastq files from a paired-end run (optionally with index reads, as well) into one file (.CPseq format)")

    parser.add_argument('-r','--read', help='fastq filename', required=True)
    parser.add_argument('-c','--subset_clusters', help='filename of clusters to keep', default='example_subset/clusters.txt')
    parser.add_argument('-o','--output', help='output basename (will have .fastq.gz appended)')

    #parse command line arguments
    args = parser.parse_args()
    clusters = np.loadtxt(args.subset_clusters, dtype=str)
    
    if args.output is None:
        args.output = fileio.stripExtension(args.read)
    
    #read in sequence files and consolidate records
    filetype= 'fastq'
    chunksize = 10000
    
    n_lines = getNumLinesInFile(args.read)
    with gzip.open(args.read, 'rb') as readHandle:
        readIter = SeqIO.parse(readHandle, filetype)
        for i, batch in enumerate(batch_iterator(readIter, chunksize)):
            # print status
            sys.stdout.write('\r%d out of %d chunks'%(i, np.ceil(float(n_lines)/chunksize)))
            sys.stdout.flush()
            # find subset
            subset_records = [record for record in batch if record.id in list(clusters)]
            # if any, save
            if subset_records:
                with gzip.open(args.output + '.fastq.gz', 'ab') as outputReadHandle:
                    for record in subset_records:
                        SeqIO.write(record, outputReadHandle, filetype)





