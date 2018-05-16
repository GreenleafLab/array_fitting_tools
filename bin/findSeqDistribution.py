#!/usr/bin/env python
# 
# Given a list of designed variants, and a CPsignal file you wish to fit,
# annotate each of the clusters in cPsignal file with variant information.
#
# Uses either unique_barcode file (from compressBarcodes.py), or looks
# within the sequence directly for the variant
# If using the unique_barcodes file, you should already have filtered for
# 'good' barcodes.
#
# Sarah Denny

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
#import ipdb
#import seaborn as sns
from fittinglibs import fileio, seqfun
#sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='map variants to sequence')
parser.add_argument('-lc', '--library_characterization', required=True,
                   help='file that lists unique variant sequences')
parser.add_argument('-out', '--out_file', 
                   help='output filename')
parser.add_argument('-cs', '--cpseq', required=True,
                   help='reduced CPseq file containing sequence information.')
parser.add_argument('-bar', '--unique_barcodes', required=True,
                   help='barcode map file. if given, the variant sequences are '
                   'mapped to the barcode rather than directly on to the sequence data')


group = parser.add_argument_group('additional option arguments to map variants')
group.add_argument('--barcodeCol', default='index1_seq',
                   help='if using barcode map, this indicates the column of CPsignal'
                   'file giving the barcode. Default is "index1_seq"')



def findSequenceRepresentation(consensus_sequences, compare_to, exact_match=False):
    """ Find the sequence matches between two lists of sequences.
    
    Inputs:
    consensus_sequences: is the list of sequencing results, sorted
    compare_to: the list of designed variants
    exact_match: if set, the designed variant sequence must be an exact match of
        the sequencing result. The default is that the designed variant must be
        entirely contained in the sequencing result and must start at the first
        index. This less stringent default criteria should work in most cases and
        avoids having to truncate the sequencing results to the designed variant
        lengths.
    Outputs:
    num_bc_per_variant: list of the same length as compare_to giving the
        number of sequenceing results that match a designed library variant.
    is_designed: list of the same length as consensus_sequences giving which
        designed variant it aligns to.
    """
    # initialize
    num_bc_per_variant = pd.Series(index=compare_to.index, dtype=int) # number consensus sequences per designed sequence
    is_designed = pd.Series(index=consensus_sequences.index)
    if not isinstance(consensus_sequences, pd.Series):
        consensus_sequences = pd.Series(consensus_sequences)
    consensus_seqlist = consensus_sequences.astype(str).values
    consensus_seqindex = consensus_sequences.index.tolist()
    # cycle through designed sequences. Find if they are in the actual sequences
    for i, (idx, sequence) in enumerate(compare_to.iteritems()):
        if i%1000==0:
            print "checking %dth sequence"%i
        
        # whether the sequence (designed) is in the actual sequence, given by the fastq
        in_fastq = True
    
        # start count
        count = -1
        
        # first location in the sorted list that the sequence might match
        first_index = np.searchsorted(consensus_seqlist, sequence)
        
        # starting from the first index given by searching the sorted list,
        # cycle until the seuqnece is no longer found
        while in_fastq:
            count += 1
            if first_index+count < len(consensus_seqlist):
                if exact_match:
                    in_fastq = consensus_seqlist[first_index+count] == sequence
                else:
                    in_fastq = consensus_seqlist[first_index+count].find(sequence)==0
            else:
                in_fastq = False
            # if the designed sequence is in the most probable location of the
            # sorted consensus sequences, give 'is_designed' at that location to the
            # indx of the matching sequence in 'compare_to'
            if in_fastq:
                is_designed[first_index+count] = idx
        num_bc_per_variant.loc[idx] = count
    
    return num_bc_per_variant, pd.Series(is_designed, consensus_seqindex)


def plotNumberClustersPerVariant(seqMap):
    """ Plot a histogram of number of clusters per variant. """
    # aggregate seqMap data by variant
    counts = seqMap.fillna(0).variant_number.value_counts()
    plt.figure()
    sns.distplot(counts, bins=1, kde=False)
    return


##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()


    if args.out_file is None:
        args.out_file = fileio.stripExtension(args.cpseq) + '.CPannot.gz'
        
    # load consensus seqs of UMI
    consensus_seqs = fileio.loadFile(args.unique_barcodes).set_index('barcode').sequence.sort_values()
    read_length = consensus_seqs.str.len().value_counts().index.tolist()[0] # mode of all
    
    # load designed sequences, make  sequences unique
    lib_char = fileio.loadFile(args.library_characterization)
    designed_sequences, unique_indices = np.unique(lib_char.sequence, return_index=True)
    
    # assign unique sequences
    lib_char.index.name = 'variant_number'
    lib_char_unique = lib_char.reset_index().groupby('sequence').first().reset_index().set_index('variant_number')
    print "reduced library size from %d to %d after unique filter"%(len(lib_char), len(lib_char_unique))
    
    # make sure the length is less than the read length
    lib_char_trimmed = lib_char_unique.loc[lib_char_unique.sequence.str.len() <= read_length].copy()
    print "reduced library size from %d to %d after trimming for read length"%(len(lib_char_unique), len(lib_char_trimmed))

    # take the reverse complement of the designed sequences, as we are comparing to read2
    lib_char_trimmed.loc[:, 'rc_sequence_trunc'] = (
        [seqfun.reverseComplement(sequence)[:read_length]
         for sequence in lib_char_trimmed.sequence])
    
    # assign each row in consensus seqs to a variant in lib_char_trimmed
    num_bc_per_variant, umi_designed_variant = findSequenceRepresentation(
        consensus_seqs, lib_char_trimmed.rc_sequence_trunc)
    
    # assign each row in cpseq file to a variant
    print "mapping variant ids to cluster id through UMI..."
    sequence_data = fileio.loadFile(args.cpseq)
    print "\tfinding subset of clusters with UMI..."
    cluster_to_umi = sequence_data.loc[np.in1d(sequence_data.loc[:, args.barcodeCol].tolist(), umi_designed_variant.index.tolist()),  args.barcodeCol].copy()
    print "\tmapping clusters to variant..."
    cluster_to_variant = pd.Series(umi_designed_variant.loc[cluster_to_umi].values, index=cluster_to_umi.index)
    print "\tSaving..."
    annotated_clusters = pd.DataFrame(cluster_to_variant.loc[sequence_data.index].rename('variant_number'))
    annotated_clusters.to_csv(args.out_file, sep='\t', compression='gzip')
    



 
