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
import seqfun
import IMlibs
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='map variants to sequence')
parser.add_argument('-lc', '--library_characterization', required=True,
                   help='file that lists unique variant sequences')
parser.add_argument('-out', '--out_file', 
                   help='output filename')
parser.add_argument('-cs', '--cpsignal', required=True,
                   help='reduced CPsignal file containing sequence information.')
parser.add_argument('-bar', '--unique_barcodes',
                   help='barcode map file. if given, the variant sequences are '
                   'mapped to the barcode rather than directly on to the sequence data')


group = parser.add_argument_group('additional option arguments to map variants')
group.add_argument('--barcodeCol', default='index1_seq',
                   help='if using barcode map, this indicates the column of CPsignal'
                   'file giving the barcode. Default is "index1_seq"')
group.add_argument('--seqCol', default='read2_seq',
                   help='when looking for variants, look within this sequence. '
                   'Default is "read2_seq"')
group.add_argument('--noReverseComplement', default=False, action="store_true",
                   help='when looking for variants, default is to look for the'
                   'reverse complement. Flag if you want to look for forward sequence')

group.add_argument('--save_text', default=False, action="store_true",
                   help='default is to save output as binary. Flag if you wish to save as text'
                   ' file instead')

def findSequenceRepresentation(consensus_sequences, compare_to, exact_match=None):
    # initialize
    if exact_match is None:
        exact_match = False # default is to search for whether it contains the sequence .
                            # set to True if the sequences must match exactly
    num_bc_per_variant = pd.Series(index=compare_to.index, dtype=int) # number consensus sequences per designed sequence
    is_designed = np.ones(len(consensus_sequences))*np.nan # -1 if no designed sequence is found that matches. else index
    
    # cycle through designed sequences. Find if they are in the actual sequences
    for i, idx in enumerate(compare_to.index):
        sequence = compare_to.loc[idx]
        if i%1000==0:
            print "checking %dth sequence"%i
        
        # whether the sequence (designed) is in the actual sequence, given by the fastq
        in_fastq = True
    
        # start count
        count = -1
        
        # first location in the sorted list that the sequence might match
        first_index = np.searchsorted(consensus_sequences, sequence)
        
        # starting from the first index given by searching the sorted list,
        # cycle until the seuqnece is no longer found
        while in_fastq:
            count += 1
            if first_index+count < len(consensus_sequences):
                if exact_match:
                    in_fastq = consensus_sequences[first_index+count] == sequence
                else:
                    in_fastq = consensus_sequences[first_index+count].find(sequence)==0
            else:
                in_fastq = False
            # if the designed sequence is in the most probable location of the
            # sorted consensus sequences, give 'is_designed' at that location to the
            # indx of the matching sequence in 'compare_to'
            if in_fastq:
                is_designed[first_index+count] = idx
        num_bc_per_variant.loc[idx] = count
    return num_bc_per_variant, is_designed

def findSeqMap(libCharacterizationFile, cpSignalFile, uniqueBarcodesFile=None,
         reverseComplement=None, seqCol=None, barcodeCol=None, mapToBarcode=None):
    if reverseComplement is None:
        reverseComplement = True
    if mapToBarcode is None:
        if uniqueBarcodesFile is not None:
            mapToBarcode = True
        else:
            mapToBarcode = False
    if not mapToBarcode and seqCol is None:
        print 'Error: need to define seqCol if using cpSignal file'
        return
    
    if mapToBarcode and barcodeCol is None:
        print 'Error: need to define barcodeCol if using unique barcodes file'
        return        
    
    print "loading consensus sequences..."
    if uniqueBarcodesFile is not None:
        consensus = IMlibs.loadCompressedBarcodeFile(uniqueBarcodesFile)
        
        # if using the unique barcode map, the unique identifier is the barcode
        identifyingColumn = 'barcode'
    elif cpSignalFile is not None:
        consensus = IMlibs.loadCPseqSignal(cpSignalFile, index_col='tileID')
        consensus.loc[:, 'sequence'] = consensus.loc[:, seqCol]
        consensus.loc[:, 'tileID'] = consensus.index
        # if using the cpsignal, the unique identifier is the cluster Id
        identifyingColumn = 'tileID'
    
    # sort by sequence to do sorted search later on
    consensus.sort('sequence', inplace=True)

    print "loading designed sequences..."
    designed_library = IMlibs.loadLibraryCharacterization(libCharacterizationFile)
    
    # make library sequences unique
    designed_sequences, unique_indices = np.unique(designed_library['sequence'], return_index=True)
    designed_library_unique = designed_library.iloc[unique_indices].copy()
    print "reduced library size from %d to %d after unique filter"%(len(designed_library), len(designed_library_unique))
    
    ## add field in designed_library_unique which gives an int to that sequence
    #designed_library_unique.insert(0, 'variant_number', unique_indices.astype(int))
    
    # figure out read length
    read_length = len(consensus.sequence.iloc[0])
    
    # reformat designed sequences to be reverse complement (as in read 2)
    if reverseComplement:
        designed_library_unique.loc[:, 'rc_sequence_trunc'] = (
            [seqfun.reverseComplement(sequence)[:read_length]
             for sequence in designed_library_unique.sequence])
        compare_to = designed_library_unique.rc_sequence_trunc
    else:
        designed_library_unique.loc[:, 'sequence_trunc'] = (
            [sequence[:read_length]
             for sequence in designed_library_unique.sequence])
        compare_to = designed_library_unique.sequence_trunc
    
    # find number of times each sequence that has at least one representation is in the
    # original block of seqeunces
    num_bc_per_variant, is_designed = findSequenceRepresentation(
        consensus.sequence.values, compare_to)
    
    # is_designed gives the variant number for those clusters that successfully mapped
    barcodeMap = pd.concat([pd.DataFrame(is_designed, consensus.index,
                                         columns=['variant_number']),
                            consensus], axis=1)
    barcodeMap.index = barcodeMap.loc[:, identifyingColumn]
    barcodeMap.sort('variant_number',  inplace=True)
    barcodeMap.drop(identifyingColumn, axis=1, inplace=True)
    
    # return not the barcode map but the seqMap. If mapped di
    print 'Mapping to cluster IDs...'
    cols = ['variant_number']
    if mapToBarcode:
        identifyingColumn = 'tileID'
        table = IMlibs.loadCPseqSignal(cpSignalFile, usecols=[identifyingColumn, barcodeCol],
                                       index_col=identifyingColumn)
        
        
        seqMap = pd.DataFrame(index=table.index, columns=cols)
        index = table.loc[:, barcodeCol].dropna()
        seqMap.loc[index.index, cols] = barcodeMap.loc[index, cols].values

        seqMap.sort('variant_number', inplace=True)
    else:
        seqMap = barcodeMap.loc[:, cols]

    
    if mapToBarcode:
        logText = ['Used Barcode File: %s'%uniqueBarcodesFile,
            '\t%d unique barcodes total'%len(barcodeMap),
            '\t%d (%4.1f%%) mapped to library variant'%(len(barcodeMap.dropna(subset=['variant_number'])),
                                                       len(barcodeMap.dropna(subset=['variant_number']))/
                                                       float(len(barcodeMap))*100)]
    else: logText = []
    logText = (logText + 
            ['Mapped to clusters in %s'%cpSignalFile,
            '\t%d clusters total'%len(seqMap),
            '\t%d (%4.1f%%) mapped to library variant'%(len(seqMap.dropna(subset=['variant_number'])),
                                                       len(seqMap.dropna(subset=['variant_number']))/
                                                       float(len(seqMap))*100)])
    print '\n'.join(logText)

    return seqMap

##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()
    
    libCharacterizationFile = args.library_characterization
    cpSignalFile = args.cpsignal
    uniqueBarcodesFile = args.unique_barcodes
    reverseComplement = not args.noReverseComplement
    seqCol = args.seqCol
    barcodeCol = args.barcodeCol
    outFile = args.out_file

    if outFile is None:
        outFile = os.path.splitext(
            cpSignalFile[:cpSignalFile.find('.pkl')])[0]
    
    
    seqMap = findSeqMap(args.library_characterization, args.cpsignal,
                  uniqueBarcodesFile=args.unique_barcodes,
                      reverseComplement=not args.noReverseComplement,
                      seqCol=args.seqCol,
                      barcodeCol=args.barcodeCol)

    if args.save_text:
        seqMap.to_csv(outFile + '.CPannot', index=True, float_format='%4.0f', sep='\t')
    else:
        seqMap.to_pickle(outFile + '.CPannot.pkl')

 
