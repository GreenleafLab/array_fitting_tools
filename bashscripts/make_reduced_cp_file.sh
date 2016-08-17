#!/bin/bash

# take the number of bases that occur before a given sequence (i.e. RNAP promoter)
# and make a new fastq file of these bases. Expected 16 bases.

ftd=$1
out=$2
filters=$(echo $@ | awk '{for (i=3; i<=NF; i++) print $i}')


set -e -o nounset
# help
if [ -z "$ftd" ]
then
    echo "Script to tak CPseq files and make CPannot file.
    "
    echo "Arguments:
    (1) a directory of CPseq files, (i.e. filtered)
    (2) an output file
    (3) filters to make reduced CPseq file. If more than one, separate by space."
    echo "Example:"
    exit
fi

files=$ftd/*CPseq


    #
## or for already reduced CPseq file
#file=AG1D1_tecto.CPseq
formatfilters=$(echo $filters | sed 's/ /|/g')
#echo "grep -E '$formatfilters' <(cat $ftd/*CPseq) > $out"
grep -E '$formatfilters' <(cat $ftd/*CPseq) 