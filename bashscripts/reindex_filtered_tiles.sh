#!/bin/bash

# take the number of bases that occur before a given sequence (i.e. RNAP promoter)
# and make a new fastq file of these bases. Expected 16 bases.

ftd=$1
od=$2
seq=$3
filt=$4

set -e -o nounset
# help
if [ -z "$ftd" ]
then
    echo "Script to process data, annotate sequences, fit single clusters,
    and bootstrap the fits."
    echo "Arguments:
    (1) a directory of CPseq files, (i.e. filtered)
    (2) an output directory that will hold the indexed files
    (3) a sequence
    (4) (optional) a filter, st you only check those with a certain filter."
    echo "Example:"
    echo "reindex_filtered_tiles.sh \\
    tiles/filtered_tiles/ \\
    tiles/filtered_tiles_indexed/ \\
    TTTATGCTATAA \\"
    exit
fi

files=$ftd/*CPseq
if [ ! -d $od ];
then
    mkdir $od
fi

if [ $# -eq 3 ];
then
    echo "no filter supplied"
    
    for file in $files;
    do 
        awk -v seq=$seq 'BEGIN{FS="\t";}{OFS="\t"}{i=index($3, seq); 
                               if (i>0) {print $1,$2,$3,$4,$5,$6,substr($3, 1, i-1),
                               substr($4,1, i-1),$9,$10} else print $0}' $file > $od/$(basename $file)
    done
else
    echo "using filter "$filt
    for file in $files;
    do 
        awk -v seq=$seq -v filt=$filt 'BEGIN{FS="\t";}{OFS="\t"}{j=index($2, filt); i=index($3, seq); 
                               if (i>0 && j>0) {print $1,$2,$3,$4,$5,$6,substr($3, 1, i-1),
                               substr($4,1, i-1),$9,$10} else print $0}' $file > $od/$(basename $file)
    done
fi
    #
## or for already reduced CPseq file
#file=AG1D1_tecto.CPseq
#cat $file |  awk -v seq=$seq '{OFS="\t"}{i=index($3, seq); if (i>0) {print $1,$2,$3,$4,$5,$6,substr($3, 0, i-1),substr($4,0, i-1),$9,$10} else print $0}' | head