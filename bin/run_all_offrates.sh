#!/bin/bash

# take the number of bases that occur before a given sequence (i.e. RNAP promoter)
# and make a new fastq file of these bases. Expected 16 bases.

ftd=$1
mf=$2
od=$3
lc=$4
bar=$5
f=$(echo $@ | awk '{for (i=6; i<=NF; i++) print $i}')

set -e -o nounset
# help
if [ -z "$ftd" ]
then
    echo "Script to process data, annotate sequences, bin times,
    and bootstrap the fits."
    echo "Arguments:
    (1) a directory of CPseq files,
    (2) a map file of CPfluor dirs,
    (3) an output directory,
    (4) a library characterization file,
    (5) a unique barcodes file,
    (6) a list of filter names to fit."
    echo "Example:"
    echo "run_all_offrates.sh \\
    ../seqData/tiles/filtered_tiles_indexed/ \\
    offRates/offrates.map \\
    offRates \\
    ../../150311_library_v2/all_10expts.library_characterization.txt \\
    ../../150608_barcode_mapping_lib2/tecto_lib2.150728.unique_barcodes \\
    anyRNA "
    exit
fi

# process data
python -m processData -fs $ftd -mf $mf -od $od -fp $f -r
output=$(find $od -maxdepth 1  -name "*CPsignal" -type f)

# check success
if [ $? -eq 0 -a -f $output ];
then
    echo "Successfully processed data"
else
    echo "Error processing data"
    exit
fi

basename=$(echo $output | awk '{print substr($1, 1, length($1)-9)}')

# annotate data
if [ -f $basename".CPannot.pkl" ];
then
    echo "CPannot file exists: "$basename".CPannot.pkl"
else
    echo "python -m findSeqDistribution -lc $lc -cs $basename".CPsignal.pkl" -bar $bar"
    python -m findSeqDistribution -lc $lc -cs $basename".CPsignal.pkl" -bar $bar
    
    # check success
    if [ $? -eq 0 ]
    then
        echo "Successfully annotated data"
    else
        echo "Error annotating data"
        exit
    fi
fi

# bin the times
if [ -f $basename".bindingSeries.pkl" ];
then
    echo "Time series file exists: "$basename".bindingSeries.pkl"
else
    python -m binTimes -cs $basename".CPsignal.pkl" -td $od"/rates.timeDict.pkl"
    
    # check success
    if [ $? -eq 0 ]
    then
        echo "Successfully binned times"
    else
        echo "Error binning times"
        exit
    fi
fi

# bin the times
if [ -f $basename".CPresults" ];
then
    echo "Fit results file exists: "$basename".CPresults.pkl"
else
    echo "python -m fitOnOffRates -a $basename".CPannot.pkl" -t $basename".times.txt" -b $basename".bindingSeries.pkl" -n 20"
    python -m fitOnOffRates -a $basename".CPannot.pkl" -t $basename".times.txt" -b $basename".bindingSeries.pkl" -n 20
    
    # check success
    if [ $? -eq 0 ]
    then
        echo "Successfully binned times"
    else
        echo "Error binning times"
        exit
    fi
fi