#!/bin/bash

# take the number of bases that occur before a given sequence (i.e. RNAP promoter)
# and make a new fastq file of these bases. Expected 16 bases.

ftd=$1
mf=$2
od=$3
lc=$4
bar=$5
c=$6
f=$(echo $@ | awk '{for (i=7; i<=NF; i++) print $i}')

set -e -o nounset
# help
if [ -z "$ftd" ]
then
    echo "Script to process data, annotate sequences, fit single clusters,
    and bootstrap the fits."
    echo "Arguments:
    (1) a directory of CPseq files,
    (2) a map file of CPfluor dirs,
    (3) an output directory,
    (4) a library characterization file,
    (5) a unique barcodes file,
    (6) file of concentrations
    (7) a list of filter names to fit."
    echo "Example:"
    echo "run_all_binding_curves.sh \\
    ../seqData/tiles/filtered_tiles_indexed/ \\
    bindingCurves/bindingCurves.map \\
    bindingCurves \\
    ../../150311_library_v2/all_10expts.library_characterization.txt \\
    ../../150608_barcode_mapping_lib2/tecto_lib2.150728.unique_barcodes \\
    concentrations.txt \\
    anyRNA "
    exit
fi

# print command
echo "run_all_binding_curves.sh $ftd $mf $od $lc $bar $c $f"

# process data
echo "python -m processData -fs $ftd -mf $mf -od $od -fp $f"
python -m processData -fs $ftd -mf $mf -od $od -fp $f
output=$(find $od -maxdepth 1  -name "*CPsignal.pkl" -type f)

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
    echo " python -m findSeqDistribution -lc $lc -cs $basename".CPsignal.pkl" -bar $bar"
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

# fit single clusters 
if [ -f $basename".CPfitted.pkl" ];
then
    echo "CPfitted file exists: "$basename".CPfitted.pkl"
else
    echo "python -m singleClusterFits -cs $basename".CPsignal.pkl" -c $c -n 20"
    python -m singleClusterFits -cs $basename".CPsignal.pkl" -c $c -n 20

    # check success
    if [ $? -eq 0 ]
    then
        echo "Successfully fit single clusters."
    else
        echo "Error fitting single clusters"
        exit
    fi
fi

# bootstrap variants 
if [ -f $basename".CPvariant" ];
then
    echo "CPfitted file exists: "$basename".CPvariant"
else
    echo "python -m bootStrapFits -t $basename".CPfitted.pkl" -c $c -a $basename".CPannot.pkl" -b $basename".bindingSeries.pkl" -n 20"
    python -m bootStrapFits -t $basename".CPfitted.pkl" -c $c -a $basename".CPannot.pkl" -b $basename".bindingSeries.pkl" -n 20

    # check success
    if [ $? -eq 0 ]
    then
        echo "Successfully bootstrapped fits."
    else
        echo "Error fitting bootstrapping fits"
        exit
    fi
fi



