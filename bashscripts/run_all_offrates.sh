#!/bin/bash

# take the number of bases that occur before a given sequence (i.e. RNAP promoter)
# and make a new fastq file of these bases. Expected 16 bases.

ftd=$1
mf=$2
od=$3
an=$4
f=$(echo $@ | awk '{for (i=5; i<=NF; i++) print $i}')

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
    (4) an annotated cluster file per chip
    (5) a list of filter names to fit."
    echo "Example:"
    echo "run_all_offrates.sh \\
    ../seqData/tiles/filtered_tiles_indexed/ \\
    offRates/offrates.map \\
    offRates \\
    ../seqData/AG3EL_Bottom_filtered_anyRNA.CPannot.pkl \\ 
    anyRNA "
    exit
fi

# print command
echo ""
echo "run_all_offrates.sh $ftd $mf $od $an $f"
echo "start run at: "
date

# process data
output=$(find $od -maxdepth 1  -name "*reduced.CPseries.pkl" -type f)
if [ -z $output ];
then
    echo ""
    echo "python -m processData -fs $ftd -mf $mf -od $od -fp $f -cf $an -r"
    python -m processData -fs $ftd -mf $mf -od $od -fp $f -cf $an -r    
    output=$(find $od -maxdepth 1  -name "*reduced.CPseries.pkl" -type f)

    # check success
    if [ $? -eq 0 -a -f $output ];
    then
        echo "### Successfully processed data ###"
    else
        echo "!!! Error processing data !!!"
        exit
    fi
    date
else
    echo "--> reduced CPseries file exists: "$output
fi

basename=$(echo $output | awk '{print substr($1, 1, length($1)-13)}')

# normalize data
extension="_normalized.CPseries.pkl"
if [ -f $basename$extension ];
then
    echo "--> normalized CPseries file exists: "$basename$extension
else
    echo ""
    echo "python -m normalizeSeries -b "$basename".CPseries.pkl -a "$basename"_red.CPseries.pkl"
    
    python -m normalizeSeries -b $basename.CPseries.pkl -a $basename"_red.CPseries.pkl"
    # check success
    if [ $? -eq 0 ]
    then
        echo "### Successfully normalized data ###"
    else
        echo "!!! Error normalizing data !!!"
        exit
    fi
    date

fi

normbasename=$basename"_normalized"

# bin the times
if [ -f $normbasename$extension ];
then
    echo "--> time series file exists: "$normbasename$extension
else
    echo ""
    echo "python -m binTimes -cs $normbasename.CPseries.pkl -td $od/rates.timeDict.p -t $basename.CPtiles.pkl"
    python -m binTimes -cs $normbasename.CPseries.pkl -td $od/rates.timeDict.p -t $basename.CPtiles.pkl
    
    # check success
    if [ $? -eq 0 ]
    then
        echo "### Successfully binned times ###"
    else
        echo "!!! Error binning times !!!"
        exit
    fi
    date
fi

# fit single clusters
extension=".CPfitted.pkl"
if [ -f $normbasename$extension ];
then
    echo "--> fit results file exists: "$normbasename$extension
else
    echo ""
    echo "python -m fitRatesPerCluster -cs $normbasename.CPseries.pkl -t $basename.CPtiles.pkl -td $od/rates.timeDict.p  -n 20 --pb_correct"
    python -m fitRatesPerCluster -cs $normbasename.CPseries.pkl -t $basename.CPtiles.pkl -td $od/rates.timeDict.p -n 20 --pb_correct
    
    # check success
    if [ $? -eq 0 ]
    then
        echo "### Successfully fit off rates ###"
    else
        echo "!!! Error fitting off rates !!!"
        exit
    fi
    date
fi

# bootsrap
extension=".CPvariant"
if [ -f $normbasename$extension ];
then
    echo "--> per variant file exists: "$normbasename$extension
else
    echo ""
    echo "python -m bootStrapFitFile -cf $normbasename.CPfitted.pkl -a $an -n 20 -p koff"
    python -m bootStrapFitFile -cf $normbasename.CPfitted.pkl -a $an -n 20 -p koff
    
    # check success
    if [ $? -eq 0 ]
    then
        echo "### Successfully fit off rates ###"
    else
        echo "!!! Error fitting off rates !!!"
        exit
    fi
    date
fi