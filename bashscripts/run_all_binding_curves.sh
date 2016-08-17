#!/bin/bash

# take the number of bases that occur before a given sequence (i.e. RNAP promoter)
# and make a new fastq file of these bases. Expected 16 bases.

mf=$1
od=$2
an=$3
c=$4


set -e -o nounset
# help
if [ -z "$mf" ]
then
    echo "Script to process data, annotate sequences, fit single clusters,
    and bootstrap the fits."
    echo "Arguments:
    (1) a map file of CPfluor dirs,
    (3) an output directory,
    (4) the CPannot file generated per chip
    (5) file of concentrations
    (6) a list of filter names to fit."
    echo "Example:"
    echo "run_all_binding_curves.sh \\
    bindingCurves/bindingCurves.map \\
    bindingCurves \\
    ../seqData/dummy_dir/AKPP5_ALL_Bottom_filtered_anyRNA.CPannot.pkl \\
    concentrations.txt "
    exit
fi

# print command
echo ""
echo "run_all_binding_curves.sh $mf $od $an $c"
echo "start run at: "
date

# process data
echo ""
echo "python -m processData -mf $mf -od $od -cf $an"
python -m processData -mf $mf -od $od -cf $an
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

# fit single clusters
extension=".CPfitted.pkl" 
if [ -f $normbasename$extension ];
then
    echo "--> CPfitted file exists: "$normbasename$extension
else
    echo ""
    echo "python -m singleClusterFits -cs $basename".CPseries.pkl" -c $c -n 20"
    python -m singleClusterFits -b $normbasename".CPseries.pkl" -c $c -n 20

    # check success
    if [ $? -eq 0 ]
    then
        echo "### Successfully fit single clusters. ###"
    else
        echo "!!! Error fitting single clusters !!!"
        exit
    fi
    date
fi

# find distribution of fmaxes
extension=".fmaxdist.p"
# bootstrap variants 
if [ -f $normbasename$extension ];
then
    echo "--> fmax dist file exists: "$normbasename$extension
else
    echo ""
    echo "python -m findFmaxDist -cf $normbasename.CPfitted.pkl -a $an -c $c"
    python -m findFmaxDist -cf $normbasename.CPfitted.pkl -a $an -c $c
    # check success
    if [ $? -eq 0 ]
    then
        echo "### Successfully fit fmax dist. ###"
    else
        echo "!!! Error fitting fmax dist !!!"
        exit
    fi
    date
fi

# bootstrap variants
extension=".CPvariant"
if [ -f $normbasename$extension ];
then
    echo "--> CPvariant file exists: "$normbasename$extension
else
    echo ""
    echo "python -m bootStrapFits -v "$normbasename".init.CPvariant.pkl -c "$c" -a "$an" -b $normbasename".CPseries.pkl" -f "$normbasename".fmaxdist.p -n 20"
    python -m bootStrapFits -v $normbasename.init.CPvariant.pkl -c $c -a $an -b $normbasename.CPseries.pkl -f $normbasename.fmaxdist.p -n 20

    # check success
    if [ $? -eq 0 ]
    then
        echo "### Successfully bootstrapped fits. ###"
    else
        echo "!!! Error fitting bootstrapping fits !!!"
        exit
    fi
    date
fi





