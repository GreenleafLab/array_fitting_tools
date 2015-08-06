# array_fitting_tools

To fit binding curves:
processData to get combined sequence and fluorescence information
findSeqDistribution to annotate variants onto sequence information
singleCluster fits to do initial, unconcstrained fits on all single clusters
bootStrapFits to revise estimates by constraining fmax and bootstrapping the fits

To fit on/off rates:
processData to get combined sequence and fluorescence information
findSeqDistribution to annotate variants onto sequence information
binTimes to make a matrix of times, binding points that can be matched tile to tile
fitOnOffRates to fit variants and boot strap fits.

Other scripts:
run_all_binding_curves.sh can be used to run all 4 scripts
compressBarcodes can take in a CPseq file and find unique barcode file. Also assesses quality
fitBackgroundTile will fit background clusters with the same method used above 
  to better understand how noise contributes to fit values
