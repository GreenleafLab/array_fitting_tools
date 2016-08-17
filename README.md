# array_fitting_tools

add to python path: `array_fitting_tools/bin`

## scripts: 
`findSeqDistribution` to annotate variants onto clusters.

### To fit binding curves:
* `processData` to make series file from CPfluor directories.
* `normalizeSeries` to normalize fluoresecence by all-cluster images.
* `singleClusterFits` to do initial, minimally constrained fits on all single clusters.
* `findFmaxDist` to fit the distribution of fmax of good variants.
* `bootStrapFits` to revise estimates by constraining fmax and bootstrapping the fits.

### To fit on/off rates:
* `processData` to make series file from CPfluor directories.
* `normalizeSeries` to normalize fluoresecence by all-cluster images.
* `fitRatesPerCluster` to fit individual clusters to on/off rates.
* `bootStrapFitFile` to bootstrap fit parameter to obtain 95% confidence intervals.

### Other scripts:
* `run_all_binding_curves.sh` can be used to run all 4 scripts.
* `compressBarcodes` can take in a CPseq file and find unique barcode file. Also assesses quality.
* `fitBackgroundTile` will fit background clusters with the same method used above to better understand how noise contributes to fit values

## More
For more info, see the [wiki page](https://github.com/GreenleafLab/array_fitting_tools/wiki)