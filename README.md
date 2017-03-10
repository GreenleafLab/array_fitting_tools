# array_fitting_tools

add to python path: `array_fitting_tools/bin`

## Requirements (parantheses indicate what I am using):
Please update this list as you encounter dependencies not listed here!

* pandas (0.16.2)
* lmfit (0.8.3) 
** Note that [lmfit 0.9.0+](https://lmfit.github.io/lmfit-py/faq.html#why-did-my-script-break-when-upgrading-from-lmfit-0-8-3-to-0-9-0) would NOT work with the current pipeline. 
* joblib (0.9.0b2)
* seaborn (0.7.0)
* scikits.bootstrap (0.3.2)
* statsmodels (0.6.1)
* scipy (0.16.0)
* numpy (1.11.1)

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
    * This one probably won't work out of the box: talk to me.
* `medianSubsetCPseries` will take the per-variant median of a CPseries, and optionally subset to only include one tile.


## More
For more info, see the [wiki page](https://github.com/GreenleafLab/array_fitting_tools/wiki)
