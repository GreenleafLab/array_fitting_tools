#!/usr/bin/env python
""" Find the fmax distribution.

Returns a class describing the fmax.

Parameters:
-----------
variant_table : per-variant DataFrame including columns fmax, dG, and fmin, pvalue
initial_points : per-cluster DataFrame including variant_number, fmax, dG, fmin
affinity_cutoff : maximum dG (kcal/mol) to be included in fit of fmax
use_simulated : (bool) whether to use distribution of median fmax or
    subsampled fmaxes

Returns:
--------
fmaxDistObject :
"""

import os
import numpy as np
import scipy.stats as st
import seqfun
from lmfit import Parameters, minimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import argparse
import sys
import pickle
from fittinglibs import (fitting, plotting, fileio, processing, distribution)



parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-cf', '--single_cluster_fits', required=True, metavar=".CPfitted.pkl",
                   help='file with single cluster fits')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')

group = parser.add_argument_group('additional option arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')
group.add_argument('-c', '--concentrations', metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
group.add_argument('-k', '--kd_cutoff', type=float,  
                   help='highest kd for tight binders (nM). default is 0.99 bound at '
                   'highest concentration')
group.add_argument('-p', '--pvalue_cutoff', type=float, default=0.01,
                   help='maximum pvalue for good binders. default is 0.01.')
group.add_argument('--use_simulated', type=int,
                   help='set to 0 or 1 if you want to use simulated distribution (1) or'
                   'not (0). Otherwise program will decide.')




def useSimulatedOrActual(variant_table, cutoff):
    # if at least 20 data points have at least 10 counts in that bin, use actual
    # data. This statistics seem reasonable for fitting
    index = variant_table.dG_init < cutoff
    counts, binedges = np.histogram(variant_table.loc[index].numTests,
                                    np.arange(1, variant_table.numTests.max()))
    if (counts > 10).sum() >= 20:
        use_actual = True
    else:
        use_actual = False
    return use_actual

if __name__=="__main__":
    args = parser.parse_args()
    fittedBindingFilename = args.single_cluster_fits
    annotatedClusterFile  = args.annotated_clusters

    outFile  = args.out_file
    kd_cutoff = args.kd_cutoff
    use_simulated = args.use_simulated
    
    # find out file
    if outFile is None:
        outFile = os.path.splitext(
            fittedBindingFilename[:fittedBindingFilename.find('.pkl')])[0]
        
    # make fig directory
    figDirectory = os.path.join(os.path.dirname(outFile), fileio.returnFigDirectory())
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)

    # need to define concentrations or kd_cutoff in order to find affinity cutoff
    if args.concentrations is not None:
        concentrations = np.loadtxt(args.concentrations)
    elif kd_cutoff is None:
        print 'Error: need to either give concentrations or kd_cutoff.'
        sys.exit()
        
    # define cutoffs
    parameters = fitting.fittingParameters()
    if kd_cutoff is not None:
        # adjust cutoff to reflect this fraction bound at last concentration
        affinity_cutoff = parameters.find_dG_from_Kd(kd_cutoff)
    else:
        parameters = fitting.fittingParameters(concentrations)
        affinity_cutoff = parameters.maxdG
    pvalue_cutoff = args.pvalue_cutoff
    print 'Using variants with kd less than %4.2f nM'%parameters.find_Kd_from_dG(affinity_cutoff)
    print 'Using variants with pvalue less than %.1e'%pvalue_cutoff

    # load initial fits per cluster
    print 'Loading data...'
    initial_points = pd.concat([pd.read_pickle(annotatedClusterFile),
                                pd.read_pickle(fittedBindingFilename).astype(float)], axis=1)

    # find variant_table
    variant_table = processing.findVariantTable(initial_points).astype(float)
    good_fits = variant_table.pvalue < pvalue_cutoff
    tight_binders = variant_table.loc[good_fits&(variant_table.dG_init<affinity_cutoff)]
    print ('%d out of %d variants pass cutoff'
           %(len(tight_binders), len(variant_table)))
    if len(tight_binders) < 10:
        print 'Error: need more variants passing cutoffs to fit'
        print sys.exit()

    # find good variants
    plotting.plotFmaxVsKd(variant_table.loc[good_fits], parameters.find_Kd_from_dG(affinity_cutoff))
    plt.savefig(os.path.join(figDirectory, 'fmax_vs_kd_init.pdf'))
    
    plotting.plotFmaxVsKd(variant_table.loc[good_fits], parameters.find_Kd_from_dG(affinity_cutoff),
                         plot_fmin=True)
    plt.savefig(os.path.join(figDirectory, 'fmin_vs_kd_init.pdf'))  

    # if use_simulated is not given, decide
    if use_simulated is None:
        use_simulated = not useSimulatedOrActual(variant_table.loc[good_fits], affinity_cutoff)
    if use_simulated:
        print 'Using fmaxes drawn randomly from clusters'
    else:
        print 'Using median fmaxes of variants'
    
    # find fmax dist object
    fmaxDist = distribution.findParams(tight_binders,
                                use_simulated=use_simulated,
                                table=initial_points)
    plt.savefig(os.path.join(figDirectory, 'counts_vs_n_tight_binders.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'offset_fmax_vs_n.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'stde_fmax_vs_n.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'fmax_dist.all.pdf'));
    
    # save
    pickle.dump(fmaxDist, open( outFile+'.fmaxdist.p', "wb" ))
    
    # what about fmin?
    print 'fmin should be: %4.3f'%distribution.returnFminFromFits(variant_table, parameters.mindG)
    
    # save variant table
    variant_table.to_pickle(outFile + '.init.CPvariant.pkl')
    
    # plot some examples
    for n in np.percentile(variant_table.numTests, [10, 50, 90]):
        bounds = [0, distribution.findUpperboundFromFmaxDistObject(fmaxDist)]
        plotting.plotAnyN(tight_binders, fmaxDist, n, bounds)
        plt.savefig(os.path.join(figDirectory, 'fmax_dist.n_%d.pdf'%n))
    
    # plot fraction fit
    plotting.plotFractionFit(variant_table, pvalue_threshold=pvalue_cutoff)
    plt.savefig(os.path.join(figDirectory, 'fraction_passing_cutoff_in_affinity_bins.pdf'))
    plt.close()
    plt.savefig(os.path.join(figDirectory, 'histogram_fraction_fit.pdf'))
    

                
    