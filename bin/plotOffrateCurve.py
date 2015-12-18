#!/usr/bin/env python
#
# will find fmax distribution given the single cluster fits on 
# good binders and good fitters.
#
# fmax distribtuion is then enforced for weak binders, allowing
# good measurements even when no reaching saturation.
# 
# Median of cluster fits are bootstrapped to obtain errors on
# fit parameters.
#
# Sarah Denny
# July 2015

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seqfun
import datetime
import IMlibs
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import plotFun
import findFmaxDist
import fitFun
import fileFun
#plt.rc('text', usetex=True)
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-f', '--variant_file', required=True, metavar=".CPvariant",
                   help='file with bootstrapped, variant fits')
parser.add_argument('-ts', '--time_series', required=True, metavar=".CPtimeseries.pkl",
                   help='file containining the fluorescence information'
                   ' binned over time.')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')
parser.add_argument('-t', '--times', metavar=".times",
                   help='file containining the binned times')
parser.add_argument('-l', '--lib_characterization', 
                   help='library characterization filename. Use if supplying variant'
                   'sequence of variant index name')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is variant number + ".pdf"')
parser.add_argument('--annotate', action="store_true",
                   help='flag if you want the plot annotated')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-v', '--variant_numbers', nargs='+', metavar="N", type=int,
                   help='index of variant(s) to plot')


if __name__ == '__main__':
    args = parser.parse_args()
    
    variantFilename = args.variant_file
    annotatedClusterFile  = args.annotated_clusters
    bindingCurveFilename  = args.time_series


    
    # find variant
    variant_table = pd.read_table(variantFilename, index_col=0)
    if not np.any(variant_table.loc[args.variant_numbers].numTests > 0):
        print 'Error: no clusters on chip with any of these variant numbers!'
        sys.exit()

    # load data
    times = fileFun.loadFile(args.times)
    annotatedClusters = pd.read_pickle(annotatedClusterFile)
    bindingSeries = pd.read_pickle(bindingCurveFilename)
    groupedBindingSeries = pd.concat([annotatedClusters, bindingSeries], axis=1).groupby('variant_number')

    offRates = variantFun.perVariant(variant_table, groupedBindingSeries, times)  
    
    for variant in args.variant_numbers:  
        offRates.plotOffrateCurve(variant)
        
        if args.out_file is None:
            args.out_file = 'binding_curve.variant_%d.pdf'%variant
        
    plt.savefig( args.out_file)
