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
import variantFun
#plt.rc('text', usetex=True)
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-f', '--variant_file', required=True, metavar=".CPvariant",
                   help='file with bootstrapped, variant fits')
parser.add_argument('-cs', '--cpseries', metavar="CPseries.pkl",
                   help='CPseries file containining the time series information')
parser.add_argument('-td', '--time_dict', metavar="timeDict.p",
                   help='file containining the timing information per tile')
parser.add_argument('--tile', metavar="NNN", default='001',
                   help='plot one this tiles median values')

parser.add_argument('-out', '--out_file', 
                   help='output filename. default is variant number + ".pdf"')
parser.add_argument('--annotate', action="store_true",
                   help='flag if you want the plot annotated')

group = parser.add_argument_group()
group.add_argument('-v', '--variant_numbers', nargs='+', metavar="N", 
                   help='index of variant(s) to plot')
group.add_argument('-an', '--annotated_clusters', metavar="CPannot.pkl",
                   help='annotated cluster file. Supply if you wish to take medians per variant.'
                   'Otherwise assumes single cluster fits.')


if __name__ == '__main__':
    args = parser.parse_args()
    
    variantFilename = args.variant_file
    bindingSeriesFile = args.cpseries
    timeDeltaFile = args.time_dict
    annotatedClusterFile = args.annotated_clusters
    tile_to_subset = args.tile
    
    # find variant
    variant_table = fileFun.loadFile(variantFilename)
    #if not np.any(variant_table.loc[args.variant_numbers].numTests > 0):
    #    print 'Error: no clusters on chip with any of these variant numbers!'
    #    sys.exit()

    # load data
    bindingSeries = fileFun.loadFile(bindingSeriesFile)
    timeDict = fileFun.loadFile(timeDeltaFile)

    if annotatedClusterFile is not None:
        annotatedClusters = fileFun.loadFile(annotatedClusterFile)
    else:
        annotatedClusters = pd.DataFrame(bindingSeries.index.tolist(), index=bindingSeries.index, columns=['variant_number'])

    offRates = variantFun.perVariant(variant_table, annotatedClusters, bindingSeries, x=timeDict[tile_to_subset])
    
    for variant in args.variant_numbers:
        try:
            variant = int(variant)
        except ValueError:
            pass
        
        offRates.plotOffrateCurve(variant, annotate=args.annotate)
        
        if args.out_file is None:
            args.out_file = 'binding_curve.variant_%s.pdf'%str(variant)
        
        plt.savefig( args.out_file)