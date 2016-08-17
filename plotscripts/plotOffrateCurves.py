#!/usr/bin/env python
#
# Sarah Denny
# July 2015

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from fittinglibs import fileio

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
parser.add_argument('-v', '--variant_numbers', nargs='+', metavar="N", 
                   help='variant(s) or clusterId(s) to plot')


group = parser.add_argument_group()
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is variant number + ".pdf"')
parser.add_argument('--annotate', action="store_true",
                   help='flag if you want the plot annotated')
group.add_argument('-an', '--annotated_clusters', metavar="CPannot.pkl",
                   help='annotated cluster file. Supply if you wish to plot a variant.'
                   'Otherwise assumes single cluster fits.')


if __name__ == '__main__':
    args = parser.parse_args()
    variantFilename = args.variant_file
    bindingSeriesFile = args.cpseries
    timeDeltaFile = args.time_dict
    annotatedClusterFile = args.annotated_clusters
    tile_to_subset = args.tile

    if args.out_file is None:
        args.out_file = 'offrate_curve'
    
    # load data
    variant_table = fileio.loadFile(variantFilename)
    bindingSeries = fileio.loadFile(bindingSeriesFile)
    timeDict = fileio.loadFile(timeDeltaFile)

    # load annotated clusters if given, else make a data structure that assigns the variant number to the cluster ID
    if annotatedClusterFile is not None:
        annotatedClusters = fileio.loadFile(annotatedClusterFile)
    else:
        annotatedClusters = pd.DataFrame(bindingSeries.index.tolist(), index=bindingSeries.index, columns=['variant_number'])

    # initialize class of results
    offRates = results.perVariant(variant_table, annotatedClusters, bindingSeries, x=timeDict[tile_to_subset])
    
    # plot
    for variant in args.variant_numbers:
        try:
            variant = int(variant)
        except ValueError:
            pass
        
        # plot
        offRates.plotOffrateCurve(variant, annotate=args.annotate)
        
        # save
        outfile = '%s.%s.pdf'%(args.out_file, str(variant))
        plt.savefig(out_file)
