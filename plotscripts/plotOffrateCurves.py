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
from fittinglibs import fileio, processresults

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
parser.add_argument('-ts', '--tile_series',  metavar="CPtiles",
                   help='file containing tile information')
parser.add_argument('-v', '--variant_numbers', nargs='+', metavar="N", 
                   help='variant(s) or clusterId(s) to plot')


group = parser.add_argument_group()
parser.add_argument('-out', '--out_dir', default='./',
                   help='output directory. default is current directory')
parser.add_argument('--annotate', action="store_true",
                   help='flag if you want the plot annotated')
group.add_argument('-an', '--annotated_clusters', metavar="CPannot.pkl",
                   help='annotated cluster file. Supply if you wish to plot a variant.'
                   'Otherwise assumes single cluster fits.')
parser.add_argument('-n', '--numtiles',  metavar="N", type=int,
                   help='If you wish to only plot some tiles, specify number here.'
                   'default is to plot all tiles.')
parser.add_argument('-t', '--tiles',  metavar="NNN", nargs='+', 
                   help='If you wish to only plot specific tiles, specify here (i.e. 001)')

if __name__ == '__main__':
    args = parser.parse_args()
    variantFilename = args.variant_file
    bindingSeriesFile = args.cpseries
    timeDeltaFile = args.time_dict
    tileSeriesFile = args.tile_series
    annotatedClusterFile = args.annotated_clusters

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    # load data
    variant_table = fileio.loadFile(variantFilename)
    bindingSeries = fileio.loadFile(bindingSeriesFile)
    timeDict = fileio.loadFile(timeDeltaFile)
    tileSeries = fileio.loadFile(tileSeriesFile)
    
    # load annotated clusters if given, else make a data structure that assigns the variant number to the cluster ID
    if annotatedClusterFile is not None:
        annotatedClusters = fileio.loadFile(annotatedClusterFile)
    else:
        annotatedClusters = pd.DataFrame(bindingSeries.index.tolist(), index=bindingSeries.index, columns=['variant_number'])

    # initialize class of results
    offRates = processresults.perVariant(variant_table, annotatedClusters, bindingSeries, x=timeDict, tiles=tileSeries)
    
    # decide on a reasonable upperbound
    if 'pvalue' in variant_table:
        index = variant_table.pvalue < 0.01
    else:
        index = pd.Series(1, index=variant_table.index).astype(bool)
    ylim = [0, variant_table.loc[index].fmax.median() + 2*variant_table.loc[index].fmax.std()]
    
    # plot
    for variant in args.variant_numbers:
        try:
            variant = int(variant)
        except ValueError:
            pass
        
        # plot
        result = offRates.plotOffrateCurve(variant, annotate=args.annotate, numtiles=args.numtiles, tiles=args.tiles)
        if result is not None:
            plt.ylim(ylim)
            
            # save
            out_file = os.path.join(args.out_dir, 'offrate_curve.%s'%str(variant))
            if args.tiles is not None:
                out_file = out_file + '.tile_%s'%('_'.join(args.tiles))
            elif args.numtiles is not None:
                out_file = out_file + '.%d_tiles'%args.numtiles
            plt.savefig(out_file + '.pdf')
