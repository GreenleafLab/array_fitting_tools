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
# Force matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
# This will prevent it from trying to show plots during code execution
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
parser.add_argument('-c', '--concentrations', metavar=".txt",
                   help='file containining the concentrations')
# parser.add_argument('-v', '--variant_numbers', nargs='+', metavar="N", 
#                    help='variant(s) or clusterId(s) to plot')
parser.add_argument('-v', '--variant_IDs', nargs='+', metavar="N", 
                   help='variant(s) or clusterId(s) to plot')


group = parser.add_argument_group()
parser.add_argument('-out', '--out_dir', default='./',
                   help='output directory. default is current directory')
parser.add_argument('--annotate', action="store_true",
                   help='flag if you want the plot annotated')
group.add_argument('-an', '--annotated_clusters', metavar="CPannot.pkl",
                   help='annotated cluster file. Supply if you wish to plot a variant.'
                   'Otherwise assumes single cluster fits.')


if __name__ == '__main__':
    args = parser.parse_args()
    variantFilename = args.variant_file
    bindingSeriesFile = args.cpseries
    annotatedClusterFile = args.annotated_clusters

    # make dir if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    # load data
    variant_table = fileio.loadFile(variantFilename)
    bindingSeries = fileio.loadFile(bindingSeriesFile)
    concentrations = fileio.loadFile(args.concentrations)

    # load annotated clusters if given, else make a data structure that assigns the variant number to the cluster ID
    if annotatedClusterFile is not None:
        annotatedClusters = fileio.loadFile(annotatedClusterFile)
    else:
        # annotatedClusters = pd.DataFrame(bindingSeries.index.tolist(), index=bindingSeries.index, columns=['variant_number'])
        annotatedClusters = pd.DataFrame(bindingSeries.index.tolist(), index=bindingSeries.index, columns=['variant_ID'])

    # initialize class of results
    affinityData = processresults.perVariant(variant_table, annotatedClusters, bindingSeries, x=concentrations)
    
    # (20170423-Ben) If no variant_IDs supplied, plot them all
    vIDs_to_plot = []
    if args.variant_IDs:
      vIDs_to_plot = args.variant_IDs
    else:
      vIDs_to_plot = variant_table.index.tolist()

    # plot
    # for variant in args.variant_numbers:
      for variant in vIDs_to_plot:  
        
        # plot
        affinityData.plotBindingCurve(variant, annotate=args.annotate)
        
        # save
        out_file = os.path.join(args.out_dir, 'binding_curve.%s.pdf'%str(variant))
        plt.savefig(out_file)
        plt.close()
