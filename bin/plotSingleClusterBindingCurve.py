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
#plt.rc('text', usetex=True)
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-b', '--binding_curves', required=True, metavar='.CPseries.pkl',
                   help='file containining the binding curve information')
parser.add_argument('-f', '--single_cluster_fits', required=True, metavar='CPfitted.pkl',
                   help='file containining the single cluster fits')
parser.add_argument('-c', '--concentrations', required=True, metavar='concentrations.txt',
                    help='text file giving the associated concentrations')
parser.add_argument('-i', '--cluster', required=False, metavar='clusterID', action='append',
                    help='individual clusterIDs to plot')
parser.add_argument('-if', '--cluster_file', required=False, metavar='clusterID file', help='text file containing clusterIDs to plot, one-per-line')
parser.add_argument('-out', '--out_dir', default='plots',
                   help='output directory. default is "plots"')


if __name__ == '__main__':
    args = parser.parse_args()
    
    if not len(sys.argv) > 1: # Print help if this script is called with no arguments (equivalent to -h)
        parser.print_help()
    
    bindingCurveFilename  = args.binding_curves
    singleClusterFilename = args.single_cluster_fits
    
    # Read in cluster file
    clusters = []
    if args.cluster_file is not None:
        with open(args.cluster_file) as f:
            clusters = f.readlines()
            clusters = [line.strip() for line in clusters] # Strip trailing newline characters and whitespace
    
    # Add any individual clusterIDs that were passed in with -i
    if args.cluster is not None:
        clusters += args.cluster

    if len(clusters) <= 0:
        sys.exit('ERROR: No clusterIDs were passed in. Use -i or -if to indicate which clusterIDs should be plotted. Exiting...')

    # load data
    concentrations = np.loadtxt(args.concentrations)
    bindingSeries =  pd.read_pickle(bindingCurveFilename)
    fittedSingles = pd.read_pickle(singleClusterFilename)

    # Prepare the output directory
    try:
        os.makedirs(args.out_dir)
    except OSError:
        if not os.path.isdir(args.out_dir):
            raise

    # Iterate through each ClusterID    
    for currClusterID in clusters:

        # plot
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        fitParameters = pd.DataFrame(columns=['fmax', 'dG', 'fmin'])
        plotFun.plotFitCurve(concentrations,
                            bindingSeries.loc[currClusterID],
                            fittedSingles.loc[currClusterID],
                            fitParameters, ax=ax)
    
        currOutputFilename = os.path.join(args.out_dir, 'cluster_%s.binding_curve.pdf'%currClusterID)
        plt.savefig(currOutputFilename)
