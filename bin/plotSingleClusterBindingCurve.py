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
parser.add_argument('-b', '--binding_curves', required=True, metavar=".CPseries.pkl",
                   help='file containining the binding curve information')
parser.add_argument('-f', '--single_cluster_fits', required=True, metavar=".CPfitted.pkl",
                   help='file containining the single cluster fits')
parser.add_argument('-c', '--concentrations', required=True, metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
parser.add_argument('-i', '--cluster', required=True, metavar="clusterID",
                    help='cluster Id for which to plot')

parser.add_argument('-out', '--out_file', 
                   help='output filename. default is "cluster_X.binding_curve.pdf"')



if __name__ == '__main__':
    args = parser.parse_args()
    
    annotatedClusterFile  = args.annotated_clusters
    bindingCurveFilename  = args.binding_curves
    singleClusterFilename = args.single_cluster_fits
    cluster = args.cluster
            
    # load data
    concentrations = np.loadtxt(args.concentrations)
    bindingSeries =  pd.read_pickle(bindingCurveFilename)
    fittedSingles = pd.read_pickle(singleClusterFilename)

    # plot
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    fitParameters = pd.DataFrame(columns=['fmax', 'dG', 'fmin'])
    fitFun.plotFitCurve(concentrations,
                        bindingSeries.loc[cluster],
                        fittedSingles.loc[cluster],
                        fitParameters, ax=ax)

    if args.out_file is None:
        args.out_file = 'cluster_%s.binding_curve.pdf'%cluster
    plt.savefig( args.out_file)
