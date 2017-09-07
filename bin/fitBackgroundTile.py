#!/usr/bin/env python
#
# after bootstrapping binding curves, this does the same procedure for
# a set of background clusters. First finds background clusters with 
# filter sets (filterNeg = background set or filterPos is opposite of
# background set). Then fits using same constraints on fmax.
#
# Sarah Denny
import os
import numpy as np
import pandas as pd
import argparse
import sys
import subprocess
import itertools

import seaborn as sns
import matplotlib.pyplot as plt

from fittinglibs import fileio
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='extract background clusters from '
                                 'CPsignal file')
parser.add_argument('-cs', '--cpseries_dir', metavar="CPseries.pkl", required=True,
                    help='directory containing the non-reduced CPseries data')
parser.add_argument('-bg', '--bg_clusters', required=True,
                    help="text file containing the names of clusters that do NOT have a RNAP init site (and aren't fiducial either)")
parser.add_argument('-an', '--annotated_clusters', metavar="CPannot.pkl", required=True,
                    help="file containing the annotated clusters with variant info")
parser.add_argument('-csr', '--cpseries_red', metavar="CPseries.pkl", required=True,
                    help="file of the RED channel CPseries (reduced) for normalization")


group = parser.add_argument_group('optional arguments for plotting')
group.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')

if __name__=="__main__":    
    args = parser.parse_args()
    
    cpseries_files = subprocess.check_output("ls {dirname}/*CPseries.pkl".format(dirname=args.cpseries_dir), shell=True).split()
    bg_clusters = fileio.loadFile(args.bg_clusters)

    # load binding series
    binding_series = pd.concat([fileio.loadFile(filename) for filename in cpseries_files])
    binding_series_sub = binding_series.loc[bg_clusters].dropna().copy()
    bg_clusters = binding_series_sub.index.tolist() # just to get rid of any NaNs
    
    # load annotations
    annotated_clusters = fileio.loadFile(args.annotated_clusters)
    num_variants = 100
    variants = np.random.choice(annotated_clusters.variant_number.unique(), size=num_variants, replace=False)
    clusters = list(itertools.chain(*[group.index.tolist() for name, group in annotated_clusters.groupby('variant_number') if name in variants]))
    
    # find subset of clusters
    cluster_sub = np.random.choice(bg_clusters, size=len(clusters), replace=False)
    
    # sub clusters file
    binding_series_sub = binding_series_sub.loc[cluster_sub].copy()
    
    # make CPannot by reindexing
    ann_clusters_sub = annotated_clusters.loc[clusters].copy()
    ann_clusters_sub.index = cluster_sub
    
    
    # save reindexed file
    if not args.out_file:
        args.out_file = os.path.join(os.path.dirname(dirname.strip('/')), os.path.splitext(os.path.basename(bg_clusters))[0])
    
    binding_series_sub.to_pickle(args.out_file + '_green.CPseries.pkl')
    ann_clusters_sub.to_pickle(args.out_file + '.CPannot.pkl')
    
    # plot
    plt.figure(figsize=(3,3))
    sns.distplot(annotated_clusters.variant_number.value_counts())
    sns.distplot(ann_clusters_sub.variant_number.value_counts())
    
    # load red channel, divide by median
    binding_series_red = fileio.loadFile(args.cpseries_red)
    binding_series_norm = binding_series_sub/binding_series_red.iloc[:, 0].median()
    binding_series_norm.to_pickle(args.out_file + '.CPseries.pkl')
