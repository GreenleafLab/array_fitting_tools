#!/usr/bin/env python
""" Evaluate false discovery rate after fitting background tiles.

Extracts data from merged, CPsignal file.
Normalizes by all cluster signal if this was given.
Fits all single clusters.

Input:
directory of background CPvariant files.
Or a single background CPvariant file.

Output:
table of values of false discovery rates for different cutoffs
pdf of histogram

Sarah Denny

"""

import os
import numpy as np
import pandas as pd
import argparse
import sys
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

import seqfun

sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import plotFun
import fitFun

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-d', '--directory', metavar="dir",
                   help='diretory name to find CPvariant files')
parser.add_argument('-f', '--cpvariant_file', metavar=".CPvariant",
                   help='file with clusters annotated by variant number')
parser.add_argument('-c', '--cutoff', metavar="[flow]", type=float,
                    help='if given, just print false discovery rate at this cutoff')
parser.add_argument('-n', '--min_num_clusters', metavar="N", type=int, default=5,
                    help='minimum number of measurements. defualt=5')
parser.add_argument('-dG', '--report_dG', action="store_true", default=False,
                    help='flag if the cutoff is in kcal/mol rather than concentration')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')

def getClosestIndex(vec, cutoff):
    return np.argmin(np.abs(vec-cutoff))


if __name__ == '__main__':
    # load files
    args = parser.parse_args()
        
    if args.directory is None and args.cpvariant_file is None:
        print ("Error: must define either directory containing background "
               "CPvariant files (-d) or a particular CPvariant file (-f)")
    
    if args.directory is not None:
        files = subprocess.check_output('find %s -name "*CPvariant"'%args.directory,
                                        shell=True).split()
        background_variants = pd.concat([pd.read_table(file, index_col=0) for file in files],
            ignore_index=True)
    else:
        background_variants = pd.read_table(args.cpvariant_file, index_col=0)
        args.directory = os.path.dirname(args.cpvariant_file)
    
    # define out file    
    if args.out_file is None:
        args.out_file = os.path.join(args.directory, 'background_cdfs.txt')
    
    # use only those variants with at least five measurements
    background_variants = (background_variants.
                           loc[background_variants.numClusters>=args.min_num_clusters])
    
    
    # find cdf of dGs
    parameters = fitFun.fittingParameters()
    if args.report_dG:
        x, y = seqfun.getCDF(background_variants.dG)
        text = 'kcal/mol'
    else:
        x, y = seqfun.getCDF(parameters.find_Kd_from_dG(background_variants.dG))
        text = 'nM'
    # if cutoff is given, just report that and print
    if args.cutoff is not None:
        cumulative_fraction = y[getClosestIndex(x, args.cutoff)]
        
        print ('cutoff %4.1f %s represents %4.3f %% of background distribution'
               %(args.cutoff, text, 100*cumulative_fraction))
        
        sys.exit()
    
    # plot distribution
    plotFun.histogramKds(background_variants)
    plt.savefig(args.out_file + '.pdf')
    
    # find cdf at various cutoffs
    if args.report_dG:
        cutoffs = np.linspace(-12, -5)
    else:
        cutoffs = np.logspace(0, 6)
    
    cumulative_fractions = pd.DataFrame(
        np.column_stack([cutoffs,
                         [y[getClosestIndex(x, cutoff)] for cutoff in cutoffs]]),
        columns = ['cutoff (%s)'%text, 'cumulative fraction'])
    
    # save
    cumulative_fractions.to_csv(args.out_file, sep='\t', index=False)
    
    
        