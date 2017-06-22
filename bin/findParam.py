#!/usr/bin/env python
""" Will take a pickled file and find the distribution of a parameter.

Sarah Denny

"""

import os
import numpy as np
import pandas as pd
import argparse
import sys
import itertools
import scipy.stats as st
import matplotlib.pyplot as plt

import seaborn as sns
from fittinglibs import (processing, fileio)
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='fit single clusters to binding curve')

group = parser.add_argument_group('required arguments')
group.add_argument('-f', '--fitting_results', required=True,
                    help='CPvariant or CPfitted file.')
group.add_argument('--param', required=True,
                   help='parameter to find median/distribution ')

group = parser.add_argument_group('optional arguments')
group.add_argument('--filter', 
                   help='name of function within module "processing" that will filter the table')
group.add_argument('--other_params', nargs='*', 
                   help='name of other params to plot against param')
group.add_argument('-out', '--out', 
                   help='output filename. default is "[param].pdf" in same directory as fitted vals.')

if __name__ == '__main__':
    args = parser.parse_args()

    # load fitted vals
    fitted_vals = fileio.loadFile(args.fitting_results)
    
    # apply fitting function if provided
    if args.filter:
        filter_func = getattr(processing, args.filter)
        fitted_vals = filter_func(fitted_vals)
    else:
        args.filter = 'no_filter'
    # find the median value of param and plot distribution
    vec = fitted_vals.loc[:, args.param]
    print ('{param}\n-------\nmedian={median}\nmean={mean}\nstd.dev={std}').format(param=args.param, median=vec.median(), mean=vec.mean(), std=vec.std())
    
    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.fitting_results), '%s_%s'%(args.filter, args.param))
    plt.figure(figsize=(3,3))
    sns.distplot(vec)
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.out + '.pdf')
    
    if args.other_params:
        for param in args.other_params:
            plt.figure(figsize=(3,3))
            plt.hexbin(fitted_vals.loc[:, param], fitted_vals.loc[:, args.param], mincnt=1, cmap='copper', bins='log')
            plt.xlabel(param)
            plt.ylabel(args.param)
            plt.tight_layout()
            plt.savefig(args.out + '_vs_%s.pdf'%param)        