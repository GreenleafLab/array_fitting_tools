#!/usr/bin/env python
#
# make plots associated with bootStrapFits
#
# Sarah Denny
# Updated by Anthony Ho
# Aug 2016

##### IMPORT #####
import pandas as pd
import sys
import os
import argparse
import datetime
import matplotlib.pyplot as plt
from fittinglibs import fitting, plotting, fileio


### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-v', '--variant_file', required=True, metavar=".CPvariant.pkl",
                   help='file with single cluster fits')


##### SCRIPT #####
if __name__ == '__main__':
    # load files
    args = parser.parse_args()
    
    variantFile = args.variant_file
    
    # make fig directory    
    figDirectory = os.path.join(os.path.dirname(variantFile),
                                'figs_%s'%str(datetime.date.today()))
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    
    # load data
    variant_table = fileio.loadFile(variantFile)
        
    # make plots    
    plotting.plotFmaxInit(variant_table)
    plt.savefig(os.path.join(figDirectory, 'initial_Kd_vs_final.colored_by_fmax.pdf'))
    
    plotting.plotErrorInBins(variant_table, xdelta=10)
    plt.savefig(os.path.join(figDirectory, 'error_in_bins.dG.pdf'))
    
    plotting.plotNumberInBins(variant_table, xdelta=10)
    plt.savefig(os.path.join(figDirectory, 'number_in_bins.Kd.pdf'))
    
