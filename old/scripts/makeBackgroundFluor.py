#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import argparse
import sys
import seqfun
import itertools
import scipy.stats as st
import IMlibs
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF               
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import lmfit
import fitFun
from fitFun import fittingParameters
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='extract background clusters from '
                                 'CPsignal file')
parser.add_argument('-cs', '--cpsignal', metavar="CPsignal.pkl", nargs='+', required=True,
                    help='list of reduced CPsignal files.')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')

parser.add_argument('-fn','--filterNeg',  nargs='+', help='set of filters '
                     'that designate "background" clusters. If not set, assume '
                     'complement to filterPos')
parser.add_argument('-fp','--filterPos', nargs='+', help='set of filters '
                    'that designate not background clusters.') 
parser.add_argument('-bp', '--binding_point', type=int, default=-1,
                    help='point in binding series to use for null scores (Default is '
                    'last concentration. -2 for second to last concentration)', )

def loadNullScores(backgroundTileFile, filterPos=None, filterNeg=None,
                   binding_point=None, return_binding_series=None,
                   concentrations=None):
    # find one CPsignal file before reduction. Find subset of rows that don't
    # contain filterSet
    if return_binding_series is None:
        return_binding_series = False
    if binding_point is None:
        binding_point = -1
    if filterPos is None:
        if filterNeg is None:
            print "Error: need to define either filterPos or filterNeg"
            return

    # Now load the file, specifically the signal specifed
    # by index.
    table = IMlibs.loadCPseqSignal(backgroundTileFile)
    table.dropna(subset=['filter'], axis=0, inplace=True)
    
    # if filterNeg is given, use only this to find background clusters
    # otherwise, use clusters that don't have any of filterPos
    if filterNeg is None:
        # if any of the positive filters are found, don't use these clusters
        subset = np.logical_not(np.asarray([[str(s).find(filterSet) > -1
                                             for s in table.loc[:, 'filter']]
            for filterSet in filterPos]).any(axis=0))
    else:
        if filterPos is None:
            # if any of the negative filters are found, use these clusters
            subset = np.asarray([[str(s).find(filterSet) > -1
                                  for s in table.loc[:, 'filter']]
                for filterSet in filterNeg]).any(axis=0)
        else:
            # if any of the negative filters are found and none of positive filters, use these clusters
            subset = np.asarray([[str(s).find(negOne) > -1 and not(str(s).find(posOne) > -1)
                                  for s in table.loc[:, 'filter']]
                for negOne, posOne in itertools.product(filterNeg, filterPos)]).any(axis=0)            

    binding_series = pd.DataFrame([s.split(',') for s in table.loc[subset].binding_series],
        dtype=float, index=table.loc[subset].index)
    
    if return_binding_series:
        return binding_series.dropna(axis=0)
    else: 
        return binding_series.iloc[:, binding_point].dropna()


if __name__=="__main__":    
    args = parser.parse_args()
    
    backgroundTileFile = args.cpsignal
    outFile            = args.out_file
    filterPos = args.filterPos
    filterNeg = args.filterNeg
    binding_point = args.binding_point

    if outFile is None:
        
        # make bindingCurve file 
        outFile = os.path.splitext(
                backgroundTileFile[:backgroundTileFile.find('.pkl')])[0]
        backgroundFilename = outFile + '.fluor.npy'

        
    fluorescence = loadNullScores(backgroundTileFile,
                                            filterPos=filterPos,
                                            filterNeg=filterNeg,
                                            binding_point=binding_point,)
    np.save(backgroundFilename, fluorescence)
    