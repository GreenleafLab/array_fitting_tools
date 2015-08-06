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
import seqfun
import itertools
import datetime
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
import plotFun
import bootStrapFits
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='extract background clusters from '
                                 'CPsignal file')
parser.add_argument('-csb', '--cpsignal_background', metavar="CPsignal",
                    required=True,
                    help='CPsignal file with all clusters')
parser.add_argument('-cs', '--cpsignal', metavar="CPsignal.pkl", required=True,
                    help='reduced CPsignal file of good clusters')
parser.add_argument('-t', '--single_cluster_fits', required=True,
                   help='file with single cluster fits of good clusters')
parser.add_argument('-a', '--annotated_clusters', required=True,
                   help='file with clusters annotated by variant number')
parser.add_argument('-c', '--concentrations', metavar="concentrations.txt", required=True,
                    help='text file giving the associated concentrations')

parser.add_argument('-out', '--out_file', 
                   help='output filename. default is basename of input filename')

parser.add_argument('-fn','--filterNeg',  nargs='+', help='set of filters '
                     'that designate "background" clusters. If not set, assume '
                     'complement to filterPos')
parser.add_argument('-fp','--filterPos', nargs='+', help='set of filters '
                    'that designate not background clusters.') 


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
    
    backgroundTileFile     = args.cpsignal_background
    pickleCPsignalFilename = args.cpsignal
    fittedBindingFilename  = args.single_cluster_fits
    annotatedClusterFile   = args.annotated_clusters
    outFile   = args.out_file
    filterPos = args.filterPos
    filterNeg = args.filterNeg
    concentrations = np.loadtxt(args.concentrations)


    if outFile is None:
        # make bindingCurve file 
        outFile = os.path.splitext(
                backgroundTileFile[:backgroundTileFile.find('.pkl')])[0]
    backgroundFilename = outFile + '.bindingSeries.pkl'

    print '\tLoading binding series and all RNA signal:'
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(
        pickleCPsignalFilename, concentrations=concentrations)
    
    # make normalized binding series
    allClusterSignal = IMlibs.boundFluorescence(allClusterSignal, plot=True)   

    bindingSeriesBackground = loadNullScores(backgroundTileFile,
                                        filterPos=filterPos,
                                        filterNeg=filterNeg,
                                        return_binding_series=True,)
    
    
    annotations = pd.read_pickle(annotatedClusterFile).sort('variant_number')
    
    # get rid of clusters that would be NaN
    annotations = (pd.concat([annotations,
                             bindingSeries], axis=1).dropna(axis=0, thresh=5).
                   loc[:, 'variant_number']).dropna().copy()
    annotations.sort('variant_number', inplace=True)
    
    # choose subset of clusters
    numClusters = len(bindingSeriesBackground)
    numVariants = int(numClusters/annotations.value_counts().median())
    variants = np.random.permutation(np.unique(annotations))[:numVariants]
    index = np.in1d(annotations, variants)

    # assign cluster ids from annotations file to background binding series
    bindingSeriesNorm = pd.DataFrame(index  =annotations.loc[index].index,
                                     columns=bindingSeriesBackground.columns)
    bindingSeriesNorm.iloc[:numClusters] = np.random.permutation(
        np.divide(bindingSeriesBackground, allClusterSignal.median()))
    
    bindingSeriesNorm.to_pickle(backgroundFilename)
    
    # now fit
    variant_table = bootStrapFits.fitBindingCurves(fittedBindingFilename, annotatedClusterFile,
                     backgroundFilename, concentrations,
                     numCores=20, n_samples=None, variants=variants,
                     use_initial=False)
    
    # remove all irrelevant data
    variant_table.dropna(axis=0, inplace=True)
    variant_table.loc[:, 'fmax_init':'fitFraction'] = np.nan
    variant_table.to_csv(outFile + '.CPvariant', sep='\t', index=True)
    
    
    plotFun.histogramKds(variant_table.loc[variant_table.numClusters>=5])
    figDirectory = os.path.join(os.path.dirname(annotatedClusterFile),
                                'figs_%s'%str(datetime.date.today()))
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)
    plt.savefig(os.path.join(figDirectory, os.path.basename(outFile)+'.histogram_kds.pdf'))
    
