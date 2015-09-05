import datetime
import scipy.cluster.hierarchy as sch
import scipy.cluster as sc
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import IMlibs
import fitFun
sys.path.append('/home/sarah/array_image_tools_SKD/specific/')
#sys.path.append('/Users/Sarah/python/array_image_tools_SKD/specific/')
from mergeReplicates import errorPropagateAverage

#sys.path.append('/home/sarah/JunctionLibrary')
import hjh.junction
import hjh.tecto_assemble

sns.set_style("white", {'lines.linewidth': 1})

parameters = fitFun.fittingParameters()
libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.3way_mod.txt'
#libCharFile = 'all_10expts.library_characterization.txt'
libChar = pd.read_table(libCharFile)
results = pd.read_pickle('flowWC.150607_150605.combined.results.pkl')
variant_table = pd.concat([libChar, results.astype(float)], axis=1)
subtable = variant_table.loc[(libChar.sublibrary=='three_way_junctions')].copy()
subtable.loc[:, 'insertions'] = subtable.junction_seq.str.slice(2) 
subtable.loc[:, 'junction_seq']  = (subtable.flank.str.get(0) + '_' +
                                subtable.flank.str.get(1) + '_' +
                                subtable.flank.str.get(2))


loop = 'UUCG_GGAA'
for topology in ['__', 'A__', 'AA__', '_A_', '_AA_', '__A', '__AA']:

    # process variants with loop, topology by length and by sequence
    y = tectoThreeWay.findVariantsByLengthAndCircularlyPermutedSeq(subtable,
                                                                   topology=topology,
                                                                   loop=loop)
    
    # if you only want to fit a subset of the data, change 'leave_out_lengths'
    leave_out_lengths = np.array([])
    other_lengths = lengths[np.logical_not(np.in1d(lengths, leave_out_lengths))]
    yprime = {}
    for length in other_lengths: yprime[length] = y.loc[length]
    yprime = pd.concat(yprime)
    
    # fit to model with least squares
    results = tectoThreeWay.fitThreeWay(yprime)
    
    # process data and fits into one mat
    data = tectoThreeWay.findDataMat(subtable, y, results)
    
    data.to_pickle('figs_3way_other_loop_2015-09-05/all_lengths.topology%s.dat.pkl'%topology)
    results.to_pickle('figs_3way_other_loop_2015-09-05/all_lengths.topology%s.results.pkl'%topology)
    
    tectoPlots.plotScatterPlotTrainingSet(data, other_lengths)
    plt.savefig('figs_3way_other_loop_2015-09-05/dG_fit.train.topology%s.pdf'%topology)
    
    # plot the test set
    tectoPlots.plotScatterplotTestSet(data, leave_out_lengths)
    plt.savefig('figs_3way_other_loop_2015-09-05/dG_fit.test.topology%s.pdf'%topology)
    
    # plot the predicted ddG with loop change
    tectoPlots.plotScatterplotLoopChange(data)
    plt.savefig('figs_3way_other_loop_2015-09-05/ddG_loop.topology%s.pdf'%topology)
    
    # plot the fit parameters
    tectoPlots.plotBarPlotFrac(results, data)
    plt.savefig('figs_3way_other_loop_2015-09-05/fraction_in_each_permute.topology%s.pdf'%topology)
            
    # plot fit binding params
    tectoPlots.plotBarPlotdG_bind(results, lengths)
    plt.savefig('figs_3way_other_loop_2015-09-05/deltaG_bind.topology%s.pdf'%topology)
    
    # plot the nn versus dG conf
    tectoPlots.plotScatterPlotNearestNeighbor(data)
    plt.savefig('figs_3way_2015-09-04/dG_conf_vs_nn.topology%s.pdf'%topology)
