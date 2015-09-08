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
sys.path.append('/Users/Sarah/python/array_image_tools_SKD/libs/')
#sys.path.append('/home/sarah/array_image_tools_SKD/libs/')

import tectoThreeWay
import tectoPlots
import hjh.junction
import hjh.tecto_assemble

sns.set_style("white", {'lines.linewidth': 1})

parameters = fitFun.fittingParameters()
#libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.3way_mod.txt'
libCharFile = '../all_10expts.library_characterization.txt'
resultsFile = '../flowWC.150607_150605.combined.results.pkl'
libChar = pd.read_table(libCharFile)
results = pd.read_pickle(resultsFile)
variant_table = pd.concat([libChar, results.astype(float)], axis=1)
subtable = variant_table.loc[(libChar.sublibrary=='three_way_junctions')].copy()
subtable.loc[:, 'insertions'] = subtable.junction_seq.str.slice(2) 
subtable.loc[:, 'junction_seq']  = (subtable.flank.str.get(2) + '_' +
                                subtable.flank.str.get(1) + '_' +
                                [tectoThreeWay.complements[s] for s in subtable.flank.str.get(0)])

figDirectory = 'figs_3way_'+str(datetime.date.today())
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)



# process variants with loop, topology by length and by sequence
y = tectoThreeWay.findVariantsByLengthAndCircularlyPermutedSeq(subtable)
lengths = np.array(y.index.levels[0].tolist())

# only fit the subset of the data in 'to_include'
to_include = {'loop':['GGAA_UUCG'],
              'length':[3,5,6],
              'topology':['AA__', 'A__', '_AA_', '_A_', '__', '__A', '__AA']}

# fit to model with least squares
results = tectoThreeWay.fitThreeWay(y, to_include=to_include)

# process data and fits into one mat
data = tectoThreeWay.findDataMat(subtable, yprime, results)

data.to_pickle(os.path.join(figDirectory, 'lengths_all.loop_%s.topology_all.dat.pkl'%(loop)))
results.to_pickle(os.path.join(figDirectory, 'lengths_all.loop_%s.topology_all.results.pkl'%(loop)))

# plot training set
loop = 'GGAA_UUCG'
for topology in ['AA__', 'A__', '_AA_', '_A_', '__', '__A', '__AA']:

    subdata = tectoThreeWay.subsetData(data, {'topology':[topology],
                                              'loop':[loop]})
    tectoPlots.plotScatterPlotTrainingSet(subdata)
    plt.savefig(os.path.join(figDirectory, 'dG_fit.train.loop_%s.topology%s.pdf'%(loop, topology)))
    
    # plot the test set
    if len(leave_out_lengths) > 0:
        tectoPlots.plotScatterplotTestSet(subdata, leave_out_lengths)
        plt.savefig(os.path.join(figDirectory, 'dG_fit.test.loop_%s.topology%s.pdf'%(loop, topology)))
        
    # plot the predicted ddG with loop change
    tectoPlots.plotScatterplotLoopChange(subdata)
    plt.savefig(os.path.join(figDirectory, 'ddG_loop.loop_%s.topology%s.pdf'%(loop, topology)))


for topology in ['AA__', 'A__', '_AA_', '_A_', '__', '__A', '__AA']:
    # plot the fit parameters
    subdata = tectoThreeWay.subsetData(data, {'topology':topology,
                                              'loop':loop})
    tectoPlots.plotBarPlotFrac(results, subdata)
    plt.savefig(os.path.join(figDirectory, 'fraction_in_each_permute.loop_%s.topology%s.pdf'%(loop, topology)))
            
    # plot fit binding params
    tectoPlots.plotBarPlotdG_bind(results, subdata)
    plt.savefig(os.path.join(figDirectory, 'deltaG_bind.loop_%s.topology%s.pdf'%(loop, topology)))
    
    # plot the nn versus dG conf
    tectoPlots.plotScatterPlotNearestNeighbor(subdata)
    plt.savefig(os.path.join(figDirectory, 'dG_conf_vs_nn.loop_%s.topology%s.pdf'%(loop, topology)))


# plot some things

loop = 'UUCG_GGAA'
for topology in ['__', 'A__', 'AA__', '_A_', '_AA_', '__A', '__AA']:
    data = pd.read_pickle(os.path.join(figDirectory, 'all_lengths.loop_%s.topology%s.dat.pkl'%(loop, topology)))
    results = pd.read_pickle(os.path.join(figDirectory, 'all_lengths.loop_%s.topology%s.results.pkl'%(loop, topology)))
    tectoPlots.plotBarPlotFrac(results, data)
    plt.savefig(os.path.join(figDirectory, 'fraction_in_each_permute.loop_%s.topology%s.pdf'%(loop, topology)))
        