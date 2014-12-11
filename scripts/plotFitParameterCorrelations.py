"""
Given that we are imposing constraints on fmax, figure out how stable
the resulting dG is for some subset of clusters to changes in fmax

"""
import sys
import os
import time
import re
import argparse
import subprocess
import multiprocessing
import shutil
import uuid
import numpy as np
import scipy.io as sio
import CPlibs
import IMlibs
import fittingParameters

# may want to make sure path is right here
parameters = fittingParameters.Parameters()

def plotHistogram(table, parameter, subset_above, subset_below, xbins=None):
    fig = plt.figure(figsize=(7,4.5))
    histogram.compare([table[parameter].iloc[subset_below], table[parameter].iloc[subset_above]],
        xbins = xbins, labels=['below cutoff', 'above cutoff'], cmap='Paired', normalize=False)
    ax = plt.gca()
    ax.set_xlabel(parameter)
    ax.set_ylabel('number')
    plt.subplots_adjust(bottom=0.15, right=0.6, left=0.15)
    return ax

def plotScatterplot(table, p1_name, p2_name, c_name, subset=None, vmin=None, vmax=None):
    if subset is None:
        subset = np.arange(len(table))
    fig = plt.figure(figsize=(5.5,5))
    ax = fig.add_subplot(111)
    im = ax.scatter(table[p1_name].iloc[subset], table[p2_name].iloc[subset], alpha=0.1, c=table[c_name].iloc[subset], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label(c_name)
    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.grid()
    plt.tight_layout()
    return ax
    
    

# load fitted dGs to be able to get a somewhat reresentative subset
fittedBindingFilename = 'binding_curves_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPfitted'
table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
indx_subset = np.array(np.argsort(table['dG'])[np.arange(0, len(table), 100)])
table_reduced = table.iloc[indx_subset]

# load null scores
signalNamesByTileDict = {'003': 'binding_curves_rigid/AAYFY_ALL_tile003_Bottom_filtered.CPsignal'}
filterSet = 'tecto'
null_scores = IMlibs.loadNullScores(signalNamesByTileDict, filterSet)

# load
numConcentrations = 8
bindingSeries = np.transpose([np.array(table_reduced[i]) for i in range(numConcentrations)])
allClusterSignal = np.array(table_reduced['all_cluster_signal'])

# now split into parts


annotatedSignalFilename = 'binding_curves_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.CPsignal'
bindingSeriesSplit = np.array_split(bindingSeries, numCores)
allClusterSignalSplit = np.array_split(allClusterSignal, numCores)
bindingSeriesFilenameParts = IMlibs.getBindingSeriesFilenameParts(annotatedSignalFilename, numCores)
fitParametersFilenameParts = IMlibs.getfitParametersFilenameParts(bindingSeriesFilenameParts)
for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
    sio.savemat(bindingSeriesFilename, {'concentrations':concentrations,
                                        'binding_curves': bindingSeriesSplit[i].astype(float),
                                        'all_cluster':allClusterSignalSplit[i].astype(float),
                                        'null_scores':null_scores,
                                        })

# fit
threshold = 5 # %
fmax_upperbound = np.max(table[7])*10
fmax_lowerbound = np.percentile(null_scores, 100-threshold)
fmax_initial = np.nan # define this per cluster

fmin_lowerbound = 0
fmin_upperbound = np.nan # define this per cluster
fmin_initial = 0

dG_upperbound = -4 # delta G of -4 kcal/mol corresponds to fraction bound of 0.0019 at 2000nM
dG_lowerbound = -16 # delta G of -16 kcal/mol corresponds to fraction bound of 0.999 at 0.91nM
dG_initial = -7.6 # delta G of -7.6 kcal/mol corresponds to Kd of 2000nM
    
for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    workerPool.apply_async(IMlibs.findKds, args=(bindingSeriesFilename, fitParametersFilenameParts[i],
                                                 fmax_lowerbound, fmax_upperbound, fmax_initial,
                                                 dG_lowerbound, dG_upperbound, dG_initial,
                                                 fmin_lowerbound, fmin_upperbound, fmin_initial),
                           )
workerPool.close()
workerPool.join()
# save results
fitParameters = IMlibs.joinTogetherFitParts(fitParametersFilenameParts)
#for name in fitParameters: table_reduced[name] = fitParameters[name]

#### plot histograms ####
dirname = 'binding_curves_wc'
num_clusters = len(fitParameters)
subset_below = np.arange(num_clusters)[np.array(fitParameters['qvalue'] < 0.05)]
subset_above = np.arange(num_clusters)[np.array(fitParameters['qvalue'] > 0.05)]

# delta G
parameter = 'dG'
numbins = 50.0
binsize = (dG_upperbound - dG_lowerbound)/numbins
xbins = np.arange(dG_lowerbound, dG_upperbound+2*binsize, binsize)-binsize/2
ax = plotHistogram(fitParameters, parameter, subset_above, subset_below, xbins=xbins)
ax.set_xlim((dG_lowerbound, dG_upperbound))
plt.savefig('%s/deltaG.histogram.png'%(dirname))

# fmax
parameter = 'fmax'
numbins = 50.0
binsize = (fmax_upperbound - fmax_lowerbound)/numbins
xbins = np.arange(fmax_lowerbound, fmax_upperbound+binsize*2, binsize)-binsize/2
ax = plotHistogram(fitParameters, parameter, subset_above, subset_below, xbins=xbins)
ax.set_xlim((fmax_lowerbound, fmax_upperbound))
plt.xticks(rotation=70)
plt.subplots_adjust(bottom=0.2)
plt.savefig('%s/%s.histogram.png'%(dirname, parameter))

# fmax
parameter = 'fmax'
numbins = 50.0
fmax_upperbound_zoom = 2000
binsize = (upperbound_zoom - fmax_lowerbound)/numbins
xbins = np.arange(fmax_lowerbound, upperbound_zoom+binsize*2, binsize)-binsize/2
ax = plotHistogram(fitParameters, parameter, subset_above, subset_below, xbins=xbins)
ax.set_xlim((fmax_lowerbound, upperbound_zoom))
plt.xticks(rotation=70)
plt.subplots_adjust(bottom=0.2)
plt.savefig('%s/%s.zoom.histogram.png'%(dirname, parameter))

# fmin
parameter = 'fmin'
binsize = (np.max(fitParameters['fmin']) - fmin_lowerbound)/numbins
xbins = np.arange(fmin_lowerbound, np.max(fitParameters['fmin'])+binsize*2, binsize)-binsize/2
ax = plotHistogram(fitParameters, parameter, subset_above, subset_below, xbins=xbins)
plt.xticks(rotation=70)
plt.ylim((0, 3000))
plt.subplots_adjust(bottom=0.2)
plt.savefig('%s/%s.histogram.png'%(dirname, parameter))

# fmin
parameter = 'fmin'
fmin_upperbound_zoom = 1000
binsize = (upperbound_zoom - fmin_lowerbound)/numbins
xbins = np.arange(fmin_lowerbound, upperbound_zoom+binsize*2, binsize)-binsize/2
ax = plotHistogram(fitParameters, parameter, subset_above, subset_below, xbins=xbins)
plt.xticks(rotation=70)
plt.ylim((0, 3000))
plt.subplots_adjust(bottom=0.2)
plt.savefig('%s/%s.zoom.histogram.png'%(dirname, parameter))

#### plot correlations ####

# delta G versus fmax
subset = np.arange(num_clusters)[np.array(fitParameters['qvalue'] < 0.05)]
p1_name = 'dG'
p2_name = 'fmax'
c_name = 'rsq'
ax.set_xlim((dG_lowerbound, dG_upperbound))
ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, vmin=0, vmax=1)
plt.savefig('%s/%s_vs_%s.color%s.all.histogram.png'%(dirname, p1_name, p2_name, c_name))
ax.set_ylim((0, fmax_upperbound_zoom))

plt.savefig('%s/%s_vs_%s.color%s.all.zoom.histogram.png'%(dirname, p1_name, p2_name, c_name))

ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, subset, vmin=0, vmax=1)
ax.set_xlim((dG_lowerbound, dG_upperbound))
plt.savefig('%s/%s_vs_%s.color%s.below_qvalue.histogram.png'%(dirname, p1_name, p2_name, c_name))
ax.set_ylim((0, fmax_upperbound_zoom))
plt.savefig('%s/%s_vs_%s.color%s.below_qvalue.zoom.histogram.png'%(dirname, p1_name, p2_name, c_name))

# delta G versus fmin
subset = np.arange(num_clusters)[np.array(fitParameters['qvalue'] < 0.05)]
p1_name = 'dG'
p2_name = 'fmin'
c_name = 'rsq'
ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, vmin=0, vmax=1)
plt.savefig('%s/%s_vs_%s.color%s.all.histogram.png'%(dirname, p1_name, p2_name, c_name))
ax.set_xlim((-14, -2))
#plt.savefig('%s/%s_vs_%s.color%s.all.zoom.histogram.png'%(dirname, p1_name, p2_name, c_name))

ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, subset, vmin=0, vmax=1)
plt.savefig('%s/%s_vs_%s.color%s.below_qvalue.histogram.png'%(dirname, p1_name, p2_name, c_name))
ax.set_xlim((-14, -2))
#plt.savefig('%s/%s_vs_%s.color%s.below_qvalue.zoom.histogram.png'%(dirname, p1_name, p2_name, c_name))

# fmin vs fmax
subset = np.arange(num_clusters)[np.array(fitParameters['qvalue'] < 0.05)]
p1_name = 'fmax'
p2_name = 'fmin'
c_name = 'rsq'
ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, vmin=0, vmax=1)
plt.savefig('%s/%s_vs_%s.color%s.all.histogram.png'%(dirname, p1_name, p2_name, c_name))
ax.set_ylim((-0.01, 0.11))
ax.set_xlim((0, 2))
plt.savefig('%s/%s_vs_%s.color%s.all.zoom.histogram.png'%(dirname, p1_name, p2_name, c_name))

ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, subset, vmin=0, vmax=1)
plt.savefig('%s/%s_vs_%s.color%s.below_qvalue.histogram.png'%(dirname, p1_name, p2_name, c_name))
ax.set_ylim((-0.01, 0.11))
ax.set_xlim((0, 2))
plt.savefig('%s/%s_vs_%s.color%s.below_qvalue.zoom.histogram.png'%(dirname, p1_name, p2_name, c_name))