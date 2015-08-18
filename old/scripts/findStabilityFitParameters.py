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
import scipy.stats as st
import CPlibs
import IMlibs
import fittingParameters
from matplotlib import rc
import matplotlib.pyplot as plt

#### FUNCTIONS ####
def fit_one(fitParametersFilenameParts, bindingSeriesFilename,initialFitParameters,scale_factor):
    workerPool = multiprocessing.Pool(processes=numCores) #create a multiprocessing pool that uses at most the specified number of cores
    for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
        result = workerPool.apply_async(IMlibs.findKds, args=(bindingSeriesFilename, fitParametersFilenameParts[i],
                                                     initialFitParameters['fmax']['lowerbound'], initialFitParameters['fmax']['upperbound'], initialFitParameters['fmax']['initial'],
                                                     initialFitParameters['dG']['lowerbound'],   initialFitParameters['dG']['upperbound'],   initialFitParameters['dG']['initial'],
                                                     initialFitParameters['fmin']['lowerbound'], initialFitParameters['fmin']['upperbound'], initialFitParameters['fmin']['initial'],
                                                     scale_factor,)
                               )
    workerPool.close()
    workerPool.join()
    fitParameters = IMlibs.joinTogetherFitParts(fitParametersFilenameParts)
    return fitParameters

def find_resulting_dGs(fitParametersFilenameParts, bindingSeriesFilename, initialFitParameters, variation_type, varying_params, scale_factor, vary_scale_factor=None):
    if vary_scale_factor is None:
        vary_scale_factor = False
    # fit with different fmaxes
    num_ubs = len(varying_params)
    resulting_dGs = ['']*num_ubs
    resulting_var_dGs = ['']*num_ubs
    for j, param in enumerate(varying_params):
        print "iterating different %s"%(' '.join(variation_type))
        fitParameters_new = initialFitParameters.copy()
        if not vary_scale_factor:   # i.e. if you want to change anything described by 'variation type'
            fitParameters_new[variation_type[0]][variation_type[1]] = param
        else: scale_factor = param # i.e. if you want to vary the scale factor
        fitParameters = fit_one(fitParametersFilenameParts, bindingSeriesFilename,fitParameters_new,scale_factor)
        print "Working on iteration %d out of %d"%(j, num_ubs)
        resulting_dGs[j] = fitParameters['dG']
        resulting_var_dGs[j] = fitParameters['dG_var']
    return np.array(resulting_dGs), np.array(resulting_var_dGs)

def plotErrorBarGraph(params, resulting_dGs, resulting_var_dGs, variation_type, subset=None, indx_0=None, logscale=None, numtoplot=None, color=None):
    if subset is None:
        subset = np.arange(len(resulting_dGs)) # define subset to plot
    if indx_0 is None:
        indx_0 = 0 # the index of the deltaG array to compare the others to
    if logscale is None:
        logscale = False # whether to cahnge the xaxis scale to log
    if numtoplot is None:
        numtoplot = 200 # default is to plot 200 traces
    if color is None:
        color = 'b'
        
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    for i in subset[np.linspace(0, len(subset)-1, numtoplot).astype(int)]:
        ax.errorbar(params, resulting_dGs[:,i]-resulting_dGs[indx_0,i], yerr=resulting_var_dGs[:,i], fmt='-o', color = color, ecolor='k', alpha = 0.1)
    ax.set_ylim((-10, 10))
    if logscale:
        ax.set_xscale('log')
    else:
        binsize = params[1] - params[0]
        ax.set_xlim((np.nanmin(params)-binsize, np.nanmax(params)+binsize))
    ax.set_xlabel('%s %s'%(variation_type[0], variation_type[1]))
    ax.set_ylabel('ddG')
    plt.tight_layout()    
    return ax

def plotFracAffectedGraph(params, resulting_dGs, resulting_var_dGs, subsets, colors, labels, logscale=None, indx_0=None):
    if indx_0 is None:
        indx_0 = 0 # the index of the deltaG array to compare the others to
    if logscale is None:
        logscale = False # whether to cahnge the xaxis scale to log
    
    # find matrix
    matrix = np.zeros(resulting_dGs.shape, dtype=bool)
    for i in range(0, matrix.shape[0]):
        # for each lower bound, is it different from previous?
        matrix[i] = np.abs(resulting_dGs[i] - resulting_dGs[indx_0]) > resulting_var_dGs[i]
        # plot
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    for i in range(len(subsets)):
        ax.plot(params, np.sum(matrix[:, subsets[i]], axis=1)/float(len(subsets[i])), colors[i]+'o-', label=labels[i])
    if logscale:
        ax.set_xscale('log')
    else:
        binsize = params[1] - params[0]
        ax.set_xlim((np.nanmin(params)-binsize, np.nanmax(params)+binsize))
    ax.set_xlabel('%s %s'%(variation_type[0], variation_type[1]))
    ax.set_ylabel('fraction that significantly differ')
    ax.set_ylim((0, 1))
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    return ax

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
    
if __name__ == '__main__':
    
    #### SCRIPT #####
    # load fitted dGs to be able to get a somewhat reresentative subset
    fittedBindingFilename = 'binding_curves_rigid_tile456/reduced_signals/barcode_mapping/binding_curves_rigid_filtered_tecto_sorted.annotated.CPfitted'
    table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
    indx_subset = np.array(np.argsort(table['dG'])[np.arange(0, len(table), 100)])
    table_reduced = table.iloc[indx_subset]
    
    # load null scores
    signalNamesByTileDict = {'003': 'binding_curves_rigid/AAYFY_ALL_tile003_Bottom_filtered.CPsignal'}
    filterSet = 'tecto'
    null_scores = IMlibs.loadNullScores(signalNamesByTileDict, filterSet)
    
    # Load binding series before fitting
    numConcentrations = 8
    bindingSeries = np.transpose([np.array(table_reduced[i]) for i in range(numConcentrations)])
    allClusterSignal = np.array(table_reduced['all_cluster_signal'])
    concentrations = np.array([0.91, 2.74, 8.23, 24.69, 74.07, 222.2, 666.7, 2000]);
    
    # now split into parts
    numCores = 20
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
    
    # define some parameters
    parameters = fittingParameters.Parameters(concentrations,
                                              table[7],
                                              table['all_cluster_signal'],
                                              null_scores)
    num_clusters = len(table_reduced)
    dirname = 'binding_curves_rigid/fit_no_norm_wtiles456_figs'
    threshold = 0.05
    subsets = [np.arange(num_clusters)[np.array(table_reduced['qvalue'] < threshold)],
               np.arange(num_clusters)[np.array(table_reduced['qvalue'] > threshold)]]
    colors = ['r', 'b']
    labels = ['below', 'above']
    
    ###### fmax upper bound #####
    variation_type = ['fmax', 'upperbound']
    varying_params = parameters.vary_fmax_upperbounds
    resulting_dGs, resulting_var_dGs = find_resulting_dGs(fitParametersFilenameParts, bindingSeriesFilename, parameters.fitParameters, variation_type, varying_params, parameters.scale_factor)
    
    # plot
    for i in range(2):
        ax = plotErrorBarGraph(varying_params, resulting_dGs, resulting_var_dGs, variation_type,
                               subset=subsets[i], logscale=True, color=colors[i], indx_0=-1)
        plt.savefig('%s/%s_stability.%s_qvalue.wider_range.png'%(dirname, '_'.join(variation_type), labels[i]))
    ax = plotFracAffectedGraph(varying_params, resulting_dGs, resulting_var_dGs, subsets, colors, labels, logscale=True, indx_0=-1)
    plt.savefig('%s/%s_stability.fraction_below_var.wider_range.pdf'%(dirname, '_'.join(variation_type)))
    ax.set_ylim((0, 0.2))
    plt.savefig('%s/%s_stability.fraction_below_var.wider_range.zoom.pdf'%(dirname, '_'.join(variation_type)))
    
    ###### fmax lower bound #####
    variation_type = ['fmax', 'lowerbound']
    varying_params = parameters.vary_fmax_lowerbounds
    resulting_dGs, resulting_var_dGs = find_resulting_dGs(fitParametersFilenameParts, bindingSeriesFilename, parameters.fitParameters, variation_type, varying_params, parameters.scale_factor)
    
    # plot
    for i in range(2):
        ax = plotErrorBarGraph(varying_params, resulting_dGs, resulting_var_dGs, variation_type,
                               subset=subsets[i], color=colors[i], indx_0=0)
        ax.set_ylim((-1, 10))
        plt.savefig('%s/%s_stability.%s_qvalue.png'%(dirname, '_'.join(variation_type), labels[i]))
    ax = plotFracAffectedGraph(varying_params, resulting_dGs, resulting_var_dGs, subsets, colors, labels, indx_0=0)
    plt.savefig('%s/%s_stability.fraction_below_var.pdf'%(dirname, '_'.join(variation_type)))
    ax.set_ylim((0, 0.2))
    plt.savefig('%s/%s_stability.fraction_below_var.zoom.pdf'%(dirname, '_'.join(variation_type)))
    
    ###### fmax initial guess  #####
    variation_type = ['fmax', 'initial']
    varying_params = parameters.vary_scale_factor
    fit_one(fitParametersFilenameParts, bindingSeriesFilename,parameters.fitParameters,varying_params[0])
    resulting_dGs, resulting_var_dGs = find_resulting_dGs(fitParametersFilenameParts, bindingSeriesFilename, parameters.fitParameters, variation_type, varying_params, parameters.scale_factor,
                                                          vary_scale_factor=True)
    indx_to_compare = 5
    # plot
    for i in range(2):
        ax = plotErrorBarGraph(varying_params, resulting_dGs, resulting_var_dGs, variation_type,
                               subset=subsets[i], color=colors[i], indx_0=indx_to_compare, logscale=True)
        ax.set_ylim((-1, 10))
        plt.savefig('%s/%s_stability.%s_qvalue.png'%(dirname, '_'.join(variation_type), labels[i]))
    ax = plotFracAffectedGraph(varying_params, resulting_dGs, resulting_var_dGs, subsets, colors, labels, indx_0=indx_to_compare, logscale=True)
    plt.savefig('%s/%s_stability.fraction_below_var.pdf'%(dirname, '_'.join(variation_type)))
    ax.set_ylim((0, 0.2))
    plt.savefig('%s/%s_stability.fraction_below_var.zoom.pdf'%(dirname, '_'.join(variation_type)))
    
    
    ###### dG initial guess ######
    variation_type = ['dG', 'initial']
    varying_params = parameters.vary_dG_initial
    resulting_dGs, resulting_var_dGs = find_resulting_dGs(fitParametersFilenameParts, bindingSeriesFilename, parameters.fitParameters, variation_type, varying_params, parameters.scale_factor)
    indx_to_compare = 7 #index to compare to
    # plot
    for i in range(2):
        ax = plotErrorBarGraph(varying_params, resulting_dGs, resulting_var_dGs, variation_type,
                               subset=subsets[i], color=colors[i], indx_0=indx_to_compare)
        plt.savefig('%s/%s_stability.%s_qvalue.png'%(dirname, '_'.join(variation_type), labels[i]))
    ax = plotFracAffectedGraph(varying_params, resulting_dGs, resulting_var_dGs, subsets, colors, labels, indx_0=indx_to_compare)
    plt.savefig('%s/%s_stability.fraction_below_var.pdf'%(dirname, '_'.join(variation_type)))
    ax.set_ylim((0, 0.2))
    plt.savefig('%s/%s_stability.fraction_below_var.zoom.pdf'%(dirname, '_'.join(variation_type)))
    
    ###### fmin upperbound ######
    variation_type = ['fmin', 'upperbound']
    varying_params = parameters.vary_fmin_upperbound
    resulting_dGs, resulting_var_dGs = find_resulting_dGs(fitParametersFilenameParts, bindingSeriesFilename, parameters.fitParameters, variation_type, varying_params, parameters.scale_factor)
    indx_to_compare = -1
    # plot
    for i in range(2):
        ax = plotErrorBarGraph(varying_params, resulting_dGs, resulting_var_dGs, variation_type,
                               subset=subsets[i], color=colors[i], indx_0=indx_to_compare, logscale=True)
        plt.savefig('%s/%s_stability.%s_qvalue.png'%(dirname, '_'.join(variation_type), labels[i]))
    ax = plotFracAffectedGraph(varying_params, resulting_dGs, resulting_var_dGs, subsets, colors, labels, indx_0=indx_to_compare, logscale=True)
    plt.savefig('%s/%s_stability.fraction_below_var.pdf'%(dirname, '_'.join(variation_type)))
    ax.set_ylim((0, 0.2))
    plt.savefig('%s/%s_stability.fraction_below_var.zoom.pdf'%(dirname, '_'.join(variation_type)))
    
    ###### Parameter correlation plots ######
    fitParameters = fit_one(fitParametersFilenameParts, bindingSeriesFilename,parameters.fitParameters,parameters.scale_factor)
    num_clusters = len(table)
    subsets = [np.arange(num_clusters)[np.array(table['qvalue'] < threshold)],
               np.arange(num_clusters)[np.array(table['qvalue'] > threshold)]]

    # delta G
    parameter = 'dG'
    numbins = 50.0
    binsize = (parameters.fitParameters[parameter]['upperbound'] - parameters.fitParameters[parameter]['lowerbound'])/numbins
    xbins = np.arange(parameters.fitParameters[parameter]['lowerbound'], parameters.fitParameters[parameter]['upperbound']+binsize*2, binsize)-binsize/2
    ax = plotHistogram(table, parameter, subsets[1], subsets[0], xbins=xbins)
    ax.set_xlim((parameters.fitParameters[parameter]['lowerbound'], parameters.fitParameters[parameter]['upperbound']))
    plt.savefig('%s/deltaG.histogram.png'%(dirname))
    
    # fmax
    parameter = 'fmax'
    numbins = 50.0
    binsize = (parameters.fitParameters[parameter]['upperbound'] - parameters.fitParameters[parameter]['lowerbound'])/numbins
    xbins = np.arange(parameters.fitParameters[parameter]['lowerbound'], parameters.fitParameters[parameter]['upperbound']+binsize*2, binsize)-binsize/2
    ax = plotHistogram(table, parameter, subsets[1], subsets[0], xbins=xbins)
    ax.set_xlim((parameters.fitParameters[parameter]['lowerbound'], parameters.fitParameters[parameter]['upperbound']))
    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig('%s/%s.histogram.png'%(dirname, parameter))
    
    # fmax zoom
    parameter = 'fmax'
    numbins = 50.0
    fmax_upperbound_zoom = 2000
    binsize = (fmax_upperbound_zoom - parameters.fitParameters[parameter]['lowerbound'])/numbins
    xbins = np.arange(parameters.fitParameters[parameter]['lowerbound'], fmax_upperbound_zoom+binsize*2, binsize)-binsize/2
    ax = plotHistogram(table, parameter,  subsets[1], subsets[0], xbins=xbins)
    ax.set_xlim((parameters.fitParameters[parameter]['lowerbound'], fmax_upperbound_zoom))
    plt.xticks(rotation=70)
    plt.ylim((0, 200000))
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('%s/%s.zoom.histogram.png'%(dirname, parameter))
    
    # fmin
    parameter = 'fmin'
    upperbound = 5000
    binsize = (upperbound - parameters.fitParameters[parameter]['lowerbound'])/numbins
    xbins = np.arange(parameters.fitParameters[parameter]['lowerbound'], upperbound+binsize*2, binsize)-binsize/2
    ax = plotHistogram(table, parameter,  subsets[1], subsets[0], xbins=xbins)
    plt.xticks(rotation=70)
    plt.ylim((0, 200000))
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('%s/%s.histogram.png'%(dirname, parameter))
    # fmin zoom
    parameter = 'fmin'
    fmin_upperbound_zoom = 600
    binsize = (fmin_upperbound_zoom - parameters.fitParameters[parameter]['lowerbound'])/numbins
    xbins = np.arange(parameters.fitParameters[parameter]['lowerbound'], fmin_upperbound_zoom+binsize*2, binsize)-binsize/2
    ax = plotHistogram(table, parameter,  subsets[1], subsets[0], xbins=xbins)
    plt.xticks(rotation=70)
    plt.ylim((0, 200000))
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('%s/%s.zoom.histogram.png'%(dirname, parameter))
    
    
    #### plot correlations ####
    below_fdr_indx = 0
    # delta G versus fmax
    p1_name = 'dG'
    p2_name = 'fmax'
    c_name = 'rsq'
    ax = plotScatterplot(table_reduced, p1_name, p2_name, c_name, vmin=0, vmax=1)
    ax.set_xlim((parameters.fitParameters[p1_name]['lowerbound'], parameters.fitParameters[p1_name]['upperbound']))
    plt.savefig('%s/%s_vs_%s.color%s.all.scatterplot.png'%(dirname, p1_name, p2_name, c_name))
    ax.set_ylim((0, fmax_upperbound_zoom))
    plt.savefig('%s/%s_vs_%s.color%s.all.zoom.scatterplot.png'%(dirname, p1_name, p2_name, c_name))
    
    ax = plotScatterplot(table_reduced, p1_name, p2_name, c_name, subsets[below_fdr_indx], vmin=0, vmax=1)
    ax.set_xlim((parameters.fitParameters[p1_name]['lowerbound'], parameters.fitParameters[p1_name]['upperbound']))
    plt.savefig('%s/%s_vs_%s.color%s.%s_qvalue.scatterplot.png'%(dirname, p1_name, p2_name, c_name, labels[below_fdr_indx]))
    ax.set_ylim((0, fmax_upperbound_zoom))
    plt.savefig('%s/%s_vs_%s.color%s.%s_qvalue.zoom.scatterplot.png'%(dirname, p1_name, p2_name, c_name, labels[below_fdr_indx]))
    
    # delta G versus fmin
    p1_name = 'dG'
    p2_name = 'fmin'
    c_name = 'rsq'
    ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, vmin=0, vmax=1)
    ax.set_xlim((parameters.fitParameters[p1_name]['lowerbound'], parameters.fitParameters[p1_name]['upperbound']))
    plt.savefig('%s/%s_vs_%s.color%s.all.scatterplot.png'%(dirname, p1_name, p2_name, c_name))
    ax.set_ylim((0, fmin_upperbound_zoom))
    plt.savefig('%s/%s_vs_%s.color%s.all.zoom.scatterplot.png'%(dirname, p1_name, p2_name, c_name))
    
    ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, subsets[below_fdr_indx], vmin=0, vmax=1)
    plt.savefig('%s/%s_vs_%s.color%s.%s_qvalue.scatterplot.png'%(dirname, p1_name, p2_name, c_name, labels[below_fdr_indx]))
    ax.set_ylim((0, fmin_upperbound_zoom))
    plt.savefig('%s/%s_vs_%s.color%s.%s_qvalue.zoom.scatterplot.png'%(dirname, p1_name, p2_name, c_name, labels[below_fdr_indx]))
    
    # fmin vs fmax
    subset = np.arange(num_clusters)[np.array(fitParameters['qvalue'] < 0.05)]
    p1_name = 'fmax'
    p2_name = 'fmin'
    c_name = 'rsq'
    ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, vmin=0, vmax=1)
    plt.savefig('%s/%s_vs_%s.color%s.all.histogram.png'%(dirname, p1_name, p2_name, c_name))
    ax.set_xlim((0, fmax_upperbound_zoom))
    ax.set_ylim((0, fmin_upperbound_zoom))
    plt.savefig('%s/%s_vs_%s.color%s.all.zoom.histogram.png'%(dirname, p1_name, p2_name, c_name))
    
    ax = plotScatterplot(fitParameters, p1_name, p2_name, c_name, subsets[below_fdr_indx], vmin=0, vmax=1)
    plt.savefig('%s/%s_vs_%s.color%s.%s_qvalue.scatterplot.png'%(dirname, p1_name, p2_name, c_name, labels[below_fdr_indx]))
    ax.set_xlim((0, fmax_upperbound_zoom))
    ax.set_ylim((0, fmin_upperbound_zoom))
    plt.savefig('%s/%s_vs_%s.color%s.%s_qvalue.zoom.scatterplot.png'%(dirname, p1_name, p2_name, c_name, labels[below_fdr_indx]))
    
    # also compare correlation of dG to fmax at different thresholds
    thresholds = np.logspace(-3, 0)
    p1_name = 'fmin'
    p2_name = 'fmax'
    correlation = np.array([st.spearmanr(fitParameters[p1_name][np.array(fitParameters['qvalue']<threshold)], fitParameters[p2_name][np.array(fitParameters['qvalue']<threshold)])[0] for threshold in thresholds])
    correlation_var = np.array([st.spearmanr(fitParameters[p1_name][np.array(fitParameters['qvalue']<threshold)], fitParameters[p2_name][np.array(fitParameters['qvalue']<threshold)])[1] for threshold in thresholds])
    fig = plt.figure()
    ax2 = fig.add_subplot(211)
    ax2.bar(np.arange(len(thresholds)), [np.sum(fitParameters['qvalue']<threshold)/float(len(fitParameters)) for threshold in thresholds])
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.set_ylabel('fraction of clusters passing')
    ax = fig.add_subplot(212)
    ax.errorbar(thresholds, correlation, yerr=correlation_var, fmt='o-')
    ax.set_xscale('log')
    ax.set_xlabel('fdr cutoff')
    ax.set_ylabel('rank correlation %s to %s'%(p1_name, p2_name))
    plt.savefig('%s/correlation_%s_to_%s.png'%(dirname, p1_name, p2_name))
    
    ##### example fmin uppersbound affecting dG #####
    variation_type = ['fmin', 'upperbound']
    params = [np.nan, 1e2, 1e3]
    numtottest = len(params)
    fitParametersFinal = ['']*numtottest
    for i in range(numtottest):
        initialFitParameters = parameters.fitParameters.copy()
        initialFitParameters[variation_type[0]][variation_type[1]] = params[i] 
        fitParametersFinal[i] = fit_one(fitParametersFilenameParts, bindingSeriesFilename,initialFitParameters,parameters.scale_factor)
    # plot
    subset_indx = np.all((np.abs(fitParametersFinal[1]['dG']-fitParametersFinal[0]['dG']) > fitParametersFinal[1]['dG_var'],
                          np.abs(fitParametersFinal[2]['dG']-fitParametersFinal[0]['dG']) > fitParametersFinal[2]['dG_var']), axis=0)
    for loc in [0,1,2,3,4,5]:
        for i in range(numtottest):
            for name in fitParametersFinal[i]: table_reduced[name] = np.array(fitParametersFinal[i][name])
            variantFun.plotCluster(table_reduced[subset_indx].iloc[loc], concentrations)
            plt.title('%s = %4.1f'%(' '.join(variation_type), params[i]))
            plt.savefig('%s/example_binding_curves.%s_%s_%4.2f.cluster_%d.pdf'%(dirname,variation_type[0], variation_type[1], params[i], table_reduced[subset_indx].iloc[loc].name ))
            
    ##### example fmax lowerbound affecting dG #####
    variation_type = ['fmax', 'lowerbound']
    params = [0, 300, 600]
    numtottest = len(params)
    fitParametersFinal = ['']*numtottest
    for i in range(numtottest):
        initialFitParameters = parameters.fitParameters.copy()
        initialFitParameters[variation_type[0]][variation_type[1]] = params[i] 
        fitParametersFinal[i] = fit_one(fitParametersFilenameParts, bindingSeriesFilename,initialFitParameters,parameters.scale_factor)    
    subset_indx = np.all((np.abs(fitParametersFinal[1]['dG']-fitParametersFinal[0]['dG']) > fitParametersFinal[1]['dG_var'],
                          np.array(fitParametersFinal[1]['qvalue'] > threshold)), axis=0)
    for loc in range(1, 5):
        for i in range(2):
            for name in fitParametersFinal[i]: table_reduced[name] = np.array(fitParametersFinal[i][name])
            variantFun.plotCluster(table_reduced[subset_indx].iloc[loc], concentrations)
            plt.title('%s = %4.1f'%(' '.join(variation_type), params[i]))
            plt.ylim((0, 600))
            plt.savefig('%s/example_binding_curves.%s_%s_%4.2f.cluster_%d.pdf'%(dirname,variation_type[0], variation_type[1], params[i], table_reduced[subset_indx].iloc[loc].name ))
            
    
    ##### what about if you leave off last point? #####
    numtottest = 9
    fitParametersFinal = ['']*numtottest
    variation_type = ['dG', 'numpoints']
    
    # all points
    a = range(8)
    indx = a
    for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
        sio.savemat(bindingSeriesFilename, {'concentrations':concentrations[indx],
                                            'binding_curves': bindingSeriesSplit[i][:, indx].astype(float),
                                            'all_cluster':allClusterSignalSplit[i].astype(float),
                                            'null_scores':null_scores,
                                            })
    fitParametersFinal[0] = fit_one(fitParametersFilenameParts, bindingSeriesFilename,parameters.fitParameters,parameters.scale_factor)
    
    # missing one point
    for j in range(8):
        # remove last binding point
        a = range(8)
        a.remove(j)
        indx = a
        for i, bindingSeriesFilename in bindingSeriesFilenameParts.items():
            sio.savemat(bindingSeriesFilename, {'concentrations':concentrations[indx],
                                                'binding_curves': bindingSeriesSplit[i][:, indx].astype(float),
                                                'all_cluster':allClusterSignalSplit[i].astype(float),
                                                'null_scores':null_scores,
                                                })
        fitParametersFinal[j+1] =  fit_one(fitParametersFilenameParts, bindingSeriesFilename,parameters.fitParameters,parameters.scale_factor)
        
    # histogram
    labels = np.hstack(('all', ['missing point %d'%i for i in range(1,9)]))
    fitParameters = fitParametersFinal[0]
    subset = np.arange(num_clusters)[np.array(fitParameters['qvalue'] < threshold)]
    parameter = 'dG'
    numbins = 50.0
    binsize = (parameters.fitParameters[parameter]['upperbound'] - parameters.fitParameters[parameter]['lowerbound'])/numbins
    xbins = np.arange(parameters.fitParameters[parameter]['lowerbound'], parameters.fitParameters[parameter]['upperbound']+binsize*2, binsize)-binsize/2
    histogram.compare([fitParametersFinal[i][parameter].iloc[subset] for i in range(numtottest)],
        cmap='Paired', xbins=xbins, normalize=False, labels = labels)
    plt.xlabel(parameter)
    plt.ylabel('number')
    plt.ylim((0, 500))
    plt.savefig('%s/%s_stability.below_qvalue.histogram_dGs.png'%(dirname, '_'.join(variation_type)))
    
    # plot
    resulting_dGs = np.array([fitParametersFinal[i]['dG'] for i in range(numtottest)])
    resulting_var_dGs = np.array([fitParametersFinal[i]['dG_var'] for i in range(numtottest)])

    varying_params = range(numtottest)
    indx_to_compare = 0
    for i in range(2):
        ax = plotErrorBarGraph(varying_params, resulting_dGs, resulting_var_dGs, variation_type,
                               subset=subsets[i], color=colors[i], indx_0=indx_to_compare)
        plt.savefig('%s/%s_stability.%s_qvalue.png'%(dirname, '_'.join(variation_type), labels[i]))
    ax = plotFracAffectedGraph(varying_params, resulting_dGs, resulting_var_dGs, subsets, colors, labels, indx_0=indx_to_compare)
    plt.savefig('%s/%s_stability.fraction_below_var.pdf'%(dirname, '_'.join(variation_type)))
    ax.set_ylim((0, 0.2))
    plt.savefig('%s/%s_stability.fraction_below_var.zoom.pdf'%(dirname, '_'.join(variation_type)))
