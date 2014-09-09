"""
Date: 9/9/14
Author: Sarah Denny

This module is intended as a library to make plots relevant to rna array output
"""

##### IMPORT MODULES #####
# import necessary for python
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import *
from scipy.optimize import curve_fit
import functools

def fitFunc(xdata, kd):
    return xdata/(kd + xdata)

def plotClusters(redArray, greenArray, criteria, label):
    """
    
    """
    
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    binwidth = 35
    xbins = np.arange(0-binwidth*0.5, 1000+binwidth*0.5, binwidth)
    bincenters = (xbins[:-1] + xbins[1:])*0.5
    
    hists = ['']*3
    hists[0] = np.histogram(redArray[criteria], bins=xbins)[0]
    hists[1] = np.histogram(greenArray[criteria], bins=xbins)[0]
    hists[2] = np.histogram(np.divide(greenArray[criteria], redArray[criteria])*np.mean(redArray[criteria]), bins=xbins)[0]
    
    ax1.fill_between(bincenters, hists[0], facecolor='0.5', alpha=0.1)
    ax1.plot(bincenters, hists[0], 'r', linewidth=2)
    ax1.set_ylim((0, np.max(np.array(hists))))
    ax1.set_ylabel('Number of clusters')
    ax1.set_title('all RNA')
    
    ax2.fill_between(bincenters, hists[1], facecolor='0.5', alpha=0.1)
    ax2.plot(bincenters, hists[1], 'g', linewidth=2)
    ax2.set_xlabel('Fluorescence value')
    ax2.set_ylim((0, np.max(np.array(hists))))
    ax2.set_yticklabels([])
    ax2.set_title(label)
    
    ax3.fill_between(bincenters, hists[2], facecolor='0.5', alpha=0.1)
    ax3.plot(bincenters, hists[2], 'b', linewidth=2)
    ax3.set_title('Normalized %s'%label)
    ax3.set_ylim((0, np.max(np.array(hists))))
    ax3.set_yticklabels([])
    plt.tight_layout()
    
    return

def findFmax(redArray, greenArray, criteria):
    fmax = np.median(greenArray[criteria]/redArray[criteria])
    return fmax

def plotBindingCurve(xvalues, yvalues, concentrations):
    """
    
    """
    minx = 0.1
    xnew = xvalues + minx
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xnew, yvalues, alpha=0.05, linewidth=0)
    ax.set_xscale('log')
    ax.plot(concentrations, np.array([np.median(yvalues[xvalues==concentration]) for concentration in concentrations]), 'o')
    ax.set_ylim((0, np.percentile(yvalues, 99)))
    
    indx = yvalues < np.percentile(yvalues, 99)
    popt, pcov = curve_fit(fitFunc, xvalues[indx], yvalues[indx])
    kd = popt[0]
    perr = np.sqrt(np.diag(pcov))*1.96
    
    xdata = np.logspace(np.log10(minx), np.log10(np.max(xvalues)), 100)
    ax.plot(xdata, fitFunc(xdata, kd), 'k', label='Kd=%4.1f (%4.1f, %4.1f)'%(kd, kd-perr[0], kd+perr[0]))
    ax.set_xlim((minx*0.9, np.max(xvalues)*1.1))
    ax.set_xticks(concentrations+minx)
    ax.set_xticklabels(concentrations)
    
    ax.set_xlabel('concentration (nM)')
    ax.set_ylabel('normalized fluorescence')
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left')
    
    return kd

def plotClustersNew(redArray, greenArray, criteria, labels):
    """
    
    """
    numConcentrations = len(labels)
    
    fig = plt.figure(figsize=(8,12))
    gs = gridspec.GridSpec(numConcentrations,3)
    
    for i in range(numConcentrations):
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 2])

        binwidth = 35
        xbins = np.arange(0-binwidth*0.5, 1000+binwidth*0.5, binwidth)
        bincenters = (xbins[:-1] + xbins[1:])*0.5
    
        hists = ['']*3
        hists[0] = np.histogram(redArray[criteria, i], bins=xbins)[0]
        hists[1] = np.histogram(greenArray[criteria, i], bins=xbins)[0]
        hists[2] = np.histogram(np.divide(greenArray[criteria, i], redArray[criteria, i])*np.mean(redArray[criteria, i]), bins=xbins)[0]
    
        ax1.fill_between(bincenters, hists[0], facecolor='0.5', alpha=0.1)
        ax1.plot(bincenters, hists[0], 'r', linewidth=2)
        ax1.set_ylim((0, np.max(np.array(hists))))
        ax1.set_ylabel('Number of clusters')
        ax1.set_yticks(np.linspace(0, np.around(np.max(np.array(hists)), -2), 4).astype(int))
        ax1.set_xticks([0, 200, 400, 600, 800])
        ax1.set_xticklabels([])
        
        ax2.fill_between(bincenters, hists[1], facecolor='0.5', alpha=0.1)
        ax2.plot(bincenters, hists[1], 'g', linewidth=2)
        ax2.set_ylim((0, np.max(np.array(hists))))
        ax2.set_yticks(np.linspace(0, np.around(np.max(np.array(hists)), -2), 4).astype(int))
        ax2.set_yticklabels([])
        ax2.set_xticks([0, 200, 400, 600, 800])
        ax2.set_xticklabels([])
        
        
        ax3.fill_between(bincenters, hists[2], facecolor='0.5', alpha=0.1)
        ax3.plot(bincenters, hists[2], 'b', linewidth=2)
        
        ax3.set_ylim((0, np.max(np.array(hists))))
        ax3.set_yticks(np.linspace(0, np.around(np.max(np.array(hists)), -2), 4).astype(int))
        ax3.set_yticklabels([])
        ax3.set_xticks([0, 200, 400, 600, 800])
        ax3.set_xticklabels([])
        ax3.set_ylabel(labels[i])
        ax3.yaxis.set_label_position("right")
        
        if i == 0:
            ax1.set_title('red channel')
            ax2.set_title('green channel')
            ax3.set_title('green/red')
    
    ax2.set_xlabel('Fluorescence value')
    ax1.set_xticklabels([0, 200, 400, 600, 800])
    ax2.set_xticklabels([0, 200, 400, 600, 800])
    ax3.set_xticklabels([0, 200, 400, 600, 800])
    plt.tight_layout()
    
    return
