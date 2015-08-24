import datetime
import scipy.cluster.hierarchy as sch
import scipy.cluster as sc
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import seaborn as sns
from scikits.bootstrap import bootstrap
from statsmodels.distributions.empirical_distribution import ECDF

def clusterKmeans(ddGs, k):
    centroid, label = sc.vq.kmeans2(ddGs, k)
    return label

def clusterHierarchical(M, k, return_z=None):
    if return_z is None:
        return_z = False
    z = sch.linkage(M, method='average')
    labels = pd.Series(sch.fcluster(z, k, 'maxclust'), index=M.index)
    if return_z:
        return labels, z
    else:
        return labels
    
def scrubInput(ddG):
    # normalize rows or columns here?
    return ddG.dropna(how='any', axis=0).astype(float)

def computeConnectivity(clusters=None, index=None):
    if clusters is None:
        if index is None:
            return
        else:
            return (pd.DataFrame(0, index=index, columns=index),
                    pd.DataFrame(0, index=index, columns=index))
            
    if index is None:
        index = clusters.index
    
    n = pd.DataFrame(0, index=index, columns=index)
    m = pd.DataFrame(0, index=index, columns=index)
    for i in np.unique(clusters):
        subclusters = clusters.loc[clusters==i].index
        m.loc[subclusters, subclusters] = 1
        n.loc[clusters.index, clusters.index] = 1
    return m, n  

def getCDF(M):
    upperTri = pd.DataFrame(np.triu(np.ones((len(M), len(M))), k=1).astype(bool),
                            index=M.index, columns=M.columns)
    flattened = np.ravel(M[upperTri])
    ecdf = ECDF(flattened[np.isfinite(flattened)], side='right')
    x = np.linspace(0, 1, 101)
    
    cdf = np.append(0, ecdf(x))
    x = np.append(0, x)

    plt.figure(figsize=(3, 3))
    plt.plot(x, cdf)
    plt.xlim(-0.01, 1.01)
    ax = plt.gca()
    ax.tick_params(right='off', top='off')
    plt.xlabel('consensus index value')
    plt.ylabel('cdf')
    plt.tight_layout()
    
    return x, cdf
        
def findAreaOfCDF(x, cdf):
    return np.sum([cdf[i]*(x[i] - x[i-1]) for i in np.arange(len(x)-1)])
   


def consensusCluster(ddGs, method=None, subsample=None, n_samples=None, k=None,
                     plot=None):
    if n_samples is None:
        n_samples = 100
    
    if k is None:
        k = 5
        
    if subsample is None:
        subsample = 0.8
    
    if method is None:
        method = clusterKmeans
    
    if plot is None:
        plot = True

    indices = ['']*n_samples
    for i in range(n_samples):
        indices[i] = np.random.choice(ddGs.index,
                                      size=int(len(ddGs)*subsample),
                                      replace=False)

    M, I = computeConnectivity(index=ddGs.index)
    for i, index in enumerate(indices):
        if i%10==0:
            print '\t%4.1f%% (%d out of %d)'%(i/float(n_samples)*100, i, n_samples)
        clusters = pd.Series(method(ddGs.loc[index], k), index=index)
        m, n = computeConnectivity(clusters, index=ddGs.index)
        M += m
        I += n
    M = M/I
    labels, z = clusterHierarchical(M, k, return_z=True)
    if plot:
        sns.clustermap(M, yticklabels=False, xticklabels=False, square=True,
                       row_linkage=z, col_linkage=z)
    return labels, M

def optimizeNumClusters(ddGs, method=None, subsample=None, n_samples=None, ks=None):
    if ks is None:
        ks = np.arange(5, 30, 5)
    
    cdfs = {}
    for k in ks:
        print k
        labels, M = consensusCluster(ddGs, method=method, subsample=subsample,
                                     n_samples=n_samples, k=k,
                                     plot=False)
        x, cdf = getCDF(M)
        cdfs[k] = pd.Series(cdf, index=x)
        
    return cdfs


def plotCDFs(cdfs):
    
    x = cdfs.index
    values = np.arange(3, 23, 2)
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(values)-1)
    cm = 'coolwarm'
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
    
    
    plt.figure()
    for i, col in enumerate(values):
        colorVal = scalarMap.to_rgba(i)
        plt.plot(x, cdfs.loc[:, col], color=colorVal, label=col)
    
    values =  np.arange(23, 50, 4) 
    cm = 'Spectral'
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(values)-1)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)

    for i, col in enumerate(values):
        if i%2==0:
            colorVal = 'k'
        else:
            colorVal = '0.7'
        #colorVal = scalarMap.to_rgba(i)
        plt.plot(x, cdfs.loc[:, col], color=colorVal, label=col)
    
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(figDirectory, 'all_cdfs.subsampled_0.8.n_samples_500.pdf'))