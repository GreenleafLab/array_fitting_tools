import datetime
import scipy.cluster.hierarchy as sch
import scipy.cluster as sc
import scipy.spatial.distance as ssd
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import seaborn as sns
from scikits.bootstrap import bootstrap
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF

def clusterKmeans(ddGs, k):
    centroid, label = sc.vq.kmeans2(ddGs, k)
    return label

def clusterHierarchical(M, k, return_z=None):
    if return_z is None:
        return_z = False
    z = sch.linkage(M, method='average', metric='euclidean')
    labels = pd.Series(sch.fcluster(z, k, 'maxclust'), index=M.index)
    if return_z:
        return labels, z
    else:
        return labels
    
def clusterHierarchicalCorr(M, k, return_z=None):
    if return_z is None:
        return_z = False
    z = sch.linkage(M, method='average', metric='correlation')
    labels = pd.Series(sch.fcluster(z, k, 'maxclust'), index=M.index)
    if return_z:
        return labels, z
    else:
        return labels
    
def scrubInput(ddG):
    # normalize rows or columns here?
    a = ddG.dropna(how='any', axis=0).astype(float)
    return pd.DataFrame(sc.vq.whiten(a), index=a.index, columns=a.columns)

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

def getCDF(M, plot=None):
    if plot is None:
        plot = False
    upperTri = pd.DataFrame(np.triu(np.ones((len(M), len(M))), k=1).astype(bool),
                            index=M.index, columns=M.columns)
    flattened = np.ravel(M[upperTri])
    ecdf = ECDF(flattened[np.isfinite(flattened)], side='right')
    x = np.linspace(0, 1, 101)
    
    cdf = np.append(0, ecdf(x))
    x = np.append(0, x)

    if plot:
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
   
def findConnectivity(D, indices, method, k, transpose=None):
    if transpose is None: transpose = False
    
    # transpose in the case that indices are subsampling genes rather than samples
    if transpose:
        index_all = D.columns
    else:
        index_all = D.index
        
    # initialize mat
    M, I = computeConnectivity(index=index_all)
    
    for index in indices:
        subD = D.loc[index]
        if transpose:
            clusters = pd.Series(method(subD.transpose(), k), index=index_all)
        else:
            clusters = pd.Series(method(subD, k), index=index)
        m, n = computeConnectivity(clusters, index=index_all)
        M += m
        I += n

    return M, I

def consensusCluster(D, method=None, subsample=None, n_samples=None, k=None,
                     plot=None, numCores=None, transpose=None):
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
    
    if numCores is None:
        numCores = 10
    
    if transpose is None:
        transpose = False
        # transpose is False: subsample items
        # transpose is True: subsample features
        
    if transpose:
        index_all = D.columns # these will be the rows/cols of the consensus matrix
    else:
        index_all = D.index   # these will be the rows/cols of the consensus matrix
        
    indices = ['']*n_samples
    for i in range(n_samples):
        indices[i] = np.random.choice(D.index,
                                      size=int(len(D)*subsample),
                                      replace=False)
    M, I = computeConnectivity(index=index_all)
    indicesSplit = np.array_split(indices, numCores)
    a = (Parallel(n_jobs=numCores, verbose=10)
         (delayed(findConnectivity)(D, index, method, k, transpose) for index in indicesSplit))
    for m, n in a:
        M += m
        I += n      
    M = M/I
        
    labels, z = clusterHierarchical(M, k, return_z=True)
    if plot:
        sns.clustermap(M, yticklabels=False, xticklabels=False, square=True,
                       row_linkage=z, col_linkage=z)
    return labels, M

def optimizeNumClusters(D, method=None, subsample=None, n_samples=None, ks=None,
                        numCores=None):
    if ks is None:
        ks = np.arange(5, 30, 5)
    
    cdfs = {}
    for k in ks:
        print k
        labels, M = consensusCluster(D, method=method, subsample=subsample,
                                     n_samples=n_samples, k=k, numCores=numCores,
                                     plot=False)
        x, cdf = getCDF(M)
        cdfs[k] = pd.Series(cdf, index=x)
        
    return pd.concat(cdfs, axis=1)


def plotCDFs(cdfs):
    
    x = cdfs.index
    values = cdfs.columns
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
    
def getDeltaK(cdfs):
    x = cdfs.index
    k = cdfs.columns
    areas = pd.Series([findAreaOfCDF(x, cdfs.loc[:, k].values) for k in cdfs],
        index = cdfs.columns)
    areasPrime = pd.Series([areas.loc[:k].max() for k in areas.index],
        index = cdfs.columns)
    deltaK = pd.Series(index=cdfs.columns[:-1])
    
    for i, k in enumerate(cdfs.columns[:-1]):
        if k == 2:
            deltaK.loc[k] = areasPrime.loc[k]
        else:
            deltaK.loc[k] = (areasPrime.loc[k+1] - areasPrime.loc[k])/areasPrime.loc[k]
            
    return deltaK

def getWeightedDistance(observations, errors):
    """ return condensed, weighted distance matrix.
    
    'errors' are the 95% confidence interval widths.
    'weight will be the inverse variance ([error/(2*1.96)]**2)"""
    sigma = errors/(2*1.96)
    

    return ssd.pdist(observations, metric='wminkowski', p=2, w=1./sigma**2)
    
    