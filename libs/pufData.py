import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import seaborn as sns
import os
import itertools

from plotFun import fix_axes

sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1, 'axes.linewidth':1,
                        'text.color': 'k',
                        'axes.labelcolor': 'k',
                        'axes.color':'k'})

class perVariant():
    """ Class to store data from puf project experiment and plot puf-project-specific plots.
    
    Parameters:
    -----------
    cpvariant : table with combined data and library characterization
    """
    def __init__(self, cpvariant):
        self.variant_table = cpvariant.loc[:, :'flag']
        self.libChar = cpvariant.loc[:, 'name':]
        
        
    def plotDoubleMutants(self, puf_id, scaffold='S1'):
        selectDict = {'PUF_SCAFFOLD':scaffold, 'PUF_ENTIRE':puf_id, 'PUF_INSERTIONS':0}
        subresults = self.libChar.loc[selectVariantsInDict(self.libChar, selectDict)]
        
        # drop duplicates
        numVariants = len(subresults)
        subresults = subresults.loc[subresults.sequence.drop_duplicates().index]
        print ("Keeping unique sequences. (%4.1f%%)"
               %(100*len(subresults)/float(numVariants)))
        
        # additionally filter for those variants with the same size
        seqLengths = pd.Series(len(s) for s in subresults.sequence).value_counts()
        print ("Keeping sequences with length %d. (%4.1f%%)"
               %(seqLengths.idxmax(), 100*seqLengths.max()/float(seqLengths.sum())))
        subresults = subresults.loc[[len(s)==seqLengths.idxmax() for s in subresults.sequence]]
        
        # only keep those wiht less than or equal to two mutations
        numVariants = len(subresults)
        subresults = subresults.loc[subresults.PUF_MUTATIONS<=2]
        print ("Keeping only variants with less than or equal to 2 mutations. (%4.1f%%)"
               %(100*len(subresults)/float(numVariants)))
        
        # filter for number of mutations?
        seqMat = subresults.sequence
        consensus = seqMat.loc[subresults.PUF_MUTATIONS==0]
        
        if len(consensus)==0:
            print "Error: no consensus sequences found!"
            return
        elif len(consensus) > 1:
            print "Error: more than one consensus sequence found!"
            return
        else:
            consensus_idx = consensus.index[0]
            consensus = consensus.loc[consensus_idx]
            
        
        # initiate matrix
        tuples = [(i, j) for i, j in itertools.product(np.arange(len(consensus)), ['A', 'C', 'G', 'T'])]
        index  = pd.MultiIndex.from_tuples(tuples, names=['location', 'base'])
        mat = pd.DataFrame(index=index, columns=index)
        mat_error = pd.DataFrame(index=index, columns=index)

        dGs = self.variant_table.dG
        dGerrors = (self.variant_table.dG_ub - self.variant_table.dG_lb)/2.
        
        print ("Comparing to consensus value %4.2f (+/- %4.2f)"
               %(dGs.loc[consensus_idx], dGerrors.loc[consensus_idx]))
               
        # assign to appropriate location
        for idx, seq in seqMat.iteritems():
            mutations = [(i, b) for i, (a, b) in enumerate(zip(list(consensus), list(seq))) if a!=b]
            deltaDeltaG = dGs.loc[idx] - dGs.loc[consensus_idx]
            
            index1 = mutations[0]
            if len(mutations)==1:
                index2 = mutations[0]
            elif len(mutations)==2:
                index2 = mutations[1]
            else:
                print 'Variant %d has no single or double mutations from consensus.'%idx
            
            mat.loc[index1, index2] = deltaDeltaG
            mat_error.loc[index1,index2] = errorPropagateAdd(
                dGerrors.loc[idx], dGerrors.loc[consensus_idx]) 
        # drop no mutation columns
        for col in zip(np.arange(len(consensus)), consensus):
            mat.drop(col, axis=0, inplace=True)
            mat.drop(col, axis=1, inplace=True)
            mat_error.drop(col, axis=0, inplace=True)
            mat_error.drop(col, axis=1, inplace=True)
            
        return mat, mat_error

def errorPropagateAdd(dG_error1, dGerror2):
    return np.sqrt(dGerror2**2 + dG_error1**2)
    
def getPredictedDoubleMutants(mat, mat_error):
    tuples =  zip(*[mat.index.get_level_values(i).tolist() for i in [0,1]])
    predMat = pd.DataFrame(index=mat.index, columns=mat.columns)
    predMat_error = pd.DataFrame(index=mat.index, columns=mat.columns)
    for location1, base1 in tuples:
        for location2, base2 in tuples:
            if location2 > location1:
                index1 = (location1, base1)
                index2 = (location2, base2)
                dG1 = mat.loc[index1, index1]
                dG2 = mat.loc[index2, index2]
                dG_error1 = mat_error.loc[index1, index1]
                dG_error2 = mat_error.loc[index2, index2]
                predMat.loc[index1, index2] = (dG1 + dG2)
                predMat_error.loc[index1, index2]  = (
                    errorPropagateAdd(dG_error1,  dG_error2))


    return predMat, predMat_error  

def plotScatterPlot(mat, predMat, mat_error=None, color='error', xlim=None, ylim=None):
    x = mat.unstack(level=0).unstack()
    y = predMat.unstack(level=0).unstack()
    if color=='error':
        c = mat_error.unstack(level=0).unstack()
    plt.figure(figsize=(3,3));
    plt.scatter(x, y, c=c, cmap='Spectral_r', vmin=0, vmax=5);
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('observed $\Delta \Delta G$ (kcal/mol)')
    plt.ylabel('predicted $\Delta \Delta G$ (kcal/mol)')

    ax = fix_axes(plt.gca())
    limits = ax.get_ylim()
    plt.plot(limits, limits, 'k--')
    plt.annotate('rsq=%4.2f'%getCorrelation(x, y)**2,
                 xy=(.025, .975),
                 xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=10)
    
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    
            

def selectVariantsInDict(libChar, selectDict):
    """ Find rows that share set values  for indicated columns in library characterization table.
    
    Parameters:
    -----------
    libChar: table describing variants.
    selectDict: dictionary with keys= columns of libChar file and value=to what value that column should equal
    """
    print "Keeping variants with:"
    for key, value in selectDict.items():
        print "\t%s = %s"%(key, str(value))
        
    return pd.concat([libChar.loc[:, key]==value for key, value in selectDict.items()], axis=1).all(axis=1)

def plotHeatmap(mat, vmin=-4, vmax=4, tick_font_size=8):
    with sns.axes_style("white", {
                        'lines.linewidth': 1, 'axes.linewidth':1,
                        'text.color': 'k',
                        'axes.labelcolor': 'k',
                        'axes.color':'k'}):
        plt.figure(figsize=(4.5,3.5));
        sns.heatmap(mat.astype(float), vmin=vmin, vmax=vmax, cmap='coolwarm') #, interpolation='nearest', aspect='auto')
        for i in np.arange(0, len(mat)+3, 3):
            plt.axhline(i, color='w', linewidth=1)
            plt.axvline(i, color='w', linewidth=1)
        plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)
        ax = plt.gca()
        ax.tick_params(top='off', right='off', pad=2, labelsize=tick_font_size, labelcolor='k')
        #plt.xlim(0, mat.shape[1]+1)
        #plt.ylim(0, mat.shape[0]+1)   
    # plot lines at location markers
    
def getCorrelation(vec1, vec2):
    index = np.all(np.isfinite(np.vstack([vec1.astype(float), vec2.astype(float)])), axis=0)
    return st.pearsonr(vec1[index], vec2[index])[0]
    