from lmfit import minimize, Parameters, report_fit, conf_interval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scikits.bootstrap import bootstrap
from statsmodels.stats.weightstats import DescrStatsW
import warnings
import itertools
from collections import deque
import seqfun
import IMlibs
import scipy.stats as st
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1})
from fitFun import fittingParameters
import hjh.junction

complements = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}

def objectiveFunction(params, y=None, return_weighted=None, return_pred=None):
    if return_pred is None:
        return_pred = False
    if return_weighted is None:
        return_weighted = False
    parameters = fittingParameters()
    
    parvals = params.valuesdict()

    diff = y.dG.copy()
    for idx in y.index:
        length, seq, pos = idx
        
        term1 = parvals['bind_%d'%length]
        term2 = parameters.RT*np.log(parvals[y.loc[idx].seq])
        
        diff.loc[idx] = term1 + term2 
    
    if return_pred:
        return diff.astype(float)
    elif return_weighted:
        return ((diff - y.dG)*y.weight).astype(float)
    else:
        return (diff - y.dG).astype(float)

    
def fitThreeWay(y, weight=None, force=None):
    if weight is None: weight = False
    if force is None: force = False
    lengths = y.index.levels[0].tolist()
    seqs = y.index.levels[1].tolist()
    
    # store fit parameters in class for fitting
    params = Parameters()
    for length in lengths:
        params.add('bind_%d'%length, value=-9, 
                       min = -16,
                       max = -4)
    
    for seq in seqs:
        circPermutedSeqs = y.loc[lengths[0],seq].seq
        if force:
            vary = False
        else:
            vary = True
            
        params.add(circPermutedSeqs.loc[0],
                    value=1./3, min=0, max=1, vary=vary)
        params.add(circPermutedSeqs.loc[1],
                    value=1./3, min=0, max=1, vary=vary)
        
        params.add(circPermutedSeqs.loc[2],
                   expr='1-%s-%s'%(circPermutedSeqs.loc[0], circPermutedSeqs.loc[1]))
    
    func = objectiveFunction
    results = minimize(func, params,
                       args=(y.loc[y.variance > 0],),
                       kws={'return_weighted':weight})
    
    # find rsq
    ss_total = np.sum((y.dG - y.dG.mean())**2)
    ss_error = np.sum((results.residual)**2)
    rsq = 1-ss_error/ss_total
    rmse = np.sqrt(ss_error)
    
    ## plot residuals
    #plt.figure()
    #im = plt.scatter(y.dropna().dG, func(params, y=y.dropna(), return_pred=True),
    #            c = y.dropna().weight, cmap='coolwarm')
    #plt.colorbar(im)
    
    param_names = params.valuesdict().keys()
    index = (param_names + ['%s_stde'%param for param in param_names] +
             ['rsq', 'exit_flag', 'rmse'])
    final_params = pd.Series(index=index)
    for param in param_names:
        final_params.loc[param] = params[param].value
        final_params.loc['%s_stde'%param] = params[param].stderr
    final_params.loc['rsq'] = rsq
    final_params.loc['exit_flag'] = results.ier
    final_params.loc['rmse'] = rmse
    
    return final_params

def findVariantsByLengthAndCircularlyPermutedSeq(subtable, junction_type=None, lengths=None,
                                                 topology=None, loop=None):
    if junction_type is None:
        junction_type = 3
        
    if lengths is None:
        lengths = np.unique(subtable.helix_one_length)
    
    if topology is None:
        topology = '__'
    
    if loop is None:
        loop = 'GGAA_UUCG'
        
    # find all seqeunce possibilities
    seqs = hjh.junction.Junction(('W', 'W', 'W')).sequences.side1
    
    # split topoloogy
    insertions = topology.split('_')
    
    y = {}
    for length in lengths:
        y[length] = {}
        done_seqs = []
        for seq in seqs:
            if np.all([i==seq[0] for i in seq]):
                pass
                #print '\tSkipping junction %s because no permutations'%seq
            elif seq in done_seqs:
                pass
                #print '\tJunction %s already done'%seq
            else:
                #print 'doing junction %s'%seq
                
                y[length][seq] = pd.DataFrame(index=np.arange(junction_type),
                                              columns=['dG', 'variance', 'weight', 'seq', 'variant', 'correct'])
                g = deque(np.arange(junction_type))
                for idx in np.arange(junction_type):
                    
                    ins = [insertions[i] for i in list(g)]
                    seq_list = [seq[i] for i in list(g)]
                    done_seqs.append(''.join(seq_list))
                    index = ((subtable.helix_one_length == length)&
                             (subtable.junction_seq == '_'.join(seq_list))&
                             (subtable.insertions   == '_'.join(ins))&
                             (subtable.loop == loop))
                    if index.sum() == 1:
                        y[length][seq].loc[idx, 'dG'] = subtable.loc[index].dG.values[0]
                        y[length][seq].loc[idx, 'variance'] = (((subtable.loc[index].eminus + subtable.loc[index].eplus)/2)**2).values[0]
                        y[length][seq].loc[idx, 'weight'] = (1/(subtable.loc[index].eminus + subtable.loc[index].eplus)).values[0]
                        y[length][seq].loc[idx, 'variant'] = index.loc[index].index[0]
                        try:
                            y[length][seq].loc[idx, 'correct'] = (subtable.loc[index].ss_correct=='True').values[0]
                        except:
                            pass
                    y[length][seq].loc[idx, 'seq'] = '_'.join(seq_list)
                    
                    # rotate g
                    g.rotate(1)
        y[length] = pd.concat(y[length])
    return pd.concat(y)
    

def findDataMat(subtable, y, results):
    parameters = fittingParameters()
    data = pd.concat([subtable.loc[y.variant].dG, pd.Series(subtable.loc[y.variant+1].dG.values,
                                                            index=y.variant)], axis=1, keys = ['fit', 'not_fit'])
    data.index = y.index
    data.loc[:, 'seq'] = y.seq
    data.loc[:, 'ddG_pred'] = np.nan
    data.loc[:, 'dG_conf'] = np.nan
    data.loc[:, 'dG_bind'] = np.nan
    data.loc[:, 'nn_pred'] = np.nan
    for idx in data.index:
        
        seq = data.loc[idx].seq
        length = idx[0]
        
        fAC = results.loc[seq]
        fBC = results.loc['_'.join([seq.split('_')[i] for i in [0, 2, 1]])]
        data.loc[idx, 'ddG_pred'] = parameters.RT*np.log(fAC/fBC)
    
        data.loc[idx, 'dG_conf'] = parameters.RT*np.log(results.loc[seq])
        if 'bind_%d'%length in results.index:
            data.loc[idx, 'dG_bind'] = results.loc['bind_%d'%length]
            
        data.loc[idx, 'residual'] = data.loc[idx, 'dG_conf'] + data.loc[idx, 'dG_bind'] - data.loc[idx, 'fit']
        
    # add nearest neighbor info
    nnFile = '~/JunctionLibrary/seq_params/nearest_neighbor_rules.txt'
    if not os.path.exists(nnFile):
        nnFile = '/Users/Sarah/python/JunctionLibrary/seq_params/nearest_neighbor_rules.txt'
    if not os.path.exists(nnFile):
        print 'Error: could not find nearest neighbor file'
        return data
    
    nn = pd.read_table(nnFile, index_col=0).astype(float)
    
    complements = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}
    for idx in data.index:
        length = idx[0]
        seq = data.loc[idx].seq.replace('T', 'U').split('_')
        v1 = pd.Series([seq[0], complements[seq[0]]], index = ['x', 'y'])
        v2 = pd.Series([seq[1], complements[seq[1]]], index = ['x', 'y'])
        v3 = pd.Series([seq[2], complements[seq[2]]], index = ['x', 'y'])
        if length == 3:
            bp3 = pd.Series(['U', 'A'], index = ['x', 'y'])
            bp1 = pd.Series(['A', 'U'], index = ['x', 'y'])
            bp2 = pd.Series(['C', 'G'], index = ['x', 'y'])
        elif length == 4:
            bp3 = pd.Series(['C', 'G'], index = ['x', 'y'])
            bp1 = pd.Series(['G', 'C'], index = ['x', 'y'])
            bp2 = pd.Series(['U', 'A'], index = ['x', 'y'])
        elif length == 5:
            bp3 = pd.Series(['C', 'G'], index = ['x', 'y'])
            bp1 = pd.Series(['A', 'U'], index = ['x', 'y'])
            bp2 = pd.Series(['C', 'G'], index = ['x', 'y'])
        elif length == 6:
            bp3 = pd.Series(['A', 'U'], index = ['x', 'y'])
            bp1 = pd.Series(['A', 'U'], index = ['x', 'y'])
            bp2 = pd.Series(['U', 'A'], index = ['x', 'y'])
        else:
            print 'error'
        dg0 = nn.loc[bp1.x + v1.x] + nn.loc[v1.x + v3.x] + nn.loc[v3.x + bp3.x]
        dg1 = nn.loc[bp3.y + v3.y] + nn.loc[v3.y + v2.x] + nn.loc[v2.x + bp2.x]
        dg2 = nn.loc[bp2.y + v2.y] + nn.loc[v2.y + v1.y] + nn.loc[v1.y + bp1.y]
        
        dg0 = nn.loc[v1.x + v3.x]
        dg2 = nn.loc[v2.y + v1.y] 
        
        data.loc[idx, 'nn_pred'] = (dg0 - dg2).values[0]
    return data