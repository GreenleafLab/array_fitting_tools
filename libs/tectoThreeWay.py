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
        loop, length, topology, seq, pos = idx
        bindingKey = returnBindingKey(loop, length, topology)
        if loop == 'GGAA_UUCG':
            # then you want the fraction that lines up
            fracKey = returnFracKey(topology, seq, pos)
        else:
            # then you want the permuted version where 1 becomes 0, 0 becomes 2
            fracKey = returnFracKey(topology, seq, pos-1)
        
        # two terms to energy function
        term1 = parvals[bindingKey]   
        term2 = parameters.RT*np.log(parvals[fracKey])
        
        diff.loc[idx] = term1 + term2 
    
    if return_pred:
        return diff.astype(float)
    elif return_weighted:
        return ((diff - y.dG)*y.weight).astype(float)
    else:
        return (diff - y.dG).astype(float)

def returnBindingKey(loop, length, topology):
    if np.in1d(list(topology),  'A').sum() == 0:
        unique_topology = '_'
    elif np.in1d(list(topology),  'A').sum() == 1:
        unique_topology = 'A'
    else:
        unique_topology = 'AA'
    return 'bind_%s_%d_%s'%(loop.split('_')[0], length, unique_topology)

def returnFracKey(topology, seq, pos):
    junction_type = 3 # it's a three way junction, so three permutations exist
    g = deque(np.arange(junction_type))
    g.rotate(pos)
    
    seq = list(seq)
    topology = topology.split('_')
    
    key = ('_'.join([seq[i] for i in list(g)]) + 'g' + 
           '_'.join([topology[i] for i in list(g)]))
    return key

def returnSeq(seq, pos):
    junction_type = 3 # it's a three way junction, so three permutations exist
    g = deque(np.arange(junction_type))
    g.rotate(pos)
    
    seq = list(seq)
    return [seq[i] for i in list(g)]

def fitThreeWay(y, weight=None, force=None, to_include=None):
    if weight is None: weight = False
    if force is None: force = False
    
    loops = subsetIndex(y, 'loop', to_include)
    lengths = subsetIndex(y, 'length', to_include)
    topologies = subsetIndex(y, 'topology', to_include)
    seqs = subsetIndex(y, 'seq', to_include)
    y = subsetData(y, to_include)
    # store fit parameters in class for fitting
    params = Parameters()
    for loop, length, topology in itertools.product(loops, lengths, topologies):

        key = returnBindingKey(loop, length, topology)
        if key in params.keys():
            print '%s already in dict'%key
        else:
            params.add(key, value=-9, min = -16, max = -4)
    
    # now store state fractions
    for topology, seq in itertools.product(topologies, seqs):
        # if force is true, don't vary these parameters
        if force:
            vary = False
        else:
            vary = True
        
        # save params for the first two circularly permuted seq/topology
        for pos in [0,1]:
            params.add(returnFracKey(topology, seq, pos),
                       value=1./3, min=0, max=1, vary=vary)
        
        # then make the third 1 - the sum of the other two
        params.add(returnFracKey(topology, seq, 2),
                   expr='1-%s-%s'%(returnFracKey(topology, seq, 0),
                                   returnFracKey(topology, seq, 1)))
    
    # now fit
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
                                                 topologies=None, loops=None,
                                                 seqs=None):
    if junction_type is None:
        junction_type = 3
        
    if lengths is None:
        lengths = np.unique(subtable.helix_one_length)
    
    if topologies is None:
        topologies = ['__', 'A__', 'AA__', '_A_', '_AA_', '__A', '__AA']
    
    if loops is None:
        loops =['GGAA_UUCG', 'UUCG_GGAA']
        
    # find all seqeunce possibilities
    if seqs is None:
        seqs = hjh.junction.Junction(('W', 'W', 'W')).sequences.side1
    
    y = {}
    for loop in loops:
        print loop
        y[loop] = {}
    
        for length in lengths:
            print '\t%d'%length
            y[loop][length] = {}
            
            for topology in topologies:
                print '\t\t%s'%topology
                y[loop][length][topology] = {}
                insertions = topology.split('_')
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
                        cols = ['dG', 'variance', 'weight', 'seq', 'topology', 'variant', 'correct']
                        cols = ['dG', 'variance', 'weight',  'variant', 'correct']
                        y[loop][length][topology][seq] = pd.DataFrame(index=np.arange(junction_type),
                                                      columns=cols)
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
                                y[loop][length][topology][seq].loc[idx, 'dG'] = subtable.loc[index].dG.values[0]
                                y[loop][length][topology][seq].loc[idx, 'variance'] = (((subtable.loc[index].eminus + subtable.loc[index].eplus)/2)**2).values[0]
                                y[loop][length][topology][seq].loc[idx, 'weight'] = (1/(subtable.loc[index].eminus + subtable.loc[index].eplus)).values[0]
                                y[loop][length][topology][seq].loc[idx, 'variant'] = index.loc[index].index[0]
                                try:
                                    y[loop][length][topology][seq].loc[idx, 'correct'] = (subtable.loc[index].ss_correct=='True').values[0]
                                except:
                                    pass
                            #y[loop][length][topology][seq].loc[idx, 'seq'] = '_'.join(seq_list)
                            #y[loop][length][topology][seq].loc[idx, 'topology'] = '_'.join(ins)
                            # rotate g
                            g.rotate(1)
                y[loop][length][topology] = pd.concat(y[loop][length][topology], names=['seq'])
            y[loop][length] = pd.concat(y[loop][length], names=['topology'])
        y[loop] = pd.concat(y[loop], names=['length'])
    return pd.concat(y,  names=['loop'])
    

def findDataMat(subtable, y, results):
    parameters = fittingParameters()
    data = pd.concat([subtable.loc[y.variant].dG,
                      pd.Series(subtable.loc[y.variant+1].dG.values, index=y.variant)],
        axis=1, keys = ['fit', 'not_fit'])
    data.index = y.index
    #data.loc[:, 'seq'] = y.seq
    data.loc[:, 'ddG_pred'] = np.nan
    data.loc[:, 'dG_conf'] = np.nan
    data.loc[:, 'dG_bind'] = np.nan
    data.loc[:, 'nn_pred'] = np.nan
    
    for idx in data.index:

        loop, length, topology, seq, pos = idx
        bindingKey = returnBindingKey(loop, length, topology)
        
        if returnFracKey(topology, seq, pos) in results.index:
            if loop == 'GGAA_UUCG':
                # then you want the fraction that lines up
                frac0 = results.loc[returnFracKey(topology, seq, pos)]
                frac1 = results.loc[returnFracKey(topology, seq, pos-1)]
            else:
                # then you want the permuted version where 1 becomes 0, 0 becomes 2
                frac0 = results.loc[returnFracKey(topology, seq, pos-1)]
                frac1 = results.loc[returnFracKey(topology, seq, pos)]
            
    
            data.loc[idx, 'ddG_pred'] = parameters.RT*np.log(frac0/frac1)
            data.loc[idx, 'dG_conf'] = parameters.RT*np.log(frac0)
        
        if bindingKey in results.index:
            data.loc[idx, 'dG_bind'] = results.loc[bindingKey]
            
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
        loop, length, topology, seq, pos = idx
        joined_seq = returnSeq(seq, pos)
        v1 = pd.Series([joined_seq[0], complements[joined_seq[0]]], index = ['x', 'y'])
        v2 = pd.Series([joined_seq[1], complements[joined_seq[1]]], index = ['x', 'y'])
        v3 = pd.Series([joined_seq[2], complements[joined_seq[2]]], index = ['x', 'y'])
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

def subsetData(data, to_include=None):
    # to_include must be a dict with names = index names and values equal to the
    # list of possible values
    if to_include is None:
        return data
    
    data = data.copy()
    for key, value in to_include.items():
        index = pd.Series(np.in1d(data.reset_index(level=key).loc[:, key], value),
                          index = data.index)
        data = data.loc[index]
    return data


def subsetIndex(data, key, to_include=None):
    if to_include is None:
        return np.unique(data.reset_index(level=key).loc[:, key])
    
    if key in to_include.keys():
        return to_include[key]
    else:
        return np.unique(data.reset_index(level=key).loc[:, key])
