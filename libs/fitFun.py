import sys
import os
import time
import re
import argparse
import subprocess
import numpy as np
import pandas as pd
import variantFun
import matplotlib.pyplot as plt
import itertools
import collections
from statsmodels.stats.weightstats import DescrStatsW

class Parameters():
    def __init__(self):
        self.chip_receptor_length = [11, 12]
        self.basePairDict = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}
        self.nonCanonicalBasePairDict = {'U':'G', 'A':np.nan, 'G':'U', 'C':np.nan}
        self.indexDict = {'i':'x', 'j':'y'}
        self.transitionDict   = {'A':'G', 'G':'A', 'C':'U', 'U':'C'}
        self.max_measurable_dG = -6
        self.helix_length = 12
        self.goodLoop = 'GGAA'
        self.numParamsDict = {'insertions_k':3,
                     'insertions_z':3,
                     'seq_change_i':2,
                     'seq_change_x':2,
                     'loop':1,
                     'bp_break':1,
                     'nc':1}

def getNumCategories(table):
    locations = [name for name in table]
    d = collections.defaultdict(list)
    for param, cat, loc in locations:
        d[param].append(cat)
    for param in d.keys():
        d[param] = len(np.unique(d[param]))
    return pd.Series(d)
    

def getWildtypeDibases(variant_table, indx_wt):
    parameters = Parameters()
    seq = variant_table.loc[indx_wt, 'sequence']
    start = parameters.chip_receptor_length[0]
    tecto_length = 24
    newseq = seq[:start] + 'UC' + seq[start:(tecto_length+start)] + 'GA' + seq[tecto_length+start:]
    # wildtype sequence is the WC cnostruct, plus a UA, CG basepair to have comparable changes in 12 bp constructs
    dibases, loop_seq, success = parseSecondaryStructure(newseq)
    return dibases
    
def addBaseToDibase(dibases, insertion_point, key, base, max_insertion_point):
    success = True
    if insertion_point <= max_insertion_point and insertion_point >= 0:
        dibases.loc[insertion_point, key] += base
    else:
        success = False
    return dibases, success

def parseSecondaryStructure(seq):
    parameters = Parameters()
    columns=['i', 'j', 'k', 'x', 'y', 'z']
    dibases = pd.DataFrame(index=np.arange(parameters.helix_length-1), columns=columns)

    # initial data storage
    vec = subprocess.check_output("echo %s | RNAfold --noPS"%seq, shell=True).split()
    [seq_parsed, dot_bracket] = vec[:2]
    dot_bracket = dot_bracket[parameters.chip_receptor_length[0]:-parameters.chip_receptor_length[1]]
    seq_parsed = seq_parsed[parameters.chip_receptor_length[0]:-parameters.chip_receptor_length[1]]

    loopLoc = dot_bracket.find('(....)')
    loopStart = loopLoc + 1; loopEnd = loopLoc + 5
    
    side1, side2 = dot_bracket[:loopStart], dot_bracket[loopEnd:][::-1]
    side1_seq, side2_seq = seq_parsed[:loopStart], seq_parsed[loopEnd:][::-1]
    
    # find helix length
    estimated_helix_length = min(len(side1), len(side2))
    offset = 2  # difference between 'loc' of dibase and estimated helix length
    max_insertion_point = estimated_helix_length - offset
    success = True
    i = 0
    j = 0
    numBasePairs = 0

    dibases.loc[:,:] = ''
    try:
        while numBasePairs < estimated_helix_length-1:
            if side1[i] == '(' and side2[j] == '.':
                insertion_point = max_insertion_point-(numBasePairs-1)
                key = 'z'
                base = side2_seq[j]
                dibases, success = addBaseToDibase(dibases, insertion_point, key, base, max_insertion_point)
                if not success: break
                j+=1
    
            if side1[i] == '.' and side2[j] == ')':
                insertion_point = max_insertion_point-(numBasePairs-1)
                key = 'k'
                base = side1_seq[i]
                dibases, success = addBaseToDibase(dibases, insertion_point, key, base, max_insertion_point)
                if not success: break
                i+=1
            if (side1[i] == '(' and side2[j] == ')') or (side1[i] == '.' and side2[j] == '.'):
                insertion_point = max_insertion_point - numBasePairs
                dibases, success = addBaseToDibase(dibases, insertion_point, 'i', side1_seq[i], max_insertion_point)
                if not success: break
                dibases, success = addBaseToDibase(dibases, insertion_point, 'x', side2_seq[j], max_insertion_point)
                if not success: break
                i+=1
                j+=1
                numBasePairs+=1
            dibases
        if success: # if the first part was successful, also parse the other location in each dibase
            i=1; j=1
            numBasePairs = 0
            while numBasePairs < estimated_helix_length-1 :
                if side1[i] == '(' and side2[j] == '.': j+=1
                if side1[i] == '.' and side2[j] == ')': i+=1
                if (side1[i] == '(' and side2[j] == ')') or (side1[i] == '.' and side2[j] == '.'):
                    insertion_point = max_insertion_point - numBasePairs
                    dibases, success = addBaseToDibase(dibases, insertion_point, 'j', side1_seq[i], max_insertion_point)
                    if not success: break
                    dibases, success = addBaseToDibase(dibases, insertion_point, 'y', side2_seq[j], max_insertion_point)
                    if not success: break
                    i+=1
                    j+=1
                    numBasePairs+=1
    except IndexError: success = False

    # everything that's blank is Nan
    dibases.loc[(dibases == '').all(axis=1)] = np.nan
    loop_seq = seq_parsed[loopStart:loopEnd]
    return dibases, loop_seq, success

def interpretDibase(dibase, compare_to=None, success=None):
    if success is None: success = True 
    parameters = Parameters()
    params = ['bp_break', 'nc', 'insertions', 'seq_change']
    index  = ['i', 'j', 'k', 'z', 'x', 'y']
    if success: # i.e. if secondary structure parsing was successful
        table = pd.DataFrame(data=np.zeros((len(index), len(params))), index=index, columns=params)
        for idx in ['i', 'j']:
            if parameters.basePairDict[dibase[idx]] != dibase[parameters.indexDict[idx]]:
                table.loc[idx, 'bp_break'] = 1
            if parameters.nonCanonicalBasePairDict[dibase[idx]] == dibase[parameters.indexDict[idx]]:
                table.loc[idx, 'nc'] = 1
        for idx in ['k', 'z']:
            if dibase[idx] != '':
                table.loc[idx, 'insertions'] = len(dibase[idx])
        if compare_to is not None:
            # compare sequences
            for idx in ['i', 'j', 'x', 'y']:
                if dibase[idx] == compare_to[idx]:
                    table.loc[idx, 'seq_change'] = 0
                elif dibase[idx] == parameters.transitionDict[compare_to[idx]]:
                    table.loc[idx, 'seq_change'] = 1
                else:
                    table.loc[idx, 'seq_change'] = 2
    else: # return NaNs
        table = pd.DataFrame( index=index, columns=params)
    return table


def interpretDibases(dibases, dibases_wt, success=None):
    if success is None: success = True
    # don't use rows of dibases that are all nans
    #dibases.dropna(axis=0, how='all', inplace=True)
    indexParam = np.arange(len(dibases))
    pieces = {}
    for i in indexParam:
        if dibases.loc[i].dropna().empty:
            pieces[i] = interpretDibase(dibases.loc[i], compare_to=dibases_wt.loc[i], success=False)
        else:
            pieces[i] = interpretDibase(dibases.loc[i], compare_to=dibases_wt.loc[i], success=success)
    table = pd.concat(pieces, axis=0)
    return table

def interpretLoop(loop_seq):
    parameters = Parameters()
    if loop_seq == parameters.goodLoop:
        return 1
    else:
        return 0


def convertParamsToMatrix(table, loopSeq):
    indexParam = table.index.levels[0]
    vec = {}
    vec_length = len(indexParam) + 1
    keys = ['bp_break', 'nc', 'insertions_k', 'insertions_z', 'seq_change_i', 'seq_change_x']
    for key in keys: vec[key] = pd.Series(index=np.arange(vec_length))
    
    # first entry for bp break or nc is the 'i' entry, the rest are 'j'
    for key in ['bp_break', 'nc']:
        loc = 0
        vec[key].loc[loc] = table.loc[indexParam[loc], 'j'].loc[key]
        
        # set the rest of them to be the 'j' points, so they are offest by one in indexParam
        for idx, loc in itertools.izip(indexParam, range(1, vec_length)):
            vec[key].loc[loc] = table.loc[idx, 'i'].loc[key]
    
    # insertions 
    for key in ['insertions_k', 'insertions_z']:
        side = key[-1]
        old_key = key.strip('kz_')
        # set the insertions location as the indexParamx, meaning an bp breaking at 0 is immediately to the left of an insertion at 0
        for idx, loc in itertools.izip(indexParam, range(0, vec_length-1)):
            vec[key].loc[loc] = table.loc[idx, side].loc['insertions']
    
    # sequence changes not in insertions
    whichind = {'i':['j', 'i'], 'x':['y', 'x']}
    for key in ['seq_change_i', 'seq_change_x']:
        side = key[-1]
        old_key = key.strip('ix_')
        loc = 0
        vec[key].loc[loc] = table.loc[loc, whichind[side][0]].loc[old_key]
        # set the rest of them to be the 'j' points, so they are offest by one in indexParam
        for idx, loc in itertools.izip(indexParam, range(1, vec_length)):
            vec[key].loc[loc] = table.loc[idx, whichind[side][1]].loc[old_key]
        
    vec['loop'] = pd.Series(interpretLoop(loopSeq))
    return pd.concat(vec)

def makeCategorical(vec):
    parameters = Parameters()
    numParamsDict = parameters.numParamsDict
    all_keys = numParamsDict.keys()    
    newvec = {}
    for key in all_keys:
        newvec[key] = {}
        for i in np.arange(numParamsDict[key]):
            # if categorical, set things equal to category (i+1) to one:
            if numParamsDict[key] == 1: # i.e. not category
                cat = 0
            else:
                cat = i+1
            newvec[key][cat] = pd.Series(index = vec[key].index)
            index = vec[key].dropna().index
            # set remainder to zero
            newvec[key][cat].loc[index] = 0
            
            # set those not equal to zero
            index = vec[key].loc[(vec[key] == i+1).values].index
            newvec[key][cat].loc[index] = 1
        newvec[key] = pd.concat(newvec[key])

    return pd.concat(newvec)

def flattenedHeader(locator, numCategories):
    parameters = Parameters()
    param, cat, loc = locator
    # check if categorical
    if numCategories[param] == 1:
        header = '%s_%d'%(param, loc)
    else:
        header = '%s%d_%d'%(param, cat, loc)
    return header

def flattenMatrix(table):
    locations = [name for name in table]
    newtable = pd.DataFrame(index=table.index)
    numCats = getNumCategories(table)
    for locator in locations:
        newtable[flattenedHeader(locator, numCats)] = table[locator]
    return newtable

def getInteractionTerms(categorical_vec):
    parameters = Parameters()
    compare = {'bp_bp':    [('bp_break',0), ('bp_break', 0)],
               'bp_ins_k1':[('bp_break',0), ('insertions_k', 1)], # 1 insertion
               'bp_ins_k2':[('bp_break',0), ('insertions_k', 2)], # 2 insertions
               'bp_ins_z1':[('bp_break', 0),('insertions_z', 1)],
               'bp_ins_z1':[('bp_break', 0),('insertions_z', 2)],
                }
    new_vec = {}
    offsets = [1] # max distance over which to record interactions
    for key, values in compare.items():
        new_vec[key] = {}
        for offset in offsets:
            new_vec[key][offset] = pd.Series(index=np.arange(parameters.helix_length))
            new_vec[key][offset].iloc[offset:] = (categorical_vec[values[1]].iloc[offset:]*
                                             categorical_vec[values[0]].iloc[:-offset].values)
        new_vec[key] = pd.concat(new_vec[key])
    return pd.concat([categorical_vec, pd.concat(new_vec)])
        
def multiprocessParametrization(variant_table, dibases_wt, indx):
    try:
        dibases, loop_seq, success = parseSecondaryStructure(variant_table.loc[indx, 'sequence'])
        table = interpretDibases(dibases, dibases_wt, success)
        vec = convertParamsToMatrix(table, loop_seq)
        categorical_vec = makeCategorical(vec)
        interaction_vec = getInteractionTerms(categorical_vec)
    except:
        print '%d not successful. Skipping'%indx
        vec = interpretDibases(dibases_wt, dibases_wt, False)
        categorical_vec = makeCategorical(vec)
        interaction_vec = getInteractionTerms(categorical_vec)
    return interaction_vec

def distanceBetweenVariants(variant_table, variant_set):
    deltaG = pd.DataFrame(columns=['median', 'plus', 'minus'], index=[8, 9, 10, 11, 12], dtype=float)
    for length in deltaG.index:
        index = variant_table.seqinfo.total_length==length
        variant_subtable = variant_table.loc[index]
        vs1 = variant_subtable.affinity.loc[variant_set[0]].dropna(axis=0, how='all')
        vs2 = variant_subtable.affinity.loc[variant_set[1]].dropna(axis=0, how='all')
        
        deltaG.loc[length, 'median'] = np.mean(vs2['dG']) - np.mean(vs1['dG'])
        
        deltaG.loc[length, 'plus']   = np.sqrt(np.power(np.sqrt(np.sum(np.power(vs1['dG_ub'] - vs1['dG'], 2))), 2) +
                                               np.power(np.sqrt(np.sum(np.power(vs2['dG_ub'] - vs2['dG'], 2))), 2) )
        
        deltaG.loc[length, 'minus']   = np.sqrt(np.power(np.sqrt(np.sum(np.power(vs1['dG'] - vs1['dG_lb'], 2))), 2) +
                                                np.power(np.sqrt(np.sum(np.power(vs2['dG'] - vs2['dG_lb'], 2))), 2) )    
    return deltaG

def correlationBetweenVariants(variant_table, variants1, variants2):
    parameters = Parameters()
    deltaG = pd.DataFrame(columns=[1, 2, 'weights'], index=[8, 9, 10, 11, 12], dtype=float)
    for length in deltaG.index:
        index = variant_table.seqinfo.total_length==length
        variant_subtable = variant_table.loc[index]
        try:
            vs1 = variant_subtable.affinity.loc[variants1].dropna(axis=0, how='all')
            vs2 = variant_subtable.affinity.loc[variants2].dropna(axis=0, how='all')
        except KeyError:
            vs1 = pd.DataFrame(columns=[name for name in variant_subtable.affinity])
            vs2 = pd.DataFrame(columns=[name for name in variant_subtable.affinity])
        # assuming vs1 and vs2 are only one variant each
        if not vs1.empty:
            vs1 = vs1.iloc[0]
            deltaG.loc[length, 1] = np.min([vs1['dG'], parameters.max_measurable_dG])
        if not vs2.empty:
            vs2 = vs2.iloc[0]
            deltaG.loc[length, 2] = np.min([vs2['dG'], parameters.max_measurable_dG])
        if not (vs1.empty or vs2.empty):
            deltaG.loc[length, 'weights'] = 1./np.sqrt(np.power(vs1['dG_ub'] - vs1['dG'], 2) +
                                                   np.power(vs2['dG_ub'] - vs2['dG'], 2) +
                                                   np.power(vs1['dG'] - vs1['dG_lb'], 2) +
                                                   np.power(vs2['dG'] - vs2['dG_lb'], 2))
    deltaG.dropna(axis=0, how='any', inplace=True)
    toreturn = np.nan
    if len(deltaG) >= 3: # must be able to compare at least three lengths
        dx = DescrStatsW(deltaG.iloc[:,:2].values, weights=deltaG.loc[:,'weights'].values)    
        toreturn = dx.corrcoef[0,1]

    return toreturn

def plotDeltaDeltaG(deltaG):
    xvalues = np.array(deltaG.index, dtype=int)
    yvalues = deltaG['median'].values
    yerr = [deltaG.minus.values, deltaG.plus.values]
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.set_xlim((7.3, 12.7))
    ax.set_ylim((-3, 3))
    ax.set_xlabel('length')
    ax.set_ylabel('ddG (kcal/mol)')
    ax.plot([7, 13], [0, 0], 'k:')
    ax.errorbar(xvalues, yvalues, yerr = yerr, fmt='o-', color='r', ecolor='k')
    ax.tick_params(direction='out', top='off', right='off')
    plt.tight_layout()
    return




    