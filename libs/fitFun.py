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

class Parameters():
    def __init__(self):
        self.chip_receptor_length = [11, 12]
        self.basePairDict = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}
        self.nonCanonicalBasePairDict = {'U':'G', 'A':'', 'G':'U', 'C':''}
        self.indexDict = {'i':'x', 'j':'y'}

def parseSecondaryStructure(seq, helix_length):
    parameters = Parameters()
    columns=['i', 'j', 'k', 'x', 'y', 'z']
    dibases = pd.DataFrame(index=np.arange(helix_length-1), columns=columns)

    # initial data storage

    [seq_parsed, dot_bracket, energy] = subprocess.check_output("echo %s | RNAfold --noPS"%seq, shell=True).split()
    dot_bracket = dot_bracket[parameters.chip_receptor_length[0]:-parameters.chip_receptor_length[1]]
    seq_parsed = seq_parsed[parameters.chip_receptor_length[0]:-parameters.chip_receptor_length[1]]
    
    loopLoc = dot_bracket.find('(....)')
    loopStart = loopLoc + 1; loopEnd = loopLoc + 5
    
    side1, side2 = dot_bracket[:loopStart], dot_bracket[loopEnd:][::-1]
    side1_seq, side2_seq = seq_parsed[:loopStart], seq_parsed[loopEnd:][::-1]
    success = True
    i = 0
    j = 0
    numBasePairs = 0
    dibases.loc[:,:] = ''
    try:
        while numBasePairs < helix_length-1:
            if side1[i] == '(' and side2[j] == '.':
                dibases.loc[numBasePairs-1, 'z'] += side2_seq[j]
                j+=1
            if side1[i] == '.' and side2[j] == ')':
                dibases.loc[numBasePairs-1, 'k'] += side1_seq[i]
                i+=1
            if (side1[i] == '(' and side2[j] == ')') or (side1[i] == '.' and side2[j] == '.'):
                dibases.loc[numBasePairs, 'i'] = side1_seq[i]
                dibases.loc[numBasePairs, 'x'] = side2_seq[j]
                i+=1
                j+=1
                numBasePairs+=1
            dibases
        i=1; j=1
        numBasePairs = 0
        while numBasePairs < helix_length-1:
            if side1[i] == '(' and side2[j] == '.': j+=1
            if side1[i] == '.' and side2[j] == ')': i+=1
            if (side1[i] == '(' and side2[j] == ')') or (side1[i] == '.' and side2[j] == '.'):
                dibases.loc[numBasePairs, 'j'] = side1_seq[i]
                dibases.loc[numBasePairs, 'y'] = side2_seq[j]
                i+=1
                j+=1
                numBasePairs+=1

    except: success = False
    return dibases, success

def interpretDibase(dibase):
    parameters = Parameters()
    params = ['bp_break', 'nc', 'insertions']
    index  = ['i', 'j', 'k', 'z']
    table = pd.DataFrame(data=np.zeros((len(index), len(params))), index=index, columns=params)
    for idx in ['i', 'j']:
        if parameters.basePairDict[dibase[idx]] != dibase[parameters.indexDict[idx]]:
            table.loc[idx, 'bp_break'] = 1
        if parameters.nonCanonicalBasePairDict[dibase[idx]] == dibase[parameters.indexDict[idx]]:
            table.loc[idx, 'nc'] = 1
    for idx in ['k', 'z']:
        if dibase[idx] != '':
            table.loc[idx, 'insertions'] = len(dibase[idx])
    return table


def convertDibasesToParams(dibases):
    indexParam = np.arange(len(dibases))
    pieces = {}
    for i in indexParam:
        pieces[i] = interpretDibase(dibases.loc[i])
    table = pd.concat(pieces, axis=0)
    vec = np.hstack([np.hstack([table.loc[idx, 'i'].values[:2],
                                     table.loc[idx, 'j'].values[:2],
                                     table.loc[idx, 'k'].values[-1],
                                     table.loc[idx, 'z'].values[-1]])
                     for idx in indexParam])
    return vec, table
        
def multiprocessParametrization(variant_table, helixLengthTotal, indx):
    seq = variant_table.loc[indx, 'sequence']
    dibases, success = parseSecondaryStructure(seq, helixLengthTotal)
    
    # Make table
    if success:
        vec, table = convertDibasesToParams(dibases)
    else:
        vec = np.ones(6*(helixLengthTotal-1))*np.nan
    return vec

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

def parseJunction(variant_table, variant_set):
    
    return

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




    