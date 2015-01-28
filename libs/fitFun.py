import sys
import os
import time
import re
import argparse
import subprocess
import numpy as np
import pandas as pd

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
    