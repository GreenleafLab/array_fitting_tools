#!/usr/bin/env python

# Methods for plotting binding curves by variant number, etc
# ---------------------------------------------
#
#
# Sarah Denny
# December 2014

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
import pandas as pd
import datetime
from hjh.helix import Helix
from hjh.junction import Junction
import variantFun
import IMlibs
import itertools

def allThreeByThreeSeqs(junctionMotif, cutOffNumber=None):
    if cutOffNumber is None: cutOffNumber = 64
    junction = Junction(junctionMotif)
    
    # take subset of junctions spanning 64 junctions. 
    if junction.howManyPossibilities() > cutOffNumber:
        subsetIndex = np.around(np.linspace(0, junction.howManyPossibilities()-1, cutOffNumber)).astype(int)
        junction.sequences = junction.sequences[subsetIndex]
    
    return ['%s_%s'%(seq[0], seq[1]) for seq in junction.sequences]
        

def allThreeByThreeVariants(variant_table, helix_context=None, offset=None, shorter_helix = None):
    junctionMotifMat = np.array([[('',), ('B2',), ('B2', 'B2'), ('B2', 'B2', 'B2')],
                                 [('B1',),  ('M',), ('B2', 'M') , ('B2', 'B2', 'M')],
                                 [('B1', 'B1'), ('M',  'B1'), ('M', 'M'), ('B2', 'M',  'M' )],
                                 [('B1', 'B1', 'B1'), ('M',  'B1', 'B1'), ('M',  'M', 'B1'), ('M','M','M')]])
    
    allvariants = {}; seqs = {}
    for i in range(4):
        allvariants[i] = {}
        seqs[i] = {}
    if helix_context is None: helix_context = 'wc'
    if offset is None: offset = 1
    for i, j in itertools.product(range(4), range(4)):
        allvariants[i][j] = [variantFun.findVariantNumbers(variant_table.seqinfo,
                                                  {'junction_sequence': seq,
                                                   'helix_context': helix_context,
                                                   'offset': offset,
                                                   'loop':'goodLoop',
                                                   'receptor':'R1'})
                             for seq in allThreeByThreeSeqs(junctionMotifMat[i][j]) ]
        seqs[i][j] = [seq for seq in allThreeByThreeSeqs(junctionMotifMat[i][j]) ]
        if shorter_helix is not None:
            for k, variants in enumerate(allvariants[i][j]):
                if shorter_helix == 'helix_one':
                    index = variant_table.seqinfo.loc[variants].helix_one_length <= variant_table.seqinfo.loc[variants].helix_two_length
                elif shorter_helix == 'helix_two':
                    index = variant_table.seqinfo.loc[variants].helix_one_length >= variant_table.seqinfo.loc[variants].helix_two_length
                allvariants[i][j][k] = variants[index.values]
    return pd.DataFrame.from_dict(allvariants), pd.DataFrame.from_dict(seqs)


    