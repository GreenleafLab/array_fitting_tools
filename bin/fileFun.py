import os
import sys
import numpy as np
import pandas as pd
import pickle
import itertools


def loadFile(filename):
    """ Find extension and return loaded file. """
    ext = os.path.splitext(filename)[-1]
    
    if ext == '.pkl':
        return pd.read_pickle(filename)

    if ext == '.CPseq':
        return _loadCPseq(filename)
    
    if ext == '.unique_barcodes':
        return _loadUniqueBarcodes(filename)
    
def _loadCPseq(filename):
    """ Return CPseq file. """
    return pd.read_table(filename, header=None,
                         names=['clusterID', 'filter',
                                'read1_seq', 'read1_quality',
                                'read2_seq', 'read2_quality',
                                'index1_seq','index1_quality',
                                'index2_seq', 'index2_quality'])

def _loadUniqueBarcodes(filename):
    """ Return unique barcodes file. """
    return pd.read_table(filename