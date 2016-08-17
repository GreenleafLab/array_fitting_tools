import os
import sys
import numpy as np
import pandas as pd
import pickle
import itertools
import datetime
from fittinglibs import distribution

def returnFigDirectory():
    return 'figs_%s'%str(datetime.date.today())

def stripExtension(filename):
    return os.path.splitext(filename[:filename.find('.pkl')])[0]

def loadFile(filename):
    """ Find extension and return loaded file. """
    if filename is None:
        print "Error: No filename given!"
        sys.exit()
    ext = os.path.splitext(filename)[-1]
    
    if ext == '.pkl':
        return pd.read_pickle(filename)

    elif ext == '.CPseq':
        return _loadCPseq(filename)
    
    elif ext == '.unique_barcodes':
        return _loadUniqueBarcodes(filename)
    
    elif ext == '.CPfluor':
        return _loadCPFluorFile(filename)
    
    elif ext == '.txt':
        return _loadTextFile(filename)
    
    elif ext == '.p':
        return _loadPickle(filename)
    
    elif ext == '.CPvariant':
        return _loadCPvariant(filename)
    
    elif ext == '.times':
        return np.loadtxt(filename)
    
    elif ext == '.libChar':
        return pd.read_table(filename)
    
    elif ext == '.fitParameters':
        return pd.read_table(filename, index_col=0)
    else:
        print 'Extension %s not recognized. No file loaded.'%ext
    
def _loadCPseq(filename):
    """ Return CPseq file. """
    return pd.read_table(filename, header=None,
                         names=['clusterID', 'filter',
                                'read1_seq', 'read1_quality',
                                'read2_seq', 'read2_quality',
                                'index1_seq','index1_quality',
                                'index2_seq', 'index2_quality'],
                         index_col=0)

def _loadUniqueBarcodes(filename):
    """ Return unique barcodes file. """
    return pd.read_table(filename)

def _loadCPFluorFile(filename):
    a = pd.read_csv(filename,  usecols=range(7, 12), sep=':', header=None, names=['success', 'amplitude', 'sigma', 'fit_X', 'fit_Y'] )
    b = pd.read_csv(filename,  usecols=range(7), sep=':', header=None,  dtype=str)
    a.index = (b.loc[:, 0] + ':' + b.loc[:, 1] + ':' + b.loc[:, 2] + ':' +
               b.loc[:, 3] + ':' + b.loc[:, 4] + ':' + b.loc[:, 5] + ':' + b.loc[:, 6])
    return a

def _loadTextFile(filename):
    try:
        a = np.loadtxt(filename)
    except ValueError:
        a  = np.loadtxt(filename, dtype=str)
    return a

def _loadPickle(filename):
    return pickle.load( open( filename, "rb" ) )

def _loadCPvariant(filename):
    return pd.read_table(filename, index_col=0)
