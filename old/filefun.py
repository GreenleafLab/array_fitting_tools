"""
Date: 9/9/14
Author: Sarah Denny

This module is intended as a library to load files important for RNA array data.
"""


# import necessary for python
import numpy as np
import os
import pandas as pd


class fluorFile():
    """
    Class to load a CPfluor file. 
    """
    def __init__(self, filename):
        a = np.loadtxt(filename, delimiter=":", dtype=str)
        self.tile = self.getTileInfo(a)
        #fits = np.empty(len(a), dtype={'names':('success', 'amplitude', 'sigma', 'x', 'y'), 'formats':('bool', 'f8', 'f8', 'f8', 'f8')})
        self.success      = a[:, 7].astype(bool)
        self.amplitude = a[:, 8].astype(float)
        self.sigma     = a[:, 9].astype(float)
        self.fitX         = a[:, 10].astype(float)
        self.fitY         = a[:, 11].astype(float)
        self.volume = (2*np.pi*self.amplitude*self.sigma*self.sigma)
    
    def getTileInfo(self, a):
        """
        joins the information about the tile of the Miseq into a single string
        """
        tileInfo = np.array([':'.join(row[:7]) for row in a])
        return tileInfo
    

class seqFile():
    """
    Class to store the filters stored in a CPseq file
    """
    def __init__(self, filename):
        a = np.loadtxt(filename, usecols=(1,), dtype=str)
        self.filter = a
        
    def getIndx(self, filterName):
        """
        return the indices of all rows in cpseq file that have a filter that matches 'filterName'
        """
        indx = np.array([row.find(filterName) > -1 for row in self.filter])
        print 'Number of clusters with filter "%s":\t%d'%(filterName, np.sum(indx))
        return indx
    
def loadCPseqSignal(filename):
    """
    function to load CPseqsignal file with the appropriate headers
    """
    cols = ['tileID','filter','read1_seq','read1_quality','read2_seq','read2_quality','index1_seq','index1_quality','index2_seq', 'index2_quality','allClusterSignal','bindingSeries']
    mydata = pd.read_csv(filename, sep='\t', header=None, names=cols, index_col=False)

    return mydata
    
def loadKdFits(filename):
    """
    function to load CPseqsignal file with the appropriate headers after fitting
    """
    cols = ['tileID',
            'filter',
            'read1_seq',
            'read1_quality',
            'read2_seq',
            'read2_quality',
            'index1_seq',
            'index1_quality',
            'index2_seq',
            'index2_quality',
            'allClusterSignal',
            'bindingSeries',
            'MSE',
            'RSS',
            'amplitude',
            'amplitude_lower',
            'amplitude_upper',
            'df',
            'lifetime',
            'lifetime_lower',
            'lifetime_upper']
    mydata = pd.read_csv(filename, sep='\t', skiprows=1, names=cols, index_col=False)

    return mydata

def findBindingSeriesAsArray(bindingseries):
    """
    takes in the bindingSeries, comma-separated thing from the dataframe object, returns as numpy array
    """
    return np.array([row.replace('NA', 'nan').split(',') for row in bindingseries]).astype(float)

def cpseqFileCols():
    return ['tileID','filter','read1_seq','read1_quality','read2_seq','read2_quality','index1_seq','index1_quality','index2_seq', 'index2_quality']

def getTimeStamp(filename):
    """
    after removing extension of filename ('i.e. 'signal' or 'CPfluor', finds the sequence of numbers after the hyphen, which should be the time stamp)
    hour.minute.second.millisecond. hour goes from 00-24, minute and second from 00-60. 
    """
    return os.path.splitext(filename)[0].split('-')[-1]

def absoluteTime(timestamp):
    """
    starting from a timestamp string, separate into hour, minutes, seconds, and millisecond.
    Add this to get an absolute time in minutes such that you can compare two time stamps.

    """
    (hour, minute, sec, ms) = np.array(timestamp.split('.')).astype(float)
    
    time = hour*60 + minute + sec/(60) + ms/(1000*60)
    return time
    