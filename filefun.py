"""
Date: 9/9/14
Author: Sarah Denny

This module is intended as a library to load files important for RNA array data.
"""


# import necessary for python
import numpy as np


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