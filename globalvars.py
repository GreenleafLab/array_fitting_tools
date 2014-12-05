#!/usr/bin/env python

# Author: Sarah Denny, Stanford University 

# Parameters to analyze the images from RNA array experiment 11/11/14


##### IMPORT #####
import numpy as np

class Parameters():
    
    """
    stores file names
    """
    def __init__(self):
        
        # save the units of concentration given in the binding series
        self.concentration_units = 'nM'
        
        # initial binding curve settings
        self.parameter_names = ['f_max', 'kd']
        self.fmax_min = 0.4
        self.fmax_max = 2.5
        self.fmax_initial = 1
        
        self.kd_min = 0
        self.kd_max = 1e6
        self.kd_initial = 400 #nM
        