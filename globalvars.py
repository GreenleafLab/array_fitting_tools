"""
Class to store parameters in common across analyses
"""

# import necessary for python
import numpy as np
import os


class fitParameters():
    """
    Class to store parameters in common across fitting analyses
    """
    def __init__(self):
        
        self.maxAnyFitParameter = 1E10  # if fit parameters is above this number, set it to inf
        self.maxLifetime = 10*1440  # ten days, in minutes
        self.minAmplitude = 0
        self.minLifetime = 0
        self.timestamps = np.array([0.0, 0.559366666667, 1.11926666667, 1.69191666667,
                                    2.25051666667, 3.18255, 3.74843333333, 4.30493333333,
                                    4.8513, 5.41561666667, 6.41718333333, 6.9815, 7.54583333333,
                                    8.1151, 8.66561666667, 10.8906166667, 11.4536333333,
                                    12.0104166667, 12.56275, 13.1432166667])
        
    def maxAmplitude(self, expectedMaxAmplitudes):
        """
        use this function to ask what the max amplitude should be during the off-Rate fit.
        'expectedMaxAmplitudes' can be the vector of the quantified clusters of first image in the off-rate series.
        """
        criteria = np.all((np.logical_not(np.isnan(expectedMaxAmplitudes)),
                       np.isfinite(expectedMaxAmplitudes)), axis=0)
        
        amplitudeCutoff = np.max(expectedMaxAmplitudes[criteria]) # let it be twice the expected maximum value

        return amplitudeCutoff
        
    
