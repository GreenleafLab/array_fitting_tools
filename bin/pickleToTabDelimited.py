#!/usr/bin/env python
""" Load pkl file, save tab-delimited file.

 Sarah Denny """


import argparse
import numpy as np
import pandas as pd
import sys
import pickle

# the following code is to stop a python "IOerror" exception being raised when you
# terminate i.e. with the unix "head" command.
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL) 

#set up command line argument parser
parser = argparse.ArgumentParser(description='master script for processing data')
parser.add_argument('-i','--input', metavar=".pkl",
                    help='input pkl file')
parser.add_argument('-s','--save', default=False, action="store_true",
                    help='save instead of just printing output')
parser.add_argument('-o','--outfile',
                    help='outfile to save to. default is to strip input of '
                    '".pkl" extension.')
args = parser.parse_args()

# laod pickled file
a = pd.read_pickle(args.input)

# two options: either save directly: this requires the input file to have extension '.pkl'
if args.save:
    # strip the file of '.pkl' extension and save
    if args.outfile is None:
        if args.input.find('.pkl') != len(args.input)-4:
            print "No '.pkl' extension found. Please specify -o, --outfile option."
            sys.exit()
        # if '.pkl' extension was found, save to file with extension removed.
        args.outfile = args.input[:-4]
    print "Saving output to file %s"%args.outfile
    a.to_csv(args.outfile, sep='\t')
else:
    # print to stdout in chunks
    chunksize = 100 # chunksize of 100 is quick to print
    for indices in np.array_split(a.index, chunksize):
        print a.loc[indices].to_string(float_format=lambda x: '%4.4f' % x)


