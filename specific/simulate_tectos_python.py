#!/usr/bin/env python
#
# Run Joe's simulate_tecto code on many sequences, or for many iterations
#

##### IMPORT #####

from joblib import Parallel, delayed
import itertools
import subprocess
import numpy as np
import pandas as pd
import sys
import os
import argparse
import datetime
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt


### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-cf', '--chip_seq_file',  metavar="seqs.txt",
                   help='file containing many sequences of chip piece.')
parser.add_argument('-ff', '--flow_seq_file',  metavar="seqs.txt",
                   help='file containing many sequences of flow piece.')
parser.add_argument('-fs', '--flow_seq', metavar="AGCT",
                    default='CTAGGAATCTGGAAGTACCGAGGAAACTCGGTACTTCCTGTGTCCTAG',
                   help='sequence of flow piece. Default is 10bp Flow: CTAGGAATCTGGAAGTACCGAGGAAACTCGGTACTTCCTGTGTCCTAG')
parser.add_argument('-cs', '--chip_seq',  metavar="AGCT",
                   help='sequence of chip piece.')
parser.add_argument('-nr', '--num_reps',  metavar="N", type=int, default=10,
                    help='number of repetitions. Default is 10')
parser.add_argument('-nc', '--num_cores', metavar="N", type=int,
                    help='number of cores. Default is number of reps')
parser.add_argument('-o', '--out_file', metavar="out.txt",
                    help='save output to this file. default is just to print')
parser.add_argument('-e', '--enforce_find',action="store_true",
                    help='flag if you want to force call until value is found.')

group = parser.add_argument_group('additional arguments for simulate_tectos_devel')
group.add_argument('-s', '--steps', metavar="N", 
                   help='number of iterations. default is 1E6')
group.add_argument('-c', '--cutoff', metavar="N", 
                   help='change the cutoff for counting formation (default 4.5)')
group.add_argument('-t', '--temperature', metavar="N", 
                   help='change the temperature of the simulate (default 298.15K)')
group.add_argument('-sr', '--steric_radius', metavar="N", 
                   help='change the current steric radius  (default 2.2)')
group.add_argument('-wd', '--weight_distance', metavar="N",
                   help='change contribution of distance to cutoff (default 1)')
group.add_argument('-wr', '--weight_rotation', metavar="N", 
                   help='change contribution of rotation to cutoff (default 1)')
group.add_argument('--static', metavar="0", 
                   help='no ensembles  (default 0)')
group.add_argument('-r', '--record', metavar="0", 
                   help='record the distance, rotation and cutoff each step of the final basepair. (default 0)')
group.add_argument('-rf', '--record_file', metavar="filename", 
                   help='change the name of the file to record in (default "test.out" )')

# Define 
FLOW_SECONDARY_STRUCTURE_BY_LENGTH = (
    {44:'((((((....((((((((((....))))))))))....))))))',
    46:'((((((....(((((((((((....)))))))))))....))))))',
    48:'((((((....((((((((((((....))))))))))))....))))))',
    50:'((((((....(((((((((((((....)))))))))))))....))))))',
    52:'((((((....((((((((((((((....))))))))))))))....))))))'})

CHIP_SECONDARY_STRUCTURE_BY_LENGTH = (
    {43:'(((((((..((((((((((....))))))))))...)))))))',
    45:'(((((((..(((((((((((....)))))))))))...)))))))',
    47:'(((((((..((((((((((((....))))))))))))...)))))))',
    49:'(((((((..(((((((((((((....)))))))))))))...)))))))',
    51:'(((((((..((((((((((((((....))))))))))))))...)))))))'})

def send_to_script(str_command):
    # try catch loop for the command
    try:
        # find results of command
        output = subprocess.check_output(str_command, shell=True).strip()
        try:
            # most of the time, output is a single integer (Nbound)
            num_success = int(output)
        except ValueError:
            # sometimes the output is three numbers: distance, rot_distance, score
            num_success = np.array(output.split()).astype(float)
        # command did not fail
        failure = False
    except subprocess.CalledProcessError:
        # command failed
        num_success = None
        failure = True
    return num_success, failure

def send_to_script_smart(str_command, enforce_find=False):
    """ Run the formatted command. """
    if enforce_find:
        # if enforced, it will keep running the command until it is successful
        failure = True
        max_num_iter = 20
        iters = 0
        for i in range(max_num_iter):
            # run command a number of times
            num_success, failure = send_to_script(str_command)
            if not failure:
                # if it worked this iteration, stop iterating
                break
    
    # if not enforced, just run it and return nothing if failed
    num_success, failure = send_to_script(str_command)
    
    if failure:
        return
    else:
        return num_success
    
def simulate_tectos(fseq, cseq, numReps, numCores=None, enforce_find=None, optional_args=None):
    """ Find number of bound states of flow, chip sequence over some iterations. """
    if not (len(fseq) in FLOW_SECONDARY_STRUCTURE_BY_LENGTH.keys() and
            len(cseq) in CHIP_SECONDARY_STRUCTURE_BY_LENGTH.keys()):
        return pd.Series([np.nan])
    
    fss = FLOW_SECONDARY_STRUCTURE_BY_LENGTH[len(fseq)]
    css = CHIP_SECONDARY_STRUCTURE_BY_LENGTH[len(cseq)]
    str_command = 'simulate_tectos_devel -cseq \'' + cseq + '\' -css \'' + css + '\' -fseq \'' + fseq + '\' -fss \'' + fss + '\''
    
    # add options
    if optional_args is not None:
        for key, value in optional_args.items():
            if value is not None:
                str_command += ' %s %s'%(key, value)
        
    print str_command
    if numCores is None:
        # don't parallelize
        successes = [send_to_script_smart(str_command, enforce_find=enforce_find) for i in range(numReps)]
    
    else:
        successes = (Parallel(n_jobs=numCores, verbose=10)
                (delayed(send_to_script_smart)(str_command, enforce_find=enforce_find) for i in range(numReps)))
    
    return pd.Series(successes)

if __name__ == '__main__':
    args = parser.parse_args()
    
    numReps = args.num_reps
    numCores = args.num_cores
    if numCores is None:
        numCores = numReps
        
    # other args
    args_to_joes_args = {'cutoff':          '-c',
                     'weight_rotation': '-wr',
                    'weight_distance':  '-wd',
                    'steric_radius':    '-sr',
                    'steps':            '-s',
                    'static':           '--static',
                    'record':           '-r',
                    'record_file':      '-rf',
                    'temperature':      '-t'}
    joes_args_to_values = {}
    for name, joe_arg in args_to_joes_args.items():
        joes_args_to_values[joe_arg] = vars(args)[name]
    
    # either load chip seqs from file or use given sequence, or exit if neither is given
    if args.chip_seq_file is not None:
        chip_seqs = np.loadtxt(args.chip_seq_file, dtype=str)
    elif args.chip_seq is not None:
        chip_seqs = [args.chip_seq]
    else:
        print "must define chip seq or provide file!"
        sys.exit()        
    
    # either load flow seqs from file or use given sequence, or use default
    if args.flow_seq_file is not None:
        flow_seqs =  np.loadtxt(args.flow_seq_file, dtype=str)
    else:
        flow_seqs = [args.flow_seq]
    
    # optimized parallelization: parallelize by variant if many variants, otherwise parallelize by reps  
    if (numCores <= len(flow_seqs)*len(chip_seqs)):
        successes = (Parallel(n_jobs=numCores, verbose=10)
            (delayed(simulate_tectos)(fseq, cseq, numReps, numCores=None,
                                      enforce_find=args.enforce_find,
                                      optional_args=joes_args_to_values)
             for fseq, cseq in itertools.product(flow_seqs, chip_seqs)))
    else:
        successes = [simulate_tectos(fseq, cseq, numReps, numCores,
                                     enforce_find=args.enforce_find,
                                     optional_args=joes_args_to_values)
            for fseq, cseq in itertools.product(flow_seqs, chip_seqs)]
    
    # print or save output
    successes = pd.concat(successes, axis=1)
    if args.out_file is None:
        print np.mean(successes)
        
    else:
        successes = successes.transpose()
        successes.loc[:, 'average_count'] = successes.mean(axis=1)
        successes.loc[:, 'flow_seq'] = [fseq for fseq, cseq in itertools.product(flow_seqs, chip_seqs)]
        successes.loc[:, 'chip_seq'] = [cseq for fseq, cseq in itertools.product(flow_seqs, chip_seqs)]
        successes.loc[:, 'flow_seq_ind'] = [i for i, cseq in itertools.product(np.arange(len(flow_seqs)), chip_seqs)]
        successes.loc[:, 'chip_seq_ind'] = [j for fseq, j in itertools.product(flow_seqs, np.arange(len(chip_seqs)))]
        successes.to_csv(args.out_file, sep='\t')
        

    sys.exit()
    
    # plot
    ref = pd.read_table('simulations/simulate_tectos//helix_seq.ref.predictions.txt', index_col=0)
    successes = pd.read_table('simulations/simulate_tectos/every_26th_flow_piece.chip_pieces_10.dat', index_col=0)
    
    successes.loc[:, 'ddG'] = -0.582*np.log(successes.average_count/ref.loc[10, 'average_count'])
    ref_vec = -0.582*np.log(ref.average_count/ref.loc[10, 'average_count']).iloc[::2]
    pivoted = successes.pivot(index='flow_seq_ind', columns='chip_seq_ind', values='ddG')
    
    # process the data
    pivoted = pivoted.interpolate(axis=1)
    pivoted = pivoted.bfill(axis=1)
    
    # find correlation distance between all
    Y = ssd.squareform(ssd.pdist(pivoted, 'correlation'))
    corrDist = pd.DataFrame(1-Y, index=pivoted.index, columns=pivoted.index)
    
    # order in some meaningful way
    centroid, label = sc.vq.kmeans2(ddGs, k)
    
    successes.loc[:, 'ref'] = 0
    successes.loc[successes.flow_seq_ind==0, 'ref'] = 1
    grid = sns.FacetGrid(successes.loc[successes.flow_seq_ind<20], col="flow_seq_ind", hue="ref", palette={0:'k', 1:'r'},
                         col_wrap=5, size=1.5, )
    grid.map(plt.plot, "chip_seq_ind", "ddG", marker="o", ms=4)
    grid.fig.tight_layout(w_pad=.1)
    
    # plot
    cluster_list = clusters.value_counts().iloc[:6].index.tolist()
    with sns.axes_style('darkgrid'):
        plt.figure(figsize=(4,3))
        for i in cluster_list:
            plt.plot(pivoted.loc[indices].loc[clusters==i].mean(), 'o-', label=i)
        plt.legend(loc='upper left')
        plt.xlabel('chip seq ind')
        plt.ylabel('predicted ddG (kcal/mol)')
        plt.tight_layout()
        
    # choose most stable variant that follows trend
    i = 10
    ind = 3455
    with sns.axes_style('darkgrid'):
        plt.figure(figsize=(4,3))
        plt.plot(pivoted.loc[indices].loc[clusters==i].mean(), 'ro-', label='mean of cluster %d'%i)
        plt.plot(pivoted.loc[ind], 'ks', label='flow seq %d'%ind)
        plt.legend(loc='upper left')
        plt.xlabel('chip seq ind')
        plt.ylabel('predicted ddG (kcal/mol)')
        plt.tight_layout()
        
    with sns.axes_style('darkgrid'):
        i = 13
        plt.figure(figsize=(4,3))
        plt.plot(pivoted.loc[indices].loc[clusters==i].mean(), 'ro-', label='mean of cluster %d'%i)
        plt.plot(ref_vec, 'ks', label='flow seq WC')
        plt.legend(loc='upper left')
        plt.xlabel('chip seq ind')
        plt.ylabel('predicted ddG (kcal/mol)')
        plt.tight_layout() 
    