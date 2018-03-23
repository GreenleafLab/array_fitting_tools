#!/usr/bin/env python
""" Make figures for paper.

Sarah Denny """

##### IMPORT #####
import numpy as np
import pandas as pd
import os
import argparse
import sys
import scipy.spatial.distance as ssd
from fittinglibs import fileio, seqfun

RT = 0.582
### MAIN ###
#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap on off rate fits')
parser.add_argument('-lc', '--library_characterization', required=True,
                   help='file that lists unique variant sequences')
parser.add_argument('-r', '--results', required=True,
                   help='for every library variant, give the measured dG with each flow piece')
parser.add_argument('-re', '--results_error', required=True,
                   help='for every library variant, give the error on the measured dG with each flow piece')
parser.add_argument('-ri', '--results_interp', required=True,
                   help='for every library variant, give the interpolated dG with each flow piece')
parser.add_argument('-out', '--outfile', required=True,
                   help='indication of output file')

def get_distance_from_median_general(data, ref_vec=None, **kwargs):
    """for a set of data points, find the distance from the median of these data points."""
    if ref_vec is None:
        ref_vec = data.median()
    return pd.Series(ssd.cdist(data,  pd.DataFrame([ref_vec]),**kwargs)[:,0], index=data.index)


if __name__ == '__main__':
    args = parser.parse_args()
    
    lib_char = fileio.loadFile(args.library_characterization)
    results = pd.read_table(args.results)
    """Do the analysis to determine neighbors."""
    # get junction subset
    subset_index = lib_char.is_junctionmat
    flow_piece_expts = ['WC9bp', 'WC10bp', 'WC11bp']

    annotations = (lib_char.loc[subset_index].
                   reset_index().rename(columns={'index':'variant_number'}).
                   set_index(['junction_id', 'chip_scaffold']))
    
    # find data and assign index
    results_dict = {}
    for key, filename in zip(['mat', 'mat_err', 'mat_interp'], [args.results, args.results_error, args.results_interp]):
        results = pd.read_table(filename)
        mat = results.loc[subset_index, flow_piece_expts]
        mat.index = annotations.index
        mat = mat.unstack()
        results_dict[key] = mat
    mat_interp = results_dict['mat_interp']
    mat = results_dict['mat']
    mat_err = results_dict['mat_err']
    
    # do PCA
    pca, transformed, loadings = seqfun.doPCA(mat_interp)
    #z = sch.linkage(transformed.iloc[:, :8], method='weighted')
    num_pcs = 6
    data_std = transformed.iloc[:, :num_pcs]/transformed.iloc[:, :num_pcs].std()        

    #interp_info_init = clustering.cluster_knn_general(data_std, threshold=2, max_num_to_average=200, metric='euclidean')
    
    #### REFINE by making sure none deviate by more than three fold the error. #####
    # define which columns to use
    all_cols = mat.columns.tolist()
    cols_with_data = [idx for idx, val in (mat < -7.1).mean().iteritems() if val>0.1]        
    weights = pd.Series(1./len(cols_with_data), index=cols_with_data).loc[all_cols].fillna(0)
    
    interp_info = {}
    threshold = 2 # normalized units, should be relative to std. dev.
    for i, idx in enumerate(mat.index):
        if i % 20 == 0:
            print i

        # find the maximum deviation allowed in any dimension
        max_deviation = mat_err.loc[idx] * 3 
        index = (weights > 0)&(~max_deviation.isnull())
        
        # find the distance between this idx and every other one
        vec = get_distance_from_median_general(data=data_std, ref_vec=data_std.loc[idx], metric='euclidean')
        
        # find subset within threshold
        neighbors = vec.loc[vec < threshold].index
        
        # of this subset, make sure none deviate by more than threshold
        new_neighbors = []
        for idx_neighbor in list(neighbors):
            deviation = np.abs(mat.loc[idx] - mat.loc[idx_neighbor])
            if (deviation < max_deviation).loc[index&(~deviation.isnull())].all():
                new_neighbors.append(idx_neighbor)
                
        interp_info[idx] = set(new_neighbors)
    interp_info = pd.DataFrame(pd.Series(interp_info).rename('neighbors'))
    
    # save
    
    with open(args.outfile, 'w') as f:
        f.write('junction_id\tneighbors\n')
        for idx, neighbors in interp_info.neighbors.iteritems():
            f.write('%d\t%s\n'%(idx, ','.join(['%d'%i for i in neighbors])))

 