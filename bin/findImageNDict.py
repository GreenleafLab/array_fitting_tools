#!/usr/bin/env python
""" Find which images were actually fit per tile.

Sarah Denny """

##### IMPORT #####
import numpy as np
import pandas as pd
import os
import argparse
import pickle

### MAIN ###
#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap on off rate fits')
parser.add_argument('-fd', '--fluor_dir',
                   help='directory of cp fluor files')
parser.add_argument('-td', '--tif_dir', 
                   help='directory of tif files')

group = parser.add_argument_group('additional option arguments')
group.add_argument('-out', '--out_file', 
                   help='output filename.')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = 'image_n_dict.p'
        
    # make sure image_ns are right
    cpfluor_files = (
        pd.concat([pd.DataFrame(int(filename[filename.find('tile')+4:filename.find('tile')+7]),
                                columns=['tile'],
                                index=[filename[filename.find('2015'):-8]])
               for filename in os.listdir(args.fluor_dir)
               if os.path.splitext(filename)[1] == '.CPfluor']))
    
    tif_files = (
        pd.concat([pd.DataFrame(int(filename[filename.find('tile')+4:filename.find('_green')]),
                                columns=['tile'],
                                index=[filename[filename.find('2015'):-4]])
               for filename in os.listdir(args.tif_dir)
               if os.path.splitext(filename)[1] == '.tif']))
    
    final = pd.concat([tif_files, cpfluor_files], axis=1, ignore_index=True)
    final.columns = ['original', 'fit']
    final.sort_index(inplace=True)
    
    imageNDict = {}
    for tile in np.unique(final.original):
        if tile < 10:
            format_tile = '00%d'%tile
        else:
            format_tile = '0%d'%tile
        
        subtile = final.loc[final.original==tile].copy()
        subtile.loc[:, 'imagen'] = np.arange(len(subtile))
        imageNDict[format_tile] = subtile.dropna(subset=['fit']).imagen.values
        
    with open(args.out_file, "wb") as f:
        pickle.dump(imageNDict, f, protocol=pickle.HIGHEST_PROTOCOL)