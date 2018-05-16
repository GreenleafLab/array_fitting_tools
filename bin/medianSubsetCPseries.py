import argparse
import pandas as pd
import numpy as np
import os
from fittinglibs import fileio, processing

#set up command line argument parser
parser = argparse.ArgumentParser(description='bin time series by time')

parser.add_argument('-cs', '--cpseries', metavar="CPseries.pkl",
                   help='CPseries file containining the time series information')
parser.add_argument('-t', '--tile_file', metavar="CPtiles.pkl",
                   help='CPtiles file containining the tile per cluster')
parser.add_argument('-td', '--time_dict', metavar="timeDict.p",
                   help='file containining the timing information per tile')
parser.add_argument('--tile', metavar="NNN", default='001',
                   help='tile you wish to subset. Default="001"')
parser.add_argument('-o', '--out_file', 
                   help='basename of out file. default is basename of cpseries file')

group = parser.add_argument_group('optional arguments')
group.add_argument('-an', '--annotated_clusters', metavar="CPannot.pkl",
                   help='annotated cluster file. Supply if you wish to take medians per variant.'
                   'If not provided, script will not take medians, otherwise it will.')

group.add_argument('-nz', '--remove_zero_point', action="store_true",
                   help='flag if you wish to remove the zero point. This is useful if the zero '
                   'point time does not reflect kinetics, i.e. if it was taken before flow.')

if __name__ == '__main__':
    # parse command line
    args = parser.parse_args()
    bindingSeriesFile = args.cpseries
    tileFile = args.tile_file
    timeDeltaFile = args.time_dict
    annotatedClusterFile = args.annotated_clusters
    tile_to_subset = args.tile
    
    # load files
    if args.out_file is None:
        outFile = fileio.stripExtension(bindingSeriesFile)
    else:
        outFile = args.out_file 
    bindingSeries = fileio.loadFile(bindingSeriesFile)
    timeDict = fileio.loadFile(timeDeltaFile)
    tileSeries = fileio.loadFile(tileFile)
    
    # look only at clusters in tile
    print 'Only looking at clusters in tile %s...'%tile_to_subset
    index = tileSeries==tile_to_subset
    bindingSeries = bindingSeries.loc[index].copy()
    tileSeries = tileSeries.loc[index].copy()
    times = timeDict[tile_to_subset] 
    
    # remove zero point if given
    if args.remove_zero_point:
        print 'Removing minimum time point...'
        print '\t min time = %4.2f'%np.min(times)
        i = pd.Series(times).idxmin()
        times = times[:i] + times[i+1:]
        bindingSeries = pd.concat([bindingSeries.iloc[:, :i], bindingSeries.iloc[:, i+1:]], axis=1)
    
    # resave time dict, enforcing starting from 0
    timeDict = {tile_to_subset:times - np.min(times)}
        
    # if annotated clusters are provided, take median
    if annotatedClusterFile is not None:
        print 'Taking median fluorescence value per variant...'
        annotatedCluster = fileio.loadFile(annotatedClusterFile)
        grouped = pd.concat([annotatedCluster, bindingSeries], axis=1).groupby('variant_number')
        bindingSeries = grouped.median()   
        tileSeries = pd.Series(tile_to_subset, index=bindingSeries.index, name='tile')
        countSeries = grouped.count().max(axis=1)
        fileAppend = tile_to_subset + '.median'
    else:
        fileAppend = tile_to_subset
        countSeries = pd.Series(1, index=bindingSeries.index)
    
    tileSeries.to_pickle('%s.%s.CPtiles.pkl'%(outFile, fileAppend))
    bindingSeries.to_pickle('%s.%s.CPseries.pkl'%(outFile, fileAppend))
    countSeries.to_pickle('%s.%s.CPcounts.pkl'%(outFile, fileAppend))
    processing.saveTimeDeltaDict(os.path.join(os.path.dirname(outFile), 'rates.timeDict.%s.p'%tile_to_subset), timeDict)

