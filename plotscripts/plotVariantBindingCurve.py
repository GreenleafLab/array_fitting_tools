#!/usr/bin/env python
#
# will find fmax distribution given the single cluster fits on 
# good binders and good fitters.
#
# fmax distribtuion is then enforced for weak binders, allowing
# good measurements even when no reaching saturation.
# 
# Median of cluster fits are bootstrapped to obtain errors on
# fit parameters.
#
# Sarah Denny
# July 2015

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seqfun
import datetime
import IMlibs
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
import plotFun
import findFmaxDist
import fitFun
import fileFun
import variantFun
#plt.rc('text', usetex=True)
### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-f', '--variant_file', required=True, metavar=".CPvariant",
                   help='file with bootstrapped, variant fits')
parser.add_argument('-a', '--annotated_clusters', required=True, metavar=".CPannot.pkl",
                   help='file with clusters annotated by variant number')
parser.add_argument('-b', '--binding_curves', required=True, metavar=".bindingSeries.pkl",
                   help='file containining the binding curve information')
parser.add_argument('-c', '--concentrations', required=True, metavar="concentrations.txt",
                    help='text file giving the associated concentrations')
parser.add_argument('-l', '--lib_characterization', 
                   help='library characterization filename. Use if supplying variant'
                   'sequence of variant index name')
parser.add_argument('-out', '--out_file', 
                   help='output filename. default is variant number + ".pdf"')
parser.add_argument('--annotate', action="store_true",
                   help='flag if you want the plot annotated')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-vn', '--variant_number', metavar="N", type=int,
                   help='index of variant to plot')
group.add_argument('-vs', '--variant_sequence', metavar="AGCT", 
                   help='sequence of variant to plot')
group.add_argument('-vi', '--variant_index_name', metavar="AGCT", 
                   help='name of variant to plot')

if __name__ == '__main__':
    args = parser.parse_args()
    
    variantFilename = args.variant_file
    annotatedClusterFile  = args.annotated_clusters
    bindingCurveFilename  = args.binding_curves


    
    # find variant
    variant_table = fileFun.loadFile(variantFilename)

    if args.variant_number is not None:
        variant = args.variant_number
        if variant_table.loc[variant].numTests < 1:
            print 'Error: no clusters on chip with this variant number!'
            sys.exit()
    else:
        if args.lib_characterization is None:
            print 'Error: need to supply lib characterization if using variant sequence or index'
            sys.exit()
        libChar = pd.read_table(args.lib_characterization)
        results = pd.concat([libChar, variant_table], axis=1)
        
        if args.variant_sequence is not None:
            # find all variants on chip with this sequence
            subset = results.loc[(results.sequence == args.variant_sequence)&(results.numTests >0)]
            if len(subset) > 1:
                print 'Error: more than one variant has clusters on chip and this sequence! %s'%args.variant_sequence
                sys.exit()
            if len(subset) == 0:
                print 'Error: no clusters on chip with this variant sequence! %s'%args.variant_sequence
                sys.exit()
            variant = subset.index[0]
        
        if args.variant_index_name is not None:
            # find all variants on chip with this sequence
            subset = results.loc[(results.name == args.variant_index_name)&(results.numTests >0)]
            if len(subset) > 1:
                print 'Error: more than one variant has clusters on chip and this variant index name! %s'%args.variant_index_name
                sys.exit()
            if len(subset) == 0:
                print 'Error: no clusters on chip with this variant index name! %s'%args.variant_index_name
                sys.exit()
            variant = subset.index[0]            
            
    # load data
    annotatedClusters = fileFun.loadFile(annotatedClusterFile)
    bindingSeries = fileFun.loadFile(bindingCurveFilename)
    concentrations = np.loadtxt(args.concentrations)
    
    affinityData = variantFun.perVariant(variant_table, annotatedClusters, bindingSeries, x=concentrations)
    
    variantClusters = bindingSeries.groupby('variant_number').get_group(variant).iloc[:, 1:]
    
    
    # plot
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    fitParameters = pd.DataFrame(columns=['fmax', 'dG', 'fmin'])
    plotFun.plotFitCurve(concentrations,
                        bindingSeries.groupby('variant_number').get_group(variant).iloc[:, 1:],
                        variant_table.loc[variant],
                        fitParameters, ax=ax)
    # annotate info
    if args.annotate:
        annotationText = ['dG= %4.2f (%4.2f, %4.2f)'%(variant_table.loc[variant].dG,
                                                              variant_table.loc[variant].dG_lb,
                                                              variant_table.loc[variant].dG_ub),
                          'fmax= %4.2f (%4.2f, %4.2f)'%(variant_table.loc[variant].fmax,
                                                            variant_table.loc[variant].fmax_lb,
                                                            variant_table.loc[variant].fmax_ub),
                          'Nclusters= %d'%variant_table.loc[variant].numTests,
                          'pvalue= %.1e'%variant_table.loc[variant].pvalue,
                          'fmax enforced= %d'%variant_table.loc[variant].flag,
                          'average Rsq= %4.2f'%variant_table.loc[variant].rsq,
                          ]
    
        ax.annotate('\n'.join(annotationText), xy=(.05, .95), xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top')
    
    
    if args.out_file is None:
        args.out_file = 'binding_curve.variant_%d.pdf'%variant
        
    plt.savefig( args.out_file)
