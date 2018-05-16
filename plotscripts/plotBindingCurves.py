#!/usr/bin/env python
#
# Sarah Denny
# July 2015

##### IMPORT #####
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fittinglibs import fileio, initfits, plotting

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='bootstrap fits')
parser.add_argument('-i', '--ids', nargs='+', required=True,
                   help='variant(s) or clusterId(s) to plot')

group = parser.add_argument_group(description='Arguments related to per-variant info. If you want '
                                  'to plot per-variant results, give either the variant params object, '
                                  'or the cpannot, cpvariant, and cpseries file. Otherwise, if you want '
                                  'to plot per-cluster results, give the cpfitted and cp series file.')
group.add_argument('-vp', '--variantparams', 
                   help='file containing the variant params (variantParams.p)')
group.add_argument('-fr', '--fitting_results', 
                   help='either per variant (CPvariant) or per cluster (CPfitted) fitting result')

group.add_argument('-cs', '--cpseries', 
                   help='CPseries file containining the fluorescence series')
group.add_argument('-an', '--annotated_clusters', 
                   help='annotated cluster file. Supply if you wish to plot a variant.'
                   'Otherwise assumes single cluster fits, unless variant params object is provided.')

group = parser.add_argument_group(description='Arguments related to fitting. If the variant params object is not '
                                  'provided (above), give either fit params object or concenetrations and function name')
group.add_argument('-fp', '--fitparams', 
                   help='file containining the fit parameters (fitParams.p)')
group.add_argument('-c', '--concentrations', metavar=".txt",
                   help='file containining the concentrations')
group.add_argument('--func', default='binding_curve',
                   help='name of function used to fit')

group = parser.add_argument_group(description='Optional additional arguments.')
group.add_argument('-out', '--out_dir', default='./',
                   help='output directory. default is current directory')
group.add_argument('--annotate', action="store_true",
                   help='flag if you want the plot annotated')
group.add_argument('--plotinit', action="store_true",
                   help='flag if you want to plot the initial fit')
group.add_argument('--initial_results', 
                   help='initial per-variant results. this is totally optional.')
group.add_argument('--fmaxdist', 
                   help='fmax distribution. this is totally optional.')

def make_plot(variantParams, idx, annotate=False, plotinit=False):
    """Make the binding series plot."""
    plt.figure(figsize=(3,3)); plt.xscale('log')
    variantParams.plot_specific_binding_curve(idx, annotate=annotate)
    plt.xlabel('concentration')
    plt.ylabel('fluoresence')
    plt.subplots_adjust(left=0.25, bottom=0.2, right=0.95, top=0.95)
    plotting.fix_axes(plt.gca())
    
    if plotinit:
        try:
            variantParams.plot_init_binding_curves(variantParams.results_all.loc[idx])
        except KeyError:
            pass
     

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.variantparams:
        # load only the variant params if given
        variantParams = fileio.loadFile(args.variantparams)
    else:
        # make variant params
        if args.fitparams:
            # load fit params if given
            fitParams = fileio.loadFile(args.fitparams)
            try:
                getattr(fitParams, 'func_name')
            except AttributeError:
                fitParams.func_name = args.func
        else:
            # make fit params using --concentrations and --func
            concentrations = fileio.loadFile(args.concentrations)
            fitParams = initfits.FitParams(args.func, concentrations)
        
        # load binding series and fitted vals
        binding_series = fileio.loadFile(args.cpseries)
        fitted_vals = fileio.loadFile(args.fitting_results)
        
        # load annotations if given, otherwise make a dummy annotated cluster file to plot individual clusters
        if args.annotated_clusters:
            annotated_clusters = fileio.loadFile(args.annotated_clusters)
        else:
            annotated_clusters = pd.DataFrame(binding_series.index.tolist(), index=binding_series.index, columns=['variant_number'])
        variantParams = initfits.MoreFitParams(fitParams, binding_series=binding_series, annotated_clusters=annotated_clusters, results=fitted_vals)
        
    # go through variants and plot
    for idx in args.ids:
        try:
            idx = int(idx)
        except ValueError:
            pass
        
        out_file = os.path.join(args.out_dir, 'binding_curve.%s.pdf'%str(idx))
        make_plot(variantParams, idx, annotate=args.annotate, plotinit=args.plotinit)
        plt.savefig(out_file)   
