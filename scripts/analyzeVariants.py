#!/usr/bin/env python

# Methods for plotting binding curves by variant number, etc
# ---------------------------------------------
#
#
# Sarah Denny
# December 2014

import sys
import os
import time
import re
import argparse
import subprocess
import multiprocessing
import shutil
import uuid
import numpy as np
import scipy.io as sio
import pandas as pd
import variantFun
import IMlibs
parameters = variantFun.Parameters()

#set up command line argument parser
parser = argparse.ArgumentParser(description="master script for the fitting clusters to binding curves pipeline")
parser.add_argument('-cf', '--CPfitted', help='name of CP fitted file')
parser.add_argument('-fd','--CPfluor_dirs_to_quantify', help='text file giving the dir names to look for CPfluor files in', required=True)
parser.add_argument('-n','--num_cores', help='maximum number of cores to use')
parser.add_argument('-gv','--fitting_parameters_path', help='path to the directory in which the "globalvars.py" parameter file for the run can be found')

if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

#parse command line arguments
args = parser.parse_args()

# outdirectory
imageDirectory = 'imageAnalysis/reduced_signals/barcode_mapping/figs'
# load concentrations
xValuesFilename, fluor_dirs_red, fluor_dirs, concentrations = IMlibs.loadMapFile(args.CPfluor_dirs_to_quantify)

# load table
fittedBindingFilename = args.CPfitted
table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
table['dG'] = parameters.RT*np.log(table['kd']*1E-9)

# get another dict that gives info on a per variant basis
variantFittedFilename = os.path.splitext(fittedBindingFilename)[0]+'.perVariant.fitParameters'
variant_table = pd.read_table(variantFittedFilename, index_col=0)
        

# make plots
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
histogram.compare([variant_table['numTests']], bar=True, xbins=np.arange(0, 30)-.5)
ax.set_xlim((-1, 31))
ax.set_xlabel('number of good measurements per variant')
ax.set_ylabel('fraction')
ax.legend_=None
plt.tight_layout()
plt.savefig(os.path.join(imageDirectory, 'variant_good_tests.histogram.pdf'))

# plot rigid
variant_descriptions = {'rigid':0,
                        'rigid+1':21744,
                        'rigid+2':21746,
                        'rigid-1':21739,
                        'rigid-2-1':21736,
                        'rigid-2-2':21738}
for name, number in variant_descriptions.items():
    variantFun.plotVariant(table[table['variant_number']==number], concentrations,name)
    plt.savefig(os.path.join(imageDirectory, 'variant_%d.%s.dG.histogram.pdf'%(number, name)))
    plt.close()
    plt.savefig(os.path.join(imageDirectory, 'variant_%d.%s.binding_curve.pdf'%(number, name)))
    plt.close()
    
# three by three
variant_descriptions = {'3x3':330,
                        '3x3+1':25699,
                        '3x3+2':25702,
                        '3x3-1': 25694,
                        '3x3-2': 25692}
for name, number in variant_descriptions.items():
    variantFun.plotVariant(table[table['variant_number']==number], concentrations,name)
    
# plot boxplot
helix_context = 'rigid'
for total_length in [8,9,10,11,12]:
    num_variants = variantFun.plotVariantBoxplots(table, variant_table, helix_context, total_length)
    plt.title('%s_%d'%(helix_context, total_length-10))
    
    plt.savefig(os.path.join(imageDirectory, 'all_topologies.%s.length_%d.boxplot.pdf'%(helix_context, total_length)))
    
helix_context = 'wc'
for total_length in [8,9,10,11,12]:
    num_variants = variantFun.plotVariantBoxplots(table, variant_table, helix_context, total_length)
    plt.title('%s_%d'%(helix_context, total_length-10))
    plt.savefig(os.path.join(imageDirectory, 'all_topologies.%s.length_%d.boxplot.pdf'%(helix_context, total_length)))
    
    # plot boxplot
helix_context = 'rigid'
for total_length in [8,9,10,11,12]:
    num_variants = variantFun.plotVariantBoxplots(table, variant_table, helix_context, total_length, loop='badLoop')
    plt.title('%s_%d'%(helix_context, total_length-10))
    plt.savefig(os.path.join(imageDirectory, 'all_topologies.%s.length_%d.badLoop.boxplot.pdf'%(helix_context, total_length)))
    
helix_context = 'wc'
for total_length in [8,9,10,11,12]:
    num_variants = variantFun.plotVariantBoxplots(table, variant_table, helix_context, total_length, loop='badLoop')
    plt.title('%s_%d'%(helix_context, total_length-10))
    plt.savefig(os.path.join(imageDirectory, 'all_topologies.%s.length_%d.badLoop.boxplot.pdf'%(helix_context, total_length)))
    
helix_context = 'rigid'
for total_length in [8,9,10,11,12]:
    num_variants = variantFun.plotVariantBoxplots(table, variant_table, helix_context, total_length, max_diff_helix_length=10)
    plt.title('%s_%d'%(helix_context, total_length-10))

helix_context = 'rigid'
for topology in ['','B1', 'B2', 'B1_B1', 'B2_B2', 'B1_B1_B1', 'B2_B2_B2', 'M','M_B1', 'B2_M', 'M_M',
                                    'B2_B2_M', 'M_B1_B1', 'B2_M_M', 'M_M_B1', 'M_M_M']:
    couldPlot = variantFun.plot_length_changes(table, variant_table, helix_context, topology)
    if couldPlot:
        plt.title('%s %s'%(helix_context, topology))
        plt.tight_layout()
        plt.savefig(os.path.join(imageDirectory, 'all_lengths.%s.topology_%s.lines.pdf'%(helix_context, topology)))

for topology in ['','B1', 'B2']:
    variantFun.plot_length_changes_helices(table, variant_table, topology)
    plt.title('%s %s'%(helix_context, topology))
    plt.tight_layout()
    plt.savefig(os.path.join(imageDirectory, 'all_lengths.all_helices.topology_%s.lines.pdf'%(topology)))
    
helix_context = 'rigid'
total_length = 10
for topology in ['','B1', 'B2', 'B1_B1', 'B2_B2', 'B1_B1_B1', 'B2_B2_B2', 'M','M_B1', 'B2_M', 'M_M',
                                    'B2_B2_M', 'M_B1_B1', 'B2_M_M', 'M_M_B1', 'M_M_M']:
    variantFun.plot_position_changes(table, variant_table, helix_context, topology, total_length)
    plt.title('%s %d %s'%(helix_context, total_length,topology))
    plt.tight_layout()
    plt.savefig(os.path.join(imageDirectory, 'all_helix1_lengths.%s.topology_%s.total_length_%d.lines.pdf'%(helix_context, topology, total_length)))