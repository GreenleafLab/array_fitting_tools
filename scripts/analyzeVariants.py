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
imageDirectory = 'binding_curves_rigid_tile456/reduced_signals/barcode_mapping/figs'
structuresDirectory = 'secondary_structures'

# load concentrations
fluor_dirs_red, fluor_dirs, concentrations = IMlibs.loadMapFile(args.CPfluor_dirs_to_quantify)

# load table
fittedBindingFilename = args.CPfitted
table = IMlibs.loadFittedCPsignal(fittedBindingFilename)
table['dG'] = parameters.RT*np.log(table['kd']*1E-9)

# get another dict that gives info on a per variant basis
indx_subset = np.array(np.argsort(table['dG'])[np.arange(0, len(table), 100)])
variants = np.unique(table['variant_number'].iloc[indx_subset][np.isfinite(table['variant_number'].iloc[indx_subset])])
perVariant = variantFun.perVariantInfo(table, variants=variants)

variantFittedFilename = os.path.splitext(fittedBindingFilename)[0]+'.perVariant.CPfitted'
variant_table = pd.read_table(variantFittedFilename)

# plot scatterplots of how qvalue relates to dG
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
#im = ax.scatter(variant_table['dG'].astype(float), -np.log10(variant_table['qvalue']), c=variant_table['fmax'], vmin=200, vmax=2000,alpha=0.1)
im = ax.scatter(table_reduced['dG'].astype(float), -np.log10(table_reduced['qvalue']), c=table_reduced['fmax'], vmin=200, vmax=2000,alpha=0.1)
ax.set_xlabel('delta G (kcal/mol)')
ax.set_ylabel('-log10(qvalue)')
ax.grid()
cbar = plt.colorbar(im)
cbar.set_label('fmax')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.savefig(os.path.join(imageDirectory, 'dG_vs_qvalue.pdf'))

# plot scatterplots of how qvalue relates to dG
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.scatter(variant_table['dG'].astype(float), -np.log10(variant_table['qvalue']), c=variant_table['fmax'], vmin=200, vmax=2000, alpha=0.1)
ax.set_xlabel('delta G (kcal/mol)')
ax.set_ylabel('-log10(qvalue)')
ax.grid()
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.savefig(os.path.join(imageDirectory, 'dG_qvalue.pdf'))
criteria = np.all((np.array(variant_table['qvalue'] < 0.06), np.array(variant_table['qvalue']>0.04)), axis=0)
histogram.compare([variant_table['dG'][criteria]])

# plot number of measurements per varaint
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
histogram.compare([variant_table['numTests']], bar=True, xbins=np.arange(0, 30)-.5)
ax.set_xlim((-1, 31))
ax.set_xlabel('number of good measurements per variant')
ax.set_ylabel('fraction')
ax.legend_=None
plt.tight_layout()
plt.savefig(os.path.join(imageDirectory, 'variant_good_tests.histogram.pdf'))

# save colorcoding/markers guide
variantFun.plotMarkers()
plt.savefig(os.path.join(imageDirectory, 'markers.guide.pdf'))
variantFun.plotColors()
plt.savefig(os.path.join(imageDirectory, 'colors.guide.pdf'))

# plot one set of variants
seq = '_'
criteria_dict = {'junction_sequence': seq, 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1'}
variants = variantFun.findVariantNumbers(table, criteria_dict)
per_variant = variantFun.perVariantInfo(table, variants=variants)
variantFun.plot_over_coordinate(variant_table.loc[variants])
plt.savefig(os.path.join(imageDirectory, 'junction_%s.central.num_variants.pdf'%seq))
plt.close()
plt.savefig(os.path.join(imageDirectory, 'junction_%s.central.length_landscape.pdf'%seq))


# plot another set
criteria_dict = {'junction_sequence': 'ATT_TT', 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1'}
variants = variantFun.findVariantNumbers(table, criteria_dict)
per_variant2 = variantFun.perVariantInfo(table, variants=variants)
variantFun.plot_over_coordinate(per_variant.append(per_variant2))
plt.savefig(os.path.join(imageDirectory, 'junction_%s.w_ATT_TT.central.num_variants.pdf'%seq))
plt.close()
plt.savefig(os.path.join(imageDirectory, 'junction_%s.w_ATT_TT.central.length_landscape.pdf'%seq))

# save sequences
variantFun.makeFasta(per_variant.append(per_variant2), image_dir=os.path.join(structuresDirectory, seq))

# plot another set
seq = 'TG_TTT'
criteria_dict = {'junction_sequence': seq, 'helix_context':'wc', 'loop':'goodLoop', 'receptor':'R1'}
variants = variantFun.findVariantNumbers(table, criteria_dict)
per_variant = variantFun.perVariantInfo(table, variants=variants)
variantFun.plot_over_coordinate(per_variant)
plt.savefig(os.path.join(imageDirectory,'junction_%s.central.num_variants.pdf'%('.'.join(criteria_dict.values()))))
plt.close()
plt.savefig(os.path.join(imageDirectory,'junction_%s.central.length_landscape.pdf'%('.'.join(criteria_dict.values()))))
variantFun.makeFasta(per_variant, image_dir=os.path.join(structuresDirectory, criteria_dict['helix_context'], seq))

# plot all 2x2 loops of length 10
topology = 'M_M'
length = 11
helix_one_length = 4
variants = variantFun.findVariantNumbers(table, {'topology':topology, 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1', 'total_length':length, 'helix_one_length':helix_one_length})
per_variant = variantFun.perVariantInfo(table, variants=variants)
variantFun.plot_over_coordinate(per_variant, x_param=np.arange(len(per_variant)), x_param_name='variant')
plt.savefig(os.path.join(imageDirectory,'junction_%s.length_%d.helix_one_length_%d.num_variants.pdf'%(topology, length, helix_one_length)))
plt.close()
plt.savefig(os.path.join(imageDirectory,'junction_%s.length_%d.helix_one_length_%d.dGs.pdf'%(topology, length, helix_one_length)))

# plot same set 2x2 loops of length 11
topology = 'M_M'
length = 10
helix_one_length = 4
variants = variantFun.findVariantNumbers(table, {'topology':topology, 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1', 'total_length':length, 'helix_one_length':helix_one_length})
per_variant = variantFun.perVariantInfo(table, variants=variants)
per_variant2 =  variantFun.perVariantInfo(table, variantFun.findVariantNumbers(table, {'topology':topology, 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1', 'total_length':length+1, 'helix_one_length':helix_one_length}))
indx = np.array([int(np.where(per_variant['junction_sequence']==seq)[0]) if len(np.where(per_variant['junction_sequence']==seq)[0]) > 0 else np.nan for seq in per_variant2.sort('dG').loc[:, 'junction_sequence']])
indx = indx[np.isfinite(indx)].astype(int)
variantFun.plot_over_coordinate(per_variant, x_param=np.arange(len(indx)), x_param_name='variant', sort_index = indx)
plt.savefig(os.path.join(imageDirectory,'junction_%s.length_%d.helix_one_length_%d.ordered_by_length_11bp.num_variants.pdf'%(topology, length, helix_one_length)))
plt.close()
plt.savefig(os.path.join(imageDirectory,'junction_%s.length_%d.helix_one_length_%d.ordered_by_length_11bp.dGs.pdf'%(topology, length, helix_one_length)))

# save sequences
indx = np.array(per_variant.sort('dG').index)[np.hstack((range(7), range(-12,0)))]
variantFun.makeFasta(per_variant.iloc[indx], image_dir=os.path.join(structuresDirectory, 'M_M_M_11bp'))
print '\t'.join(np.array(per_variant.iloc[indx[:7]].loc[:, 'variant_number'], dtype=int).astype(str))

# plot Kink Turns
topology = 'KT'
variants = variantFun.findVariantNumbers(table, {'topology':topology, 'helix_context':'rigid', 'loop':'goodLoop', 'receptor':'R1'})
per_variant = variantFun.perVariantInfo(table, variants=variants)
variantFun.plot_over_coordinate(per_variant, x_param=np.arange(len(indx)), x_param_name='variant', sort_index = indx)


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

for topology in ['','B1', 'B2','B1_B1','B2_B2','B1_B1_B1','B2_B2_B2']:
    variantFun.plot_length_changes_helices(table, variant_table, topology)
    #plt.title('%s %s'%(helix_context, topology))
    plt.title('%s'%(variantFun.ConvertNomen([topology])[0]))
    #plt.tight_layout()
    plt.savefig(os.path.join(imageDirectory, 'all_lengths.all_helices.topology_%s.lines.pdf'%(topology)))
    
helix_context = 'rigid'
total_length = 10
for topology in ['','B1', 'B2', 'B1_B1', 'B2_B2', 'B1_B1_B1', 'B2_B2_B2', 'M','M_B1', 'B2_M', 'M_M',
                                    'B2_B2_M', 'M_B1_B1', 'B2_M_M', 'M_M_B1', 'M_M_M']:
    variantFun.plot_position_changes(table, variant_table, helix_context, topology, total_length)
    plt.title('%s %d %s'%(helix_context, total_length,topology))
    plt.tight_layout()
    plt.savefig(os.path.join(imageDirectory, 'all_helix1_lengths.%s.topology_%s.total_length_%d.lines.pdf'%(helix_context, topology, total_length)))