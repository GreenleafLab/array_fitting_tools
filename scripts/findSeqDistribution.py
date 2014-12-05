"""
Sarah Denny
Stanford University

Using the compressed barcode file (from Lauren) and the list of designed variants,
figure out how many barcodes per designed library variant (d.l.v.), and plot histograms.
"""

##### IMPORT #####
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append('/home/sarah/atac_tools/')
import histogram
import seqfun
import IMlibs


##### OPTIONS #####
# read options from command line
opts = OptionParser()
usage = "usage: %prog [options] [inputs]"
opts = OptionParser(usage=usage)
opts.add_option("-b", help="<.unq> accepts barcode mapping")
opts.add_option("-l", help="<library> accepts library as it is formatted from library generator scripts")
opts.add_option("-o", help="output file")
options, arguments = opts.parse_args()

# return usage information if no argvs given
if len(sys.argv)==1:
    os.system(sys.argv[0]+" --help")
    sys.exit()

##### FUNCTIONS #####
def poisson(l, k=None):
    if k is None:
        k = np.arange(0, 15)
    return np.power(l, k)*np.exp(-l)/np.array([np.math.factorial(ki) for ki in k])

def doublepoisson(l1, l2, k=None):
    if k is None:
        k = np.arange(0, 15)
    return np.power(l1, k)*np.exp(-l1)/np.array([np.math.factorial(ki) for ki in k])*np.power(l2, k)*np.exp(-l2)/np.array([np.math.factorial(ki) for ki in k])



def plotBarcodesPerVariant(num_represented, outdir):
    # number of barcodes/unique sequence variant 
    plt.figure(figsize=(6, 4))
    ylimit = np.sum(num_represented>0)/np.sum(num_represented)
    xlimit = np.percentile(num_represented, 99.9)
    histogram.compare([num_represented], xbins=np.arange(0, xlimit)-0.5,
        labels=['%4.1f%% library variants\nrepresented on chip'%(np.sum(num_represented>0)/float(len(num_represented))*100)],
        bar=True,
        normalize=False)
    #plt.ylim((0, 0.9))
    plt.xlim((-1, 30))
    plt.xlabel('number of barcodes per designed variant')
    plt.ylabel('fraction')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'barcodes_per_designed_library_variant.histogram.pdf'))
    return

def plotClustersPerBarcode(consensus, is_designed, outdir):
    # clusters per barcode, using subset of barcodes whose consensus sequence is a designed library variant (d.l.v.)
    num_clusters_per_bc = consensus['clusters_per_barcode'].iloc[is_designed > -1]
    plt.figure(figsize=(6, 4))
    xlimit = np.percentile(num_clusters_per_bc, 99.9)
    histogram.compare([num_clusters_per_bc], xbins=np.arange(0, xlimit, 1)-0.5,
        labels=['%d unique barcodes w/d.l.v.'%(len(num_clusters_per_bc))],
        normalize=False,
        bar=True)
    #plt.ylim((0, 0.7))
    plt.xlim((0, xlimit))
    plt.xlabel('number of clusters per barcode')
    plt.ylabel('number')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'clusters_per_barcode_of_designed_library_variants.histogram.pdf'))
    return

def plotClustersPerVariant(num_represented, consensus, is_designed, outdir):
    # number of clusters/unique sequence variant that is actually represented on chip
    num_clusters = np.zeros(len(num_represented))
    for i, idx in enumerate(is_designed):
        # does this barcode contain a designed library variant?
        if idx > -1:
            # if so, add the number of consensus sequences associated with this barcode
            # to that index in 'num_clusters'
            num_clusters[idx] += consensus['clusters_per_barcode'].iloc[i]
    plt.figure(figsize=(6, 4))
    ylimit = np.sum(num_clusters>0)/np.sum(num_represented)
    xlimit = np.max(num_clusters)
    histogram.compare([num_clusters[num_represented > 0]], xbins=np.arange(0, xlimit, 1)-0.5,
        labels=['median #measurements/variant=%4.1f'%(np.median(num_clusters[num_represented > 0]))],
        normalize=False)
    plt.xlim((0, xlimit))
    plt.xlabel('number of clusters per d.l.v.')
    plt.ylabel('number')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'clusters_per_designed_library_variant.onchip.histogram.pdf'))
    return

def plotPiePlots():
    # do the number of sequences designed versus not
    df = pd.Series(np.array([np.sum(is_designed > -1), np.sum(is_designed == -1)]), index=['a', 'b'], name='all barcodes')
    plt.figure()
    ax = df.plot(kind='pie', subplots=True, figsize=(4,4), labels=['our library', 'not designed'] , autopct='%.2f%%', fontsize=15, colors=['r', '0.5'],  )
    ax.set_aspect('equal')
    ax.legend_ = None
    plt.savefig('number_of_barcodes.designedvsnot.pie.pdf')
    
    # do the number of designed variants on chip
    plt.figure()
    df = pd.Series(np.array([np.sum(num_clusters > 0), np.sum(num_clusters == 0)]), index=['a', 'b'], name='all library variants')
    ax = df.plot(kind='pie', subplots=True, figsize=(4,4), labels=['on chip', 'not on chip'] , autopct='%.2f%%', fontsize=15, colors=['r', '0.5'],  )
    ax.set_aspect('equal')
    ax.legend_ = None
    plt.savefig('number_of_variants.onchip_vs_not.pie.pdf')
    
    plt.figure()
    df = pd.Series(np.array([np.sum(num_clusters >= 5), np.sum(num_clusters < 5)]), index=['a', 'b'], name='all library variants')
    ax = df.plot(kind='pie', subplots=True, figsize=(4,4), labels=['on chip >=5 times', '< 5 times'] , autopct='%.2f%%', fontsize=15, colors=['r', '0.5'],  )
    ax.set_aspect('equal')
    ax.legend_ = None
    plt.savefig('number_of_variants.onchip>=5times_vs_not.pie.pdf')
    return

def findBarcodeDict(consensus, is_designed, designed_library_unique):
    # make a new dataframe containing all the extra information
    names = [col for col in designed_library_unique] + ['fraction_consensus', 'clusters_per_barcode']
    barcodeMap = pd.DataFrame(data=np.nan, columns=names, index=consensus['barcode'])
    read2_seq = 'CCTAGTGATCCAGC'
    # put in sequence of those that didn't match exactly with library:
    # reverse complement the sequence up to the stall sequence
    barcodeMap['sequence'] = np.array([seqfun.reverseComplement(sequence[:sequence.find(read2_seq)]) for sequence in np.array(consensus['sequence'])])
    barcodeMap['fraction_consensus'] = 100*np.array(consensus['fraction_consensus'])
    barcodeMap['clusters_per_barcode'] = np.array(consensus['clusters_per_barcode'])
    indices_barcode = np.arange(len(barcodeMap))[is_designed > -1]
    indices_sequence = is_designed[is_designed > -1]
    for name in designed_library_unique:
        barcodeMap[name].iloc[indices_barcode] = np.array(designed_library_unique[name].iloc[indices_sequence])
    return barcodeMap
    

##### SCRIPT #####
if __name__ == '__main__':
    # load files
    consensusFile = options.b
    libraryFile = options.l # i.e. '../../all3x3.all.library'
    outFile = options.o
    outDir = os.path.dirname(outFile)
    
    print "loading consensus sequences..."
    consensus = IMlibs.loadCompressedBarcodeFile(consensusFile)
    consensus.sort('sequence', inplace=True)
    consensus_sequences = np.array(consensus['sequence'])
    
    print "loading designed sequences..."
    designed_library = IMlibs.loadLibraryCharacterization(libraryFile)
    
    # make library sequences unique
    designed_sequences, unique_indices = np.unique(designed_library['sequence'], return_index=True)
    designed_library_unique = designed_library.iloc[unique_indices]
    print "reduced library size from %d to %d after unique filter"%(len(designed_library), len(designed_library_unique))
    
    # add field in designed_library_unique which gives an int to that sequence
    designed_library_unique.insert(0, 'variant_number', unique_indices.astype(int))
    
    # figure out read length
    read_length = len(consensus_sequences[0])
    
    # reformat designed sequences to be reverse complement (as in read 2)
    compare_to = np.array([seqfun.reverseComplement(sequence)[:read_length] for sequence in designed_sequences])
    
    # find number of times each sequence that has at least one representation is in the
    # original block of seqeunces
    num_bc_per_variant, is_designed = IMlibs.findSequenceRepresentation(consensus_sequences, compare_to)
    print "Done with mapping. Making plots.."
    
    # make plots
    plotBarcodesPerVariant(num_bc_per_variant, outDir)
    plotClustersPerBarcode(consensus, is_designed, outDir)
    plotClustersPerVariant(num_bc_per_variant, consensus, is_designed, outDir)
    
    # get dict of barcode map
    print "making final barcode to sequence mapping file"
    barcodeMap = findBarcodeDict(consensus, is_designed, designed_library_unique)
    barcodeMap.sort('variant_number', inplace=True)
    IMlibs.saveDataFrame(barcodeMap, outFile, index=True, float_format='%4.0f')
 
