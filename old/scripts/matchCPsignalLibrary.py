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
opts.add_option("-b", help="accepts barcode mapping")
opts.add_option("-i", help="accepts cpsignal")
opts.add_option("-o", help="output file")
options, arguments = opts.parse_args()

# return usage information if no argvs given
if len(sys.argv)==1:
    os.system(sys.argv[0]+" --help")
    sys.exit()

##### FUNCTIONS #####


def findSeqMapDict(cpsignal,barcodeMap, is_designed):
    names = [col for col in cpsignal] + [col for col in barcodeMap]
    seqMap = pd.DataFrame(data=np.nan, columns=names, index=np.arange(len(cpsignal))) 
    for name in  [col for col in cpsignal]:
        seqMap.loc[:,name] = np.array(cpsignal[name])
    indices_cpsignal = np.arange(len(cpsignal))[is_designed > -1]
    indices_barcode = is_designed[is_designed > -1]
    for name in [col for col in barcodeMap]:
        seqMap.loc[:,name].iloc[indices_cpsignal] = np.array(barcodeMap[name].iloc[indices_barcode])
    return seqMap
    

##### SCRIPT #####
if __name__ == '__main__':
    # load files
    barcodeMapFilename = options.b
    cpSignalFilename = options.i # i.e. '../../all3x3.all.library'
    outFile = options.o

    print "loading CP signal file ..."
    cpsignal = IMlibs.loadCPseqSignal(cpSignalFilename)
    # if there is no barcode, add the first 16 bases, just in case that matches one of the mapped ones
    index = cpsignal.index1_seq.isnull()
    cpsignal.loc[index, 'index1_seq'] = [s[:16] for s in cpsignal.loc[index, 'read1_seq']]
    cpsignal.index = cpsignal.index1_seq
    
    print "loading barcode Map..."
    barcodeMap = pd.read_table(barcodeMapFilename)
    barcodeMap.index = barcodeMap.barcode
    
    print "making final barcode to sequence mapping file"
    seqMap = pd.concat([cpsignal, barcodeMap.loc[cpsignal.index1_seq]], axis=1)
    seqMap.sort('variant_number', inplace=True)
    IMlibs.saveDataFrame(seqMap, outFile, index=False, float_format='%4.0f')
    
    sys.exit()
    num_bc_per_variant, is_designed = IMlibs.findSequenceRepresentation(np.array(cpsignal['index1_seq']),
                                                                        np.array(barcodeMap['barcode']),
                                                                        exact_match = True)
    # get dict of barcode map
   
    seqMap = findSeqMapDict(cpsignal,barcodeMap, is_designed)
    
    IMlibs.saveDataFrame(seqMap, outFile, index=False, float_format='%4.0f')

