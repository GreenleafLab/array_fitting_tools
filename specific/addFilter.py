
# laod barcode map and apply filter 
barcodeMap = pd.read_table('150608_barcode_mapping_lib2/tecto_lib2.barcode_to_seq')

# in this case, find those tectos that are 11bp long
barcodes = barcodeMap.loc[(barcodeMap.length == 11) | (barcodeMap.length == 10) | (barcodeMap.length == 12), 'barcode']
filterText = 'around_11bp'

# load old filtered ile CPseqs
inDir = '150607_chip/seqData/tiles/filtered_tiles_indexed'
outDir = '150607_chip/seqData/tiles/filtered_tiles_indexed_11bp'
if not os.path.exists(outDir): os.mkdir(outDir)
filteredTileFilenames = subprocess.check_output('ls %s/*CPseq'%inDir, shell=True).split()

# apply filters
for filteredTileFile in filteredTileFilenames:
    cols = ['tileID','filter','read1_seq','read1_quality','read2_seq','read2_quality','index1_seq','index1_quality','index2_seq', 'index2_quality']
    tileData = pd.read_table(filteredTileFile, header=None, names=cols )
    tileData.index = tileData.index1_seq
    barcodesSub = barcodes.loc[np.in1d(barcodes, tileData.index)]
    tileData.loc[barcodesSub, 'filter'] = ['%s:%s'%(s, filterText) for s in tileData.loc[barcodesSub, 'filter']]
    tileData.to_csv(os.path.join(outDir, os.path.basename(filteredTileFile)), sep='\t', header=None, index=None)