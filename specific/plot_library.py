import itertools
import hjh.junction

libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.txt'
libChar = pd.read_table(libCharFile).loc[:, :'ss_correct']

# make a list of alphabetically sorted sequences of length 10
allLengths = [8,9,10,11,12]
matrixAll = {}
name = 'sequence'
for length in allLengths:
    allPossibleBpSeqs = hjh.junction.Junction(tuple(['W']*(length-1))).sequences
    allPossibleBpSeqs.sort('side1', inplace=True)
    
    sequences = libChar.loc[(libChar.length==length)].sequence
    
    subSequence = pd.DataFrame(np.column_stack([[s[11:11+length-1] for s in sequences],
                                                 [s[-12-length+1:-12] for s in sequences]]),
        index=sequences.index, columns=['side1', 'side2'])
    index = pd.Series([seqfun.reverseComplement(subSequence.loc[idx].side1) ==
                       subSequence.loc[idx].side2 for idx in subSequence.index],
        index=sequences.index)
    bpSeqs = subSequence.loc[index].side1.str.replace('T', 'U')
    
    matrix = pd.DataFrame(index=np.arange(len(allPossibleBpSeqs)), columns=allLengths)
    
    index = np.searchsorted(allPossibleBpSeqs.side1, bpSeqs)
    matching_locs = (allPossibleBpSeqs.side1.iloc[index] == bpSeqs).values
    matrix.loc[index[matching_locs], length] = (bpSeqs.iloc[matching_locs].index.tolist())
    #print 'length:\t\t%d'%length
    #print 'possible bp sequences:\t%d'%len(allPossibleBpSeqs)
    #print 'number of sequences designed:\t%d'%len(subSequence)
    #print 'percent of possible: %4.4f%%'%(100*len(subSequence)/float(len(allPossibleBpSeqs)))
    matrixAll['%s_%d'%(name, length)] = matrix
    
# now do all motifs
#matrixAll = {}
associatedMotifs = {'B1':('W', 'W', 'B1', 'W', 'W'),
                    'B2':('W', 'W', 'B2', 'W', 'W',),
                    'B1,B1':('W', 'W', 'B1', 'B1', 'W', 'W',),
                    'B2,B2':('W', 'W', 'B2', 'B2', 'W', 'W',),
                    'B1,B1,B1':('W', 'W', 'B1', 'B1', 'B1','W', 'W',),
                    'B2,B2,B2':('W', 'W', 'B2', 'B2', 'B2','W', 'W',),
                    'W,B1,M':('W', 'W', 'B1', 'M', 'W'),
                    'M,B1,W':('W', 'M', 'B1', 'W', 'W'),
                    'W,B2,M':('W', 'W', 'B2', 'M', 'W'),
                    'M,B2,W':('W', 'M', 'B2', 'W', 'W'),
                    'W,B1,B1,M':('W', 'W', 'B1', 'B1', 'M', 'W'),
                    'M,B1,B1,W':('W', 'M', 'B1', 'B1', 'W', 'W'),
                    'W,B2,B2,M':('W', 'W', 'B2', 'B2', 'M', 'W'),
                    'M,B2,B2,W':('W', 'M', 'B2', 'B2', 'W', 'W'),
                    'M,W':('W', 'M', 'W', 'W'),
                    'W,M':('W', 'W', 'M', 'W'),
                    'M,M':('W', 'M', 'M', 'W'),
                    'W,M,M,M':('W', 'M', 'M', 'M'),
                    'M,M,M,W':('M', 'M', 'M', 'W')}

#associatedMotifsAlong = {'B1':('W', 'B1', 'W'),
#    'B2':('W', 'B1', 'W')}

#flanks = ['GCGC', 'CUAG']
#associatedMotifs = {}
#for motif, flank in itertools.product(['B1', 'B2', 'B1,B1', 'B2,B2', 'B1,B1,B1',
#                                       'B2,B2,B2'], flanks):
#    final_motif = tuple(list(flank[:2]) + motif.split(',') + list(flank[2:]))
#    associatedMotifs.update({','.join(final_motif):final_motif})


positions = np.unique(libChar.loc[libChar.sublibrary=='along'].offset)
for position in positions:
    for motif in associatedMotifs.keys():
        
        allPossibleSeqs = hjh.junction.Junction(associatedMotifs[motif]).sequences
        junctionSeqs = hjh.junction.Junction(tuple(motif.split(','))).sequences
        
        # with U,A flanker
        a = 'U' + allPossibleSeqs.side1 + 'U_A' + allPossibleSeqs.side2 + 'A'
        a.sort()
        
        # parsed no_flank seq
        b = junctionSeqs.side1 + '_' + junctionSeqs.side2 
        #b = 'U' + junctionSeqs.side1 + 'U_A' + junctionSeqs.side2 + 'A'
        
        # compared to sequences
        matrix = pd.DataFrame(index=np.arange(len(a)), columns=allLengths)
        for length in allLengths:

            s = libChar.loc[(libChar.length==length)&
                            (libChar.offset==position)&
                            pd.Series(np.in1d(libChar.no_flank, b),
                            #pd.Series(np.in1d(libChar.junction_seq, b),
                                      index=libChar.index)].junction_seq
            index = np.searchsorted(a, s)
            matching_locs = (a.iloc[index] == s).values
            
            matrix.loc[index[matching_locs], length] = s.iloc[matching_locs].index.tolist()

        matrixAll['%s_%d'%(motif, position)] = matrix


labelMat = {}
for position in np.unique(libChar.loc[libChar.sublibrary=='along'].offset):

    labelMat.update({'1x0_%d'%position:'B1_%d'%position,
                '0x1_%d'%position:'B2_%d'%position,
                '2x0_%d'%position:'B1,B1_%d'%position,
                '0x2_%d'%position:'B2,B2_%d'%position,
                '0x3_%d'%position:'B2,B2,B2_%d'%position,
                '3x0_%d'%position:'B1,B1,B1_%d'%position,
                '1x1_%d'%position:  'W,M_%d'%position,
                '1x1\'_%d'%position:'M,W_%d'%position,
                '2x1_%d'%position:  'W,B1,M_%d'%position,
                '2x1\'_%d'%position:'M,B1,W_%d'%position,
                '1x2_%d'%position:'W,B2,M_%d'%position,
                '1x2\'_%d'%position:'M,B2,W_%d'%position,
                '2x2_%d'%position:  'M,M_%d'%position,
                '1x3_%d'%position:  'W,B1,B1,M_%d'%position,
                '1x3\'_%d'%position:'M,B1,B1,W_%d'%position,
                '3x1_%d'%position:  'W,B2,B2,M_%d'%position,
                '3x1\'_%d'%position:'M,B2,B2,W_%d'%position,
                '3x3_%d'%position  :'W,M,M,M_%d'%position,
                '3x3\'_%d'%position:'M,M,M,W_%d'%position,
                })
for length in allLengths:
    labelMat.update({'sequence_%d_0'%length:'sequence_%d'%length})

plotFun.plotLibraryFigure(matrixAll, labelMat, libChar

# along
associatedMotifs = {'B1':('W', 'B1', 'W'),
                    'B2':('W', 'B2', 'W',),
                    'B1,B1':('W', 'W', 'B1', 'B1', 'W', 'W',),
                    'B2,B2':('W', 'W', 'B2', 'B2', 'W', 'W',),
                    'B1,B1,B1':('W', 'W', 'B1', 'B1', 'B1','W', 'W',),
                    'B2,B2,B2':('W', 'W', 'B2', 'B2', 'B2','W', 'W',),
                    'W,B1,M':('W', 'W', 'B1', 'M', 'W'),
                    'M,B1,W':('W', 'M', 'B1', 'W', 'W'),
                    'W,B2,M':('W', 'W', 'B2', 'M', 'W'),
                    'M,B2,W':('W', 'M', 'B2', 'W', 'W'),
                    'W,B1,B1,M':('W', 'W', 'B1', 'B1', 'M', 'W'),
                    'M,B1,B1,W':('W', 'M', 'B1', 'B1', 'W', 'W'),
                    'W,B2,B2,M':('W', 'W', 'B2', 'B2', 'M', 'W'),
                    'M,B2,B2,W':('W', 'M', 'B2', 'B2', 'W', 'W'),
                    'M,W':('W', 'M', 'W', 'W'),
                    'W,M':('W', 'W', 'M', 'W'),
                    'M,M':('W', 'M', 'M', 'W'),
                    'W,M,M,M':('W', 'M', 'M', 'M'),
                    'M,M,M,W':('M', 'M', 'M', 'W')}