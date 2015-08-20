import itertools
import hjh.junction

libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.txt'
libChar = pd.read_table(libCharFile).loc[:, :'ss_correct']

# make a list of alphabetically sorted sequences of length 10
allLengths = [8,9,10,11,12]
matrixAll = {}
name = 'sequence'
for length in [9,10,11]:
    allPossibleBpSeqs = hjh.junction.Junction(tuple(['W']*(length-1))).sequences
    allPossibleBpSeqs.sort('side1', inplace=True)
    
    sequences = libChar.loc[(libChar.sublibrary=='sequences')&(libChar.length==length)].sequence
    subSequence = pd.Series([s[11:20].replace('T', 'U') for s in sequences], index=sequences.index)
    
    matrix = pd.DataFrame(index=np.arange(len(allPossibleBpSeqs)), columns=allLengths)
    matrix.loc[np.searchsorted(allPossibleBpSeqs.side1, subSequence), length] = (
        subSequence.index.tolist())
    print 'length:\t\t%d'%length
    print 'possible bp sequences:\t%d'%len(allPossibleBpSeqs)
    print 'number of sequences designed:\t%d'%len(subSequence)
    print 'percent of possible: %4.4f%%'%(100*len(subSequence)/float(len(allPossibleBpSeqs)))
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

#flanks = ['GCGC', 'CUAG']
#associatedMotifs = {}
#for motif, flank in itertools.product(['B1', 'B2', 'B1,B1', 'B2,B2', 'B1,B1,B1',
#                                       'B2,B2,B2'], flanks):
#    final_motif = tuple(list(flank[:2]) + motif.split(',') + list(flank[2:]))
#    associatedMotifs.update({','.join(final_motif):final_motif})



for position in [-1, 0, 1]:
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

            s = libChar.loc[(libChar.sublibrary=='junction_conformations')&
                            (libChar.length==length)&
                            (libChar.offset==position)&
                            pd.Series(np.in1d(libChar.no_flank, b),
                            #pd.Series(np.in1d(libChar.junction_seq, b),
                                      index=libChar.index)].junction_seq
            index = np.searchsorted(a, s)
            if np.all(a.iloc[index] == s):
                matrix.loc[np.searchsorted(a, s), length] = s.index.tolist()
            else:
                print 'possible error with %s'%motif
            
        matrixAll['%s_%d'%(motif, position)] = matrix


labelMat = {}
for position in [-1, 0, 1]: 

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
for length in [9, 10, 11]:
    labelMat.update({'sequence_%d_0'%length:'sequence_%d'%length})


numLengths = len(allLengths)

labels = ['sequence_9', 'sequence_10', 'sequence_11',
            '0x1', '0x2', '0x3',
            '1x0', '2x0', '3x0',
            '1x1', "1x1'",
            '1x2', "1x2'",
            '1x3', "1x3'",
            '2x1', "2x1'",
            '3x1', "3x1'",
            '2x2',  '3x3', "3x3'"]
width_keys = ['%s_0'%key for key in labels]
sns.set_style("whitegrid", {'grid.linestyle': u':', 'axes.edgecolor': '0.9'})
fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(2, len(labels), wspace=0, hspace=0.05,
                       width_ratios=[np.log10(len(matrixAll[labelMat[key]].dropna(axis=0, how='all')))
                       #width_ratios=[np.log10(len(matrixAll[labelMat[key]]))
                                     for key in width_keys],
                       height_ratios=[1,5],
                       bottom=0.25, right=0.97, left=0.1, top=0.97)
markers = ['^', 'o', 'v']
cmap = sns.cubehelix_palette(start=0.75, rot=1.25, light=0.40, dark=0.05, reverse=True, hue=1, as_cmap=True)
colors= ['black', 'red brown', 'orange brown', 'greenish blue', 'dark teal']
colors= ['black', 'red brown', 'tomato red', 'blue', 'navy blue']
for i, key in enumerate(labels):
    ax = fig.add_subplot(gs[1, i])
    number = ['']*3
    for j, position in enumerate([-1, 0, 1]):
        try:
            a = matrixAll[labelMat['%s_%d'%(key, position)]].dropna(axis=0, how='all')
            indices = np.hstack(np.column_stack(a.values)).astype(float)
            index = np.isfinite(indices)
            
            x = np.array(a.index.tolist())/float(np.max(a.index.tolist()))
            x = np.hstack([x]*numLengths)[index]
            
            y = libChar.loc[indices[index]].length.values
            jitter = st.norm.rvs(loc=position*0.2, scale=0.05, size=len(y))
            c = (libChar.loc[indices[index]].helix_one_length).fillna(0).values
            if not key.find('sequence') == 0:
                c += 1
            for k in range(5):
                index = c == k
                ax.scatter(x[index], (y+jitter)[index], s=1, marker='.',
                           facecolors=sns.xkcd_rgb[colors[k]], edgecolors='none')
            number[j] = (len(a))
            numPossible = len(matrixAll[labelMat['%s_%d'%(key, position)]])
        except:
            pass
    plt.ylim(7.5, 12.5)
    plt.xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    ax.set_yticks(allLengths)
    if i != 0:
        ax.set_yticklabels([])
    print numPossible
    
    if key.find('sequence')==0:
        ax.annotate('%s'%(key),
                     xy=(0.5, -0.005),
                     xycoords='axes fraction',
                     horizontalalignment='center', verticalalignment='top',
                     rotation=90,
                     fontsize=10)
    else:
        ax.annotate('%s'%(key),
                     xy=(0.5, -0.13),
                     xycoords='axes fraction',
                     horizontalalignment='center', verticalalignment='bottom',
                     rotation=90,
                     fontsize=10)   
    # plot second box
    n = number[1]
    aspect = 0.1
    ax = fig.add_subplot(gs[0, i])
    ax.bar(left=[0], height=np.log10(numPossible), width=1,
           facecolor='grey', edgecolor='0.9', alpha=0.5)
    ax.bar(left=[0], height=np.log10(n), width=0.8,
           facecolor=sns.xkcd_rgb['charcoal'], edgecolor='0.9', alpha=1)

    
    #ax.bar(left=[0], height=numPossible/np.log10(numPossible), width=1, color='grey', alpha=0.5)
    #ax.bar(left=[0], height=n/(aspect*np.log10(numPossible)), width=aspect, color='navy', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0,1])
    ax.set_ylim([0, 6])
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(['$10^%d$'%y if (y-1)%2==0 else '' for y in np.arange(6)] )
    #ax.set_yscale('log')
    ax.set_xticklabels([])
    if i != 0:
        ax.set_yticklabels([])
    
    
    #ax.annotate('%d/\n%d'%(number[1],numPossible),
    #             xy=(0.05, 1.005),
    #             xycoords='axes fraction',
    #             horizontalalignment='right', verticalalignment='top',
    #             rotation=90,
    #             fontsize=10)
    
    
    #print '%s\t%4.4f%%'%(key, 100*len(a)/float(len(matrixAll[labelMat[key]])))
                                 
    

lengths = [9, 10, 11]
numLengths = len(lengths)
fig = plt.figure()
gs = gridspec.GridSpec(1, numLengths, wspace=0, 
                       width_ratios=[len(matrix.dropna(axis=0, how='all'))
                                     for matrix in matrixAll.items()])
for i, motif in enumerate(lengths):

    ax = fig.add_subplot(gs[0, i])
    a = matrixAll.dropna(subset=[length])
    x = np.array(a.index.tolist())/float(np.max(a.index.tolist()))
    y = matrixAll.dropna(subset=[length]).loc[:, length].values
        
        
    jitter = st.norm.rvs(loc=0, scale=0.1, size=len(y))
    
    ax.scatter(x, y+jitter, s=2, marker='.', c='k')
    plt.ylim(7.5, 12.5)
    plt.xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    ax.set_yticks(allLengths)
    ax.set_yticklabels([])
                                                                                                                                                                                                                                                   
