import pandas as pd
import tectoData
import seaborn as sns
sns.set_style('white')

figDirectory = 'flow_rigid/figs_07_15_15'
#filename = 'binding_curves_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.abbrev.CPfitted'
filename = 'AAYFY_ALL_filtered_tecto_sorted.annotated.abbrev.CPfitted'
table = pd.read_table(filename, index_col=0)
#filename = 'binding_curves_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'
filename = 'AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.bootStrapped.CPfitted'
variant_table = pd.read_table(filename, index_col=0)

filename = 'off_rates_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.abbrev.CPfitted'
table_kinetics = pd.read_table(filename, index_col=0)
filename = 'off_rates_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted'
variant_table_kinetics = pd.read_table(filename, index_col=0)
times = pd.read_table('off_rates_rigid/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.times', index_col=0)

table.dropna(axis=0, subset=['variant_number'], inplace=True)

bins = np.linspace(0, 2, 50)
fig = plt.figure(figsize=(4.5,3.75))
index = nan_filter&table.loc[:, 'barcode_good']&(table.loc[:, 'qvalue']<=0.05)
index2 = nan_filter&table.loc[:, 'barcode_good']&(table.loc[:, 'qvalue']>0.05)
plt.hist(table.loc[index, 'dG_var'].dropna().astype(float), normed=False, bins=bins, color=sns.xkcd_rgb['vermillion'], alpha=0.5, label='q<=0.05')
plt.hist(table.loc[index2, 'dG_var'].dropna().astype(float), normed=False, bins=bins, color=sns.xkcd_rgb['light grey'], alpha=0.5, label='q>0.05')
plt.xlabel('variance in fit dG, per cluster (kcal/mol)')
plt.ylabel('number of clusters')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'histogram.variance_dG.nan_barcode_filters.pdf'))

# plot histogram of dGs
bins = np.arange(-14, -2, 0.2 )
fig = plt.figure(figsize=(4.5,3.75))
variant = 0
subtable = table.loc[(table.loc[:, 'variant_number'] == variant)&(table.loc[:, 'barcode_good'])&nan_filter]
plt.hist(subtable.loc[:, 'dG'], bins=bins, normed=False, alpha=0.5, color=sns.xkcd_rgb['vermillion'])

variant = 21745
subtable = table.loc[(table.loc[:, 'variant_number'] == variant)&(table.loc[:, 'barcode_good'])&nan_filter]
plt.hist(subtable.loc[:, 'dG'], bins=bins, normed=False, alpha=0.5, color=sns.xkcd_rgb['mahogany'])


# variant table
variant_table.loc[:, 'conf_int_median'] = variant_table.loc[:, 'dG_ub'] - variant_table.loc[:, 'dG_lb']
variant_table.loc[~np.isfinite(variant_table.loc[:, 'conf_int_median']), 'conf_int_median'] = np.nan


fig = plt.figure(figsize=(4.5,3.75))
bins = np.linspace(0, 4, 50)
index = (variant_table.loc[:, 'numTests'] >= 5)&(variant_table.loc[:, 'qvalue'] < 0.05)
plt.hist(variant_table.loc[index, 'conf_int_median'].dropna().astype(float),
         normed=False, bins=bins, color=sns.xkcd_rgb['vermillion'], alpha=0.5, label='q>=0.5')
index = (variant_table.loc[:, 'numTests'] >= 5)&(variant_table.loc[:, 'qvalue'] > 0.05)
plt.hist(variant_table.loc[index, 'conf_int_median'].dropna().astype(float).values,
         normed=False, bins=bins, color=sns.xkcd_rgb['light grey'], alpha=0.5, label='q>0.05')
plt.xlabel('width of 95% confidence interval \nin the median fit dG (kcal/mol)')
plt.ylabel('number of variants')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'histogram.conf_interval.at_least_5variants.pdf'))

fig = plt.figure(figsize=(4.5,3.75))
bins = np.linspace(0, 4, 50)
index = (variant_table.loc[:, 'qvalue'] <= 0.05)
plt.hist(variant_table.loc[index, 'conf_int_median'].dropna().astype(float),
         normed=False, bins=bins, color=sns.xkcd_rgb['vermillion'], alpha=0.5, label='q>=0.5')

index = (variant_table.loc[:, 'qvalue'] > 0.05)
plt.hist(variant_table.loc[index, 'conf_int_median'].dropna().astype(float).values,
         normed=False, bins=bins, color=sns.xkcd_rgb['light grey'], alpha=0.5, label='q>0.05')
plt.xlabel('width of 95% confidence interval \nin the median fit dG (kcal/mol)')
plt.ylabel('number of variants')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'histogram.conf_interval.pdf'))

# plot binding curves
variant = 21747
a = tectoData.loadTectoData(table, variant_table, variant)
a.plotMedianNormalizedClusters()
plt.savefig(os.path.join(figDirectory, 'binding_curves.variant_%d.pdf'%variant)); plt.close()
a.plotHistogram()
plt.savefig(os.path.join(figDirectory, 'histogram_dG.variant_%d.pdf'%variant)); plt.close()
cmd = a.getSecondaryStructure(name=os.path.join(figDirectory, 'variant_%d.ss.eps'%variant))
os.system(cmd)

filename = 'library_and_chip_info/variant_over_length.offset_0.both.dat'
variantDict = pd.read_table(filename, index_col=[0,1, 2])
#variantDict.index.names = ['context', 'junction', 'sequence']

filename = 'flow_rigid/delta_g.rigid_wc.dat'
yvaluesVariants = pd.read_table(filename, index_col=[0,1,2], header=[0,1])
#yvaluesVariants.index.names = ['context', 'junction', 'sequence']

context = 'wc'
for context in ['rigid', 'wc']:
    a = tectoData.getAllVariantNumbers(table, variant_table, ['B2_B2_M','B2_M', 'B2_M_M', 'M_B1', 'M_B1_B1','M_M_B1'], helix_context=context)
    variantDict.append(pd.concat({context:a}, names=['context']))

variantDict.append(pd.concat({context:a}, names=['context'])).to_csv(filename, sep='\t')

for motif in ['B2_B2_M','B2_M', 'B2_M_M', 'M_B1', 'M_B1_B1','M_M_B1']:
    b = tectoData.parseVariantDeltaGsByLength(table, variant_table, motif,variantDict.loc[context] )
    yvaluesVariants = yvaluesVariants.append(pd.concat({context: pd.concat({motif:b}, names=['motif'])}, names=['context']))
    
# plot all motifs of a particular type
for motif in ['B2_M', 'M_B1', 'M_B1_B1', 'B2_B2_M', 'M_M_B1', 'B2_B2_M']:
    context='wc'
    ax = tectoData.plotOverLength(yvaluesVariants.loc[context].loc[motif], how_to_index=4)
    yvaluesAll = yvaluesVariants.loc[context].loc[motif]
    index = yvaluesAll.loc[:, 'bp_correct'].all(axis=1)
    lengths = [8,9,10,11,12]
    ax.errorbar(lengths, yvaluesAll.loc[index, 'dG'].mean(axis=0), yerr=yvaluesAll.loc[index, 'dG'].std(axis=0),
                fmt='o-', elinewidth=2, capsize=4, capthick=2, alpha=0.5, marker='.', markersize=4,
                color=sns.xkcd_rgb['cerulean'], linewidth=1)
    plt.savefig(os.path.join(figDirectory, 'junction.%s.by_length.%s_context.pdf'%(motif, context)))

colors = sns.color_palette('Paired'); count = 0
jitter = np.linspace(-0.05, 0.05, 2)
fig = plt.figure(figsize=(7.5, 2.75))
gs = gridspec.GridSpec(1, 3, wspace=0)
nameDict = {'wc':'-2CG;UA', 'rigid':'-2GU;UG'}
# plot all moitf average for  of a particular type
for i, motif in enumerate(['B1', 'B1_B1', 'B1_B1_B1']):
    for j, context in enumerate(['wc', 'rigid']):
        ax = fig.add_subplot(gs[0,i])
        yvaluesAll = yvaluesVariants.loc[context].loc[motif]
        index = yvaluesAll.loc[:, 'bp_correct'].all(axis=1)
        ax.errorbar(lengths+jitter[j], yvaluesAll.loc[index, 'dG'].mean(axis=0), yerr=yvaluesAll.loc[index, 'dG'].std(axis=0),
                fmt='o-', elinewidth=2, capsize=4, capthick=2, alpha=0.5, marker='o', markersize=4,
                color=colors[count], linewidth=1, label='%s %s'%(nameDict[context], motif))
        count += 1
        ax.set_xlim(7.5, 12.5)
        ax.set_ylim(-12, -6)
        ax.set_xticks([8, 9, 10, 11, 12])
        if i != 0:
            ax.set_yticks([])

        plt.legend(loc='upper center')
plt.savefig(os.path.join(figDirectory, 'junction.%s.wc_rigid.by_length.pdf'%('_'.join(['B1', 'B1B1', 'B1B1B1']))))        

# plot position changes

motifs = ['M_B1', 'M_B1_B1', 'M_M_B1']
motifslist = [['B1', 'B1_B1', 'B1_B1_B1'], ['B2', 'B2_B2', 'B2_B2_B2'], ['M_B1', 'M_B1_B1', 'M_M_B1'], ['B2_M', 'B2_B2_M','B2_M_M'],
    ['M', 'M_M', 'M_M_M']]
for motifs in motifslist:
    context='wc'
    tectoData.plotPositionChangesHistogram(table, variant_table, motifs, helix_context=context)
    plt.savefig(os.path.join(figDirectory, 'position_changes.%s.context_%s.pdf'%('_'.join([m.replace('_', '')  for m in motifs]), context)))
# plot single variant motifs
tectoData.makePerSeqLengthDiagram(table, variant_table, variantDict.loc[('B1_B1_B1', 'AAA_')], figDirectory=figDirectory)

context = 'rigid'
for motif, seq in itertools.izip(['B1', 'B1_B1', 'B1_B1_B1'], ['A_', 'AA_', 'AAA_']):
    tectoData.makePerSeqLengthDiagram(table, variant_table, variantDict.loc[(context, motif, seq)].dropna().astype(int), colorBySs=True)
    tectoData.makeOffRateBarGraph(table, variant_table, variantDict.loc[(context, motif, seq)].dropna().astype(int), table_kinetics, variant_table_kinetics, times)
    plt.savefig(os.path.join(figDirectory, 'all_length_plots', 'junction.%s.by_length.%s.pdf'%(seq, context)))
for motif in ['M', 'M_M', 'M_M_M']:
    for seq in variantDict.loc[(context, motif)].index:
        tectoData.makePerSeqLengthDiagram(table, variant_table, variantDict.loc[(context, motif, seq)].dropna().astype(int), colorBySs=True)
        
        plt.savefig(os.path.join(figDirectory, 'all_length_plots', 'junction.%s.by_length.%s.pdf'%(seq, context)))
    tectoData.getSecondaryStructureDiagrams(table, variant_table, variantDict.loc[(motif, seq)], figDirectory)
    #tectoData.makePerSeqLengthDiagram(table, variant_table, variantDict.loc[(motif, seq)], figDirectory=figDirectory)

tectoData.makePerSeqLengthDiagram(table, variant_table, variantDict.loc[('_', 'wc')], figDirectory=figDirectory)

# plot on rate/off rate effects
for variant in variantDict.loc[('rigid', 'M_M_M', 'AAA_AAA')]:
    a = tectoData.loadTectoData(table, variant_table, variant, table_kinetics, variant_table_kinetics, times )
    print a.offrate_curves.loc[:, ['0']].mean()/a.offrate_curves.loc[:, ['39']].mean().values
    a.plotMedianNormalizedClustersOffrates()

# for those that we think we can measure: fold change > 10
variants = []
for variant in variant_table.index[variants[-1]:]:
    a = tectoData.loadTectoData(table, variant_table, variant, table_kinetics, variant_table_kinetics, times )
    if a.offrate_curves.loc[:, ['0']].mean().values[0]/a.offrate_curves.loc[:, ['39']].mean().values[0] > 10:
        variants.append(variant)
        a.findDeltaGDoubleDagger(table, variant_table,  table_kinetics, variant_table_kinetics)
        print variant

ddGs = pd.DataFrame(index = np.array(variants, dtype=int), columns=pd.MultiIndex.from_product([['affinity', 'offrate', 'onrate'],['dG', 'eminus', 'eplus']]))
for variant in variants[336:]:
    a = tectoData.loadTectoData(table, variant_table, variant, table_kinetics, variant_table_kinetics, times )
    ddG = a.findDeltaGDoubleDagger(table, variant_table,  table_kinetics, variant_table_kinetics)
    for param in ['affinity', 'offrate', 'onrate']:
        ddGs.loc[variant, param] = ddG.loc[param].values
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
im = ax.scatter(ddGs.loc[:, ('affinity', 'dG')], ddGs.loc[:, ('onrate', 'dG')], c=ddGs.loc[:, ('offrate', 'dG')], cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-1, 1)


#plt.errorbar(ddGs.loc[:, ('affinity', 'dG')], ddGs.loc[:, ('offrate', 'dG')],
#             xerr=[ddGs.loc[:, ('affinity', 'eminus')], ddGs.loc[:, ('affinity', 'eplus')]],
#             yerr=[ddGs.loc[:, ('offrate', 'eminus')],  ddGs.loc[:, ('offrate', 'eplus')]], fmt='.',
#             elinewidth=1, capsize=1, capthick=1, alpha=0.5, marker='s', markersize=4, color=sns.xkcd_rgb['charcoal'], linewidth=1)

index = (((ddGs.loc[:, ('affinity', 'eminus')] + ddGs.loc[:, ('affinity', 'eplus')]) < 1)&
         ((ddGs.loc[:, ('offrate', 'eminus')] + ddGs.loc[:, ('offrate', 'eplus')]) < 1))
x = ddGs.loc[index, ('affinity', 'dG')].astype(float)
y = ddGs.loc[index, ('offrate', 'dG')]
c = ddGs.loc[index, ('onrate', 'dG')]
slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
im = ax.scatter(x, y, c=c, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-1.5, 0.5)
ax.plot([-0.5, 1.5], [.5, -1.5], '0.75')
ax.plot(x, slope*x+intercept, ':', color='0.75')

p = numpy.poly1d(coeffs)
# fit values, and mean
x = ddGs.loc[:, ('affinity', 'dG')].astype(float)
y = ddGs.loc[:, ('offrate', 'dG')]
yhat = -x # assuming line of slope = -1                         # or [p(z) for z in x]
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
ssres = np.sum((y - yhat)**2)
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
rsquared = 1-ssres/sstot



# load variant table for wc wxperiment and plot scatterplot
filename = 'binding_curves_wc/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.abbrev.CPfitted'
table_wc = pd.read_table(filename, index_col=0)

index = table.index
xlim = ylim = [-14, -4]

index = index &~np.isnan(table.loc[:, 'dG'])&~np.isnan(table_wc.loc[:, 'dG'])

xlim =  [-11.5, -5.5]
ylim = [-11, -5]
index = ~np.isnan(table.loc[:, ['0','1']]).all(axis=1)&table.loc[:, 'barcode_good']
fig = plt.figure(figsize=(3.5,3.25))
gs = gridspec.GridSpec(1, 2,
                   width_ratios=[1,3],
                   wspace=0.025)
ax1 = fig.add_subplot(111, aspect='equal')
im = ax1.hexbin(table.loc[index, 'dG'], table_wc.loc[index, 'dG'], cmap='Greys', bins='log',
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]], gridsize=100)
ax1.set_xlabel('Flow -2GU dG (kcal/mol)')
ax1.set_ylabel('Flow -2AU dG (kcal/mol)')
plt.tight_layout()
plt.colorbar(im)
plt.savefig(os.path.join(figDirectory, 'hexbin.flow_pieces.log.pdf'))