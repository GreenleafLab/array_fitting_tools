import pandas as pd
import tectoData
import IMlibs
import functools
import multiprocessing
import datetime
import seaborn as sns
sys.path.insert(0, '/home/sarah/array_image_tools_SKD')
import fittingParameters
sns.set_style('white')
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4})
              
chip = 'AG3EL'
flow = 'WC'
dirname = '/lab/sarah/RNAarray/150605_onchip_binding'
stddirs = 'binding_curves/reduced_signals/barcode_mapping/'
figDirectory = os.path.join(dirname, flow, 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)
    


table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.CPfitted'), index_col=0)
variant_table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.perVariant.bootStrapped.CPfitted'), index_col=0)
table.dropna(axis=0, subset=['variant_number'], inplace=True)
concentrations = 2000*np.power(3., np.arange(0, -8, -1))[::-1]


# plot different n's
n = 10
plt.figure(figsize=(3,3));
x = np.linspace(0.4, 1.6)
sns.distplot(parameters.tight_binders_fit_params.loc[parameters.tight_binders_fit_params.number==n, 'fmax'],
             color='0.5', hist_kws={'histtype':'stepfilled'});
plt.plot(x, parameters.find_fmax_bounds_given_n(n, return_dist=True).pdf(x), color='r')
plt.annotate('n = %d'%n, xy=(.025, .975),
                xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=12)
ax = plt.gca(); ax.tick_params(right='off', top='off')
plt.xlabel('fit $f_{max}$')
plt.ylabel('probability')
plt.xlim([x[0], x[-1]])
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'histogram.fit_fmax.n_%d.pdf'%n));


parameters = fittingParameters.Parameters(concentrations, table=table)
table_dropped = table.drop(['222.2', '666.7','2000'], axis=1)
parameters.concentrations = concentrations[:-3]
variants = variant_table.loc[(variant_table.dG < -8.9)].sort('dG').iloc[::10].index
results = IMlibs.getBootstrappedErrors(table_dropped, parameters, 20, variants=variants)


index = results.index
cmap = sns.diverging_palette(20, 220, center="dark", as_cmap=True, )
c = results.flag
c.loc[variant_table.loc[index].flag == 1] = -1
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111, aspect='equal')
xlim = [-12, -8.5]

im = ax.scatter(variant_table.loc[index].dG, results.loc[index].dG, marker='.', alpha=0.5, s=8,
                c=c.astype(float), vmin=-1, vmax=1, cmap=cmap, linewidth=0)

plt.xticks(np.arange(xlim[0], xlim[1]))
plt.xlim(xlim); plt.xlabel('$\Delta$G (kcal/mol) all 8 concentrations')
plt.ylim(xlim); plt.ylabel('$\Delta$G (kcal/mol) 5 concentrations')
plt.plot(xlim, xlim, 'r', linewidth=0.5)
ax.tick_params(top='off', right='off')
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'scatterplot.first_five_concentrations_subsampled_10.pdf'));

# plot
binedges = np.arange(-12, -5, 0.25)
variant_table.loc[:, 'binned_dG'] = np.digitize(variant_table.dG, bins=binedges)
variant_table.loc[:, 'dG (kcal/mol)'] = np.nan
for idx, binname in itertools.izip(variant_table.index, variant_table.binned_dG):
    if binname == 0:
        variant_table.loc[idx, 'dG (kcal/mol)'] = '< %4.1f'%binedges[0]
    elif binname == len(binedges):
        variant_table.loc[idx, 'dG (kcal/mol)'] = '> %4.1f'%binedges[-1]
    else:
        variant_table.loc[idx, 'dG (kcal/mol)'] = '%4.2f to %4.2f'%(binedges[binname-1], binedges[binname])
order = (['< %4.1f'%binedges[0]] +
    ['%4.2f to %4.2f'%(binedges[binname-1], binedges[binname]) for binname in np.arange(1, len(binedges))] +
    ['> %4.1f'%binedges[-1]])
g = sns.factorplot(x="dG (kcal/mol)", y="diff_fmin", data=variant_table, kind='bar', order=order, color='seagreen', size=3, aspect=1.5)
g.set_xticklabels(rotation=90)


plt.figure(figsize=(3,3));
#plt.hist(results.dG, bins=np.arange(-12, -4, 0.25), histtype='stepfilled', alpha=0.5, color='0.5'); ax=plt.gca(); ylim = ax.get_ylim();
#plt.plot([np.percentile(results.dG, 2.5)]*2, ylim, ':', color='0.5');

plt.hist(results.loc[results.numClusters>=5].dG, bins=np.arange(-12, -4, 0.25), histtype='stepfilled', alpha=0.5, color='0.5'); ax=plt.gca(); ylim = ax.get_ylim();
plt.plot([np.percentile(results.loc[results.numClusters>=5].dG, 2.5)]*2, ylim,  ':', color='0.5');
plt.xlabel('$\Delta$G (kcal/mol)'); plt.tight_layout()
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.savefig(os.path.join(figDirectory, 'results.background.min_num_clusters_5.pdf'));

#fitUnconstrainedAbs = IMlibs.splitAndFit(bindingSeries, allClusterSignal, variantFittedFilename,
#                                              concentrations, parameters, numCores, index=bindingSeries.index)
detection_limit = -6.93
cmap = sns.diverging_palette(220, 20, center="dark", as_cmap=True)
index = variant_table.loc[variant_table.numClusters >= 5].index
xlim = [-12.5, -5]
fig = plt.figure(figsize=(4.5,3.75))
ax = fig.add_subplot(111, aspect='equal')
im = ax.scatter(variant_table.loc[index].dG_init, variant_table.loc[index].dG, marker='.', alpha=0.5,
                c=variant_table.loc[index].fmax_init, vmin=0.5, vmax=1.5, cmap=cmap, linewidth=0)
plt.plot(xlim, xlim, 'c:', linewidth=1)
plt.plot([detection_limit]*2, xlim, 'r:', linewidth=1)
plt.plot(xlim, [detection_limit]*2, 'r:', linewidth=1)
plt.xlim(xlim); plt.xlabel('$\Delta$G initial (kcal/mol)')
plt.ylim(-12.5, -5); plt.ylabel('$\Delta$G final (kcal/mol)')
plt.colorbar(im, label='fmax')
ax.tick_params(top='off', right='off')
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'inital_vs_final.colored_by_fmax_initial.min_num_clusters_5.png'));

# plot variation due to sequence
tectoData.plotSequenceJoe(variant_table)
plt.savefig(os.path.join(figDirectory, 'joes_Sequences.confInt.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'joes_Sequences.residuals.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'joes_Sequences_scatterplot.pdf'))

# plot length changes
tectoData.plotSequenceJoeLengthChanges(variant_table)
plt.savefig(os.path.join(figDirectory, 'joes_Sequences_length_changes.11G.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'joes_Sequences_length_changes.11T.pdf'))

# plot tiled length changes
tectoData.plotSequencesLengthChanges(variant_table)
plt.savefig(os.path.join(figDirectory, 'tiled_Sequences_length_changes.pdf'))

# plot histograms
tectoData.compareSeqTiled(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.tiled_random.length10.pdf'))

# plot tertiary contacts
for length in [8,9,10,11]:
    tectoData.plotTertiaryContacts(variant_table, length=length)
    plt.savefig(os.path.join(figDirectory, 'histogram.tertiary_contacts.length%d.pdf'%length))

plt.savefig(os.path.join(figDirectory, 'histogram.tertiary_contacts.length10.pdf'))

# plot noise in fmax
# plot noise in fmax
filterName = 'anyRNA'
a = IMlibs.loadCPseqSignal(os.path.join(dirname, flow, 'binding_curves', chip+'_tile003_Bottom_filtered.CPsignal'))
index = [str(s).find(filterName) == -1 for s in a.loc[:, 'filter']]
tectoData.plotAbsFluorescence(a, index)
plt.savefig(os.path.join(figDirectory, 'fabs.binding_series.null.pdf'))

# plot signal
tectoData.plotDeltaAbsFluorescence(a, filterName)
plt.savefig(os.path.join(figDirectory, 'fabs.binding_series.both.pdf'))

# plot RNA make
tectoData.plotThreeWayJunctions(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.threeways.2.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'histogram.threeways.1.pdf')); plt.close()