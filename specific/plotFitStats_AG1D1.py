import pandas as pd
import tectoData
import IMlibs
import functools
import multiprocessing
import datetime
import seaborn as sns
sns.set_style('white')

chip = 'AG3EL'
flow = 'WC'
dirname = '/lab/sarah/RNAarray/150605_onchip_binding'
stddirs = 'binding_curves/reduced_signals/barcode_mapping/'

table_old = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.CPfitted'), index_col=0)
variant_table_old = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.perVariant.bootStrapped.CPfitted'), index_col=0)
table_old.dropna(axis=0, subset=['variant_number'], inplace=True)

chip = 'AG1D1'
flow = 'WC'
dirname = '/lab/sarah/RNAarray/150607_chip'
stddirs = 'binding_curves/reduced_signals/barcode_mapping/'


figDirectory = os.path.join(dirname, flow, 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)

table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_ALL_filtered_anyRNA_sorted.annotated.CPfitted'), index_col=0)
variant_table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_ALL_filtered_anyRNA_sorted.annotated.perVariant.bootStrapped.CPfitted'), index_col=0)
table.dropna(axis=0, subset=['variant_number'], inplace=True)

# plot scatterplot
index = pd.concat([variant_table_old.pvalue <= 0.05,
                   variant_table_old.numClusters >= 5,
                   variant_table.pvalue <=  0.05,
                   variant_table.numClusters >= 5], axis=1).all(axis=1)
#index = pd.concat([variant_table_old.loc[variant_table_old.pvalue <= 0.05], variant_table], axis=1).dropna(subset=['dG'], axis=0).index
tectoData.plotScatterplotReplicates([variant_table_old, variant_table], labels= ['AG3EL', 'AG1D1'], index=index)
plt.savefig(os.path.join(figDirectory, 'replicates.residuals.AG3EL_vs_AG1D1.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'hexbin.replicates.AG3EL_vs_AG1D1.pdf'))

# plot the error in bins
IMlibs.plotErrorInBins(variant_table)
plt.savefig(os.path.join(figDirectory, 'error_in_bins.AG1D1.pdf'))

IMlibs.plotErrorInBins(variant_table_old)
plt.savefig(os.path.join(figDirectory, 'error_in_bins.AG3EL.pdf'))


index = pd.concat([variant_table_old.pvalue <= 0.05,
                   variant_table_old.numTests*variant_table_old.fitFraction >= 10,
                   variant_table.pvalue <= 0.05,
                   variant_table.numTests*variant_table.fitFraction >= 80], axis=1).all(axis=1)
tectoData.plotScatterplotReplicates([variant_table_old, variant_table], labels= ['AG3EL', 'AG1D1'], index=index)
plt.savefig(os.path.join(figDirectory, 'replicates.residuals.AG3EL_vs_AG1D1.numTestfiltered.bindingFiltered.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'hexbin.replicates.AG3EL_vs_AG1D1.numTestfiltered.bindingFiltered.pdf'))

# plot number of correct fits
IMlibs.plotFractionFit(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.fraction_fit.AG1D1.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'barplot.fraction_passed.AG1D1.pdf')); plt.close()

IMlibs.plotFractionFit(variant_table_old)
plt.savefig(os.path.join(figDirectory, 'histogram.fraction_fit.AG3EL.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'barplot.fraction_passed.AG3EL.pdf')); plt.close()

# plot number of clusters per variant
IMlibs.plotNumber(variant_table_old)
plt.savefig(os.path.join(figDirectory, 'histogram.clusters_per_variant.before_fit_filter.AG3EL.pdf')); 

IMlibs.plotNumber(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.clusters_per_variant.before_fit_filter.AG1D1.pdf')); 

variant = 176
a =tectoData.loadTectoData(table_old, variant_table_old, variant=variant)
a.plotClustersAll(fmax_lb=table_old.fmax.min())
plt.savefig(os.path.join(figDirectory, 'binding_curves.variant_%d.AG3EL.pdf'%variant)); 

a =tectoData.loadTectoData(table, variant_table, variant=variant)
a.plotClustersAll(fmax_lb=table.fmax.min())
plt.savefig(os.path.join(figDirectory, 'binding_curves.variant_%d.AG1D1.pdf'%variant)); 

# plot variance in fit parameters per cluster
tectoData.plotVarianceDataset(table)
plt.savefig(os.path.join(figDirectory, 'histogram.variance_dG.nan_barcode_filters.pdf'))


# plot confidence intervals
tectoData.plotConfIntervals(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.conf_interval.at_least_5variants.pdf'))

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
a = IMlibs.loadCPseqSignal(os.path.join(dirname, flow, 'binding_curves', chip+'_ALL_tile003_Bottom_filtered.CPsignal'))
index = [str(s).find(filterName) == -1 for s in a.loc[:, 'filter']]
tectoData.plotAbsFluorescence(a, index, concentrations=concentrations)
plt.savefig(os.path.join(figDirectory, 'fabs.binding_series.null.pdf'))

# plot signal
tectoData.plotDeltaAbsFluorescence(a, filterName)
plt.savefig(os.path.join(figDirectory, 'fabs.binding_series.both.pdf'))

# plot RNA make
tectoData.plotThreeWayJunctions(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.threeways.2.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'histogram.threeways.1.pdf')); plt.close()