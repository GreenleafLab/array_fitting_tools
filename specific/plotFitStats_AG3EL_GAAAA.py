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


flow = 'WC'
table_wc = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.CPfitted'), index_col=0)
variant_table_wc = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.perVariant.bootStrapped.CPfitted'), index_col=0)
table_wc.dropna(axis=0, subset=['variant_number'], inplace=True)

concentrations = 2000*np.power(3., np.arange(0, -6, -1))[::-1]
flow = 'WC_GAAAA'
table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.CPfitted'), index_col=0)
variant_table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.perVariant.bootStrapped.CPfitted'), index_col=0)
table.dropna(axis=0, subset=['variant_number'], inplace=True)

figDirectory = os.path.join(dirname, flow, 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)

# plot variance in fit parameters per cluster
tectoData.plotVarianceDataset(table)
plt.savefig(os.path.join(figDirectory, 'histogram.variance_dG.nan_barcode_filters.pdf'))


# plot confidence intervals
tectoData.plotConfIntervals(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.conf_interval.at_least_5variants.pdf'))

#plot scatterplot

index = variant_table.loc['dG_ub'] - variant_table
tectoData.plotScatterplotDelta([variant_table_wc, variant_table])
plt.xlim([-12, -8])
plt.ylim([-11, -7])
plt.xlabel('dG (kcal/mol) WC')
plt.ylabel('dG (kcal/mol) WC GAAAA')
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'scatterplot.WC_vs_WCGAAAA.pdf'))


plt.figure(); plt.hexbin(variant_table_wc.loc[index, 'dG'], variant_table.loc[index, 'dG'])

index = (table.barcode_good & (table.dG_var.astype(float) < 1) & (table.qvalue < 0.05) &
         (table_wc.dG_var.astype(float) < 1) & (table_wc.qvalue < 0.05))
plt.figure()
plt.hexbin(table.loc[index, 'dG'], table_wc.loc[index, 'dG'])
tectoData.plotScatterplot([table_wc.loc[index], table.loc[index]])

# plot variation due to sequence
tectoData.plotSequenceJoe(variant_table)
plt.savefig(os.path.join(figDirectory, 'joes_Sequences.confInt.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'joes_Sequences.residuals.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'joes_Sequences_scatterplot.pdf'))


# plot tertiary contacts

for length in [8,9,10,11]:
    tectoData.plotTertiaryContacts(variant_table, length=length)
    plt.savefig(os.path.join(figDirectory, 'histogram.tertiary_contacts.length%d.pdf'%length))


# plot noise in fmax
filterName = 'anyRNA'
a = IMlibs.loadCPseqSignal(os.path.join(dirname, flow, 'binding_curves', chip+'_tile003_Bottom_filtered.CPsignal'))
index = [str(s).find(filterName) == -1 for s in a.loc[:, 'filter']]
tectoData.plotAbsFluorescence(a, index, concentrations=concentrations)
plt.savefig(os.path.join(figDirectory, 'fabs.binding_series.null.pdf'))

# plot signal
tectoData.plotDeltaAbsFluorescence(a, filterName, concentrations)
plt.savefig(os.path.join(figDirectory, 'fabs.binding_series.both.pdf'))

# plot RNA make
tectoData.plotThreeWayJunctions(variant_table)
plt.savefig(os.path.join(figDirectory, 'histogram.threeways.2.pdf')); plt.close()
plt.savefig(os.path.join(figDirectory, 'histogram.threeways.1.pdf')); plt.close()