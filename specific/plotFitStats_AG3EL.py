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
figDirectory = os.path.join(dirname, flow, 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)
    


table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.abbrev.CPfitted'), index_col=0)
variant_table = pd.read_table(os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.perVariant.bootStrapped.CPfitted'), index_col=0)
table.dropna(axis=0, subset=['variant_number'], inplace=True)
concentrations = 2000*np.power(3., np.arange(0, -8, -1))[::-1]

# fit medians
variant_table, bindingSeries, allClusterSignal,  bindingSeriesNorm = IMlibs.findVariantTable(table, concentrations=concentrations)

numCores = 20
variantFittedFilename = os.path.join(dirname, flow, stddirs, chip+'_filtered_anyRNA_sorted.annotated.perVariant.CPfitted')
null_scores = bindingSeries.iloc[:, -2]
parameters = fittingParameters.Parameters(concentrations, bindingSeries.iloc[:,-1], allClusterSignal.loc[:, 'all_cluster_signal'], null_scores.values)
fitUnconstrainedAbs = IMlibs.splitAndFit(bindingSeries, allClusterSignal, variantFittedFilename,
                                              concentrations, parameters, numCores, index=bindingSeries.index)

parameters.fitParameters.loc['upperbound', 'fmax'] = 100*bindingSeriesNorm.loc[np.isfinite(bindingSeriesNorm).all(axis=1)].max(axis=1).max()
fitUnconstrained = IMlibs.splitAndFit(bindingSeriesNorm, pd.Series(data=1, index=bindingSeriesNorm.index), variantFittedFilename,
                                              concentrations, parameters, numCores, index=bindingSeriesNorm.index)



IMlibs.plotFitFmaxs(fitUnconstrainedAbs)
plt.title('absolute fluorescence'); plt.xlim(0, 3000); plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'constrained_fmax.AG3EL.absolute_fluorescence.pdf'))
IMlibs.plotFitFmaxs(fitUnconstrained)
plt.title('normalized fluorescence'); plt.xlim(0, 1.7); plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'constrained_fmax.AG3EL.normalized_fluorescence.pdf'))

# plot those with lower bounds
index = ((fitUnconstrainedAbs.dG < parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.5, 2000)))&
         (fitUnconstrainedAbs.dG > parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.9, 2000))))
IMlibs.plotFitFmaxs(fitUnconstrainedAbs, index=index)
plt.title('absolute fluorescence'); plt.xlim(0, 3000); plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'constrained_fmax.AG3EL.absolute_fluorescence.pdf'))
IMlibs.plotFitFmaxs(fitUnconstrained)
plt.title('normalized fluorescence'); plt.xlim(0, 1.7); plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'constrained_fmax.AG3EL.normalized_fluorescence.pdf'))

compareResults.plotContour(x=fitUnconstrained.dG, y=fitUnconstrained.fmax, xlim=[-12, -6], ylim=[0, 2], aspect='auto', labels=['dG', 'fmax'], min_value=5, plot_points=False)
plt.savefig(os.path.join(figDirectory, 'unconstrained_fmax_vs_dG.normalizes_fluorescence.pdf'))
compareResults.plotContour(x=fitUnconstrainedAbs.dG, y=fitUnconstrainedAbs.fmax, xlim=[-12, -6], ylim=[0, 3000], aspect='auto', labels=['dG', 'fmax'], min_value=5, plot_points=False)
plt.savefig(os.path.join(figDirectory, 'unconstrained_fmax_vs_dG.absolute_fluorescence.pdf'))

# do constrained fit
maxdG = parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.9, concentrations[args.null_column]))
parameters.fitParameters.loc[:, 'fmax'] = IMlibs.plotFitFmaxs(fitUnconstrained, maxdG=maxdG)
fitConstrained = IMlibs.splitAndFit(bindingSeriesNorm, pd.Series(data=1, index=bindingSeriesNorm.index), variantFittedFilename,
                                              concentrations, parameters, numCores, index=bindingSeriesNorm.index)
compareResults.plotContour(x=fitConstrained.dG, y=fitConstrained.fmax, xlim=[-12, -6], ylim=[0, 2], aspect='auto', labels=['dG', 'fmax'], min_value=5, plot_points=False)
plt.savefig(os.path.join(figDirectory, 'constrained_fmax_vs_dG.normalized_fluorescence.pdf'))



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