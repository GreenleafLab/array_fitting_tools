import datetime
sns.set_style("white", {'xtick.major.size': 4,  'ytick.major.size': 4,
                        'xtick.minor.size': 2,  'ytick.minor.size': 2,
                        'lines.linewidth': 1})

libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.txt'
libChar = pd.read_table(libCharFile).loc[:, :'ss_correct']

chip1 = 'AG3EL'
flow1 = 'WC'
dirname = '/lab/sarah/RNAarray/150605_onchip_binding'

outFile = os.path.join(dirname, flow1, 'bindingCurves', '%s_Bottom_filtered_reduced'%chip1)
variant_table_wc = IMlibs.loadLibCharVariantTable(libCharFile, outFile + '.CPvariant')

chip = 'AG1D1_ALL'
flow = 'WC'
dirname = '/lab/sarah/RNAarray/150607_chip/'

figDirectory = os.path.join(dirname, flow, 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)
    
outFile = os.path.join(dirname, flow, 'bindingCurves', '%s_Bottom_filtered_reduced'%chip)
variant_table = IMlibs.loadLibCharVariantTable(libCharFile, outFile + '.CPvariant')

variant_tables = [variant_table_wc, variant_table]
plotFun.plotReplicates(variant_tables, vmax=100)
plt.savefig(os.path.join(figDirectory,
                         'hexbin.%s_%s.vs.%s_%s.pdf'%(chip1, flow1, chip, flow)))

plotFun.plotReplicatesKd(variant_tables, scatter=True)
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.savefig(os.path.join(figDirectory,
                         'scatterplot.%s_%s.vs.%s_%s.pdf'%(chip1, flow1, chip, flow)))
plt.savefig(os.path.join(figDirectory,
                         'scatterplot.%s_%s.vs.%s_%s.png'%(chip1, flow1, chip, flow)))

plotFun.plotNumberTotal(variant_table_wc, variant_table2=variant_table)
plt.savefig(os.path.join(figDirectory,
                         'histogram.numTests.%s_%s.and.%s_%s.pdf'%(chip1, flow1, chip, flow)))


plotFun.plotResidualsKd(variant_tables)

## plot fraction within confidence intervals
variant_tables = [variant_table_wc, variant_table]

#variant_table.loc[:, ['dG_lb', 'dG', 'dG_ub']] -= offset

eminus = pd.concat([(table.dG - table.dG_lb) for table in variant_tables],
                     axis=1, keys=['rep1', 'rep2'])
eplus  = pd.concat([(table.dG_ub - table.dG) for table in variant_tables],
                     axis=1, keys=['rep1', 'rep2'])

combined = pd.concat([variant_table_wc.dG, variant_table.dG], axis=1,
    keys=['rep1', 'rep2'])

offset = 0.240
combined.loc[:, 'difference'] = combined.loc[:, 'rep2'] - combined.loc[:, 'rep1']
combined.loc[:, 'eplus'] = np.sqrt((eplus**2).sum(axis=1))
combined.loc[:, 'eminus'] = np.sqrt((eminus**2).sum(axis=1))
combined.loc[:, 'within_bound'] = ((combined.difference - offset - combined.eminus <= 0)&
                    (combined.difference - offset + combined.eplus >= 0))


binedges = np.arange(-12, -5, 0.5)
combined.loc[:, 'rep1_bin'] = np.digitize(combined.loc[:, 'rep1'], binedges)
combined.loc[:, 'rep1_n'] = variant_table_wc.numTests
combined.loc[:, 'rep2_n'] = variant_table.numTests

binedges = np.arange(0, 220, 20)
combined.loc[:, 'rep2_n_bin'] = np.digitize(combined.loc[:, 'rep2_n'], binedges)
# maximize those within bound
index = (combined.loc[:, ['rep1_n', 'rep2_n']] >=5).all(axis=1)

plt.figure(figsize=(4,3));
x = np.arange(25)
sns.pointplot(x="rep2_n", y="within_bound", data=combined.loc[combined.rep1_n>=5],
              order=x, linestyles=["none"], markers=['>'])
ax = plt.gca()
majorLocator   = mpl.ticker.MultipleLocator(5)
majorFormatter = mpl.ticker.FormatStrFormatter('%d')
minorLocator   = mpl.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(top='off', right='off')
ax.tick_params(which='minor', top='off', right='off')
plt.xlabel('number of measurements')
plt.ylabel('fraction not different from replicate')
plt.ylim(0, 1)
plt.tight_layout()

g = sns.JointGrid(x="rep1_n", y="rep2_n", data=combined, size=4,
                  xlim=[0, 100], ylim=[0, 300])
g = g.plot_joint(plt.hexbin, extent=[0, 100, 0, 300], cmap='Spectral_r',
                 mincnt=1, gridsize=75)
g = g.plot_marginals(sns.distplot, hist_kws={'histtype':'stepfilled'},
                     bins=np.arange(0, 300), color='0.5', kde_kws={'clip':[0,300]})

# plot errors for each

binedges = [-15, -10.5, -9.5, -9,.0 -8.5, -8.0, -7.5, -7.0]
binlabels = np.array(['min', '-10.5', ' -9.5', ' -9.0', ' -8.5', ' -8.0', ' -7.5', ' -7.0', 'max']).astype(str)
combined.loc[:, 'rep1_bin'] = np.digitize(combined.loc[:, 'rep1'], binedges)

plt.figure(figsize=(5,3))
marker_styles = ['.', '<', 's', '*',]*2
colors = sns.color_palette('Spectral', n_colors=len(binedges))
for i, bin_idx in enumerate(np.arange(1, len(binedges))):
    grouped = combined.loc[combined.rep1_bin==bin_idx].groupby('rep1_n')
    error = grouped['error_rep1'].mean().reset_index()
    index = error.rep1_n >= 5
    x = error.reset_index
    plt.scatter(error.loc[index].rep1_n, error.loc[index].error_rep1,
                label='%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx]),
                marker=marker_styles[i], c=colors[i], s=10)
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 1.2)
plt.ylabel('width of confidence interval (kcal/mol)')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figure(figsize=(5,3))
plt.hist(combined.rep1_n, bins=np.arange(5, 100), histtype='stepfilled', alpha=0.5, color='0.5')
plt.xlim(0, 100)
plt.ylabel('number of variants')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])


x = np.arange(1, 100)
plt.figure(figsize=(5,3))
marker_styles = ['.', '<', 's', '*',]*2
colors = sns.color_palette('Spectral', n_colors=len(binedges))
for i, bin_idx in enumerate(np.arange(1, len(binedges))):
    
    vec = combined.loc[combined.rep1_bin==bin_idx].rep1_n
    
    kernel = st.gaussian_kde(vec)
    plt.plot(x, kernel(x)*len(vec), color=colors[i],
             label='%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx]),)
plt.legend()
plt.xlim(0, 100)
plt.ylabel('number of variants')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# do it for the other replicate
binedges = [-15, -10.5, -9.5, -9,.0 -8.5, -8.0, -7.5, -7.0]
binlabels = np.array(['min', '-10.5', ' -9.5', ' -9.0', ' -8.5', ' -8.0', ' -7.5', ' -7.0', 'max']).astype(str)
combined.loc[:, 'rep2_bin'] = np.digitize(combined.loc[:, 'rep2'], binedges)

plt.figure(figsize=(5,3))
marker_styles = ['.', '<', 's', '*',]*2
colors = sns.color_palette('Spectral', n_colors=len(binedges))
for i, bin_idx in enumerate(np.arange(1, len(binedges))):
    grouped = combined.loc[combined.rep2_bin==bin_idx].groupby('rep2_n')
    error = grouped['error_rep1'].mean().reset_index()
    index = error.rep2_n >= 5
    x = error.reset_index
    plt.scatter(error.loc[index].rep2_n, error.loc[index].error_rep1,
                label='%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx]),
                marker=marker_styles[i], c=colors[i], s=10)
plt.legend()
plt.xlim(0, 200)
plt.ylim(0, 1.2)
plt.ylabel('width of confidence interval (kcal/mol)')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figure(figsize=(5,3))
plt.hist(combined.rep2_n, bins=np.arange(5, 100), histtype='stepfilled', alpha=0.5, color='0.5')
plt.xlim(0, 100)
plt.ylabel('number of variants')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])


x = np.arange(1, 200)
plt.figure(figsize=(5,3))
marker_styles = ['.', '<', 's', '*',]*2
colors = sns.color_palette('Spectral', n_colors=len(binedges))
for i, bin_idx in enumerate(np.arange(1, len(binedges))):
    
    vec = combined.loc[combined.rep2_bin==bin_idx].rep2_n
    
    kernel = st.gaussian_kde(vec.dropna())
    plt.plot(x, kernel(x)*len(vec.dropna()), color=colors[i],
             label='%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx]),)
plt.legend()
plt.xlim(0, 200)
plt.ylabel('number of variants')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# how much do you expect to overlap??
delta1 = (eplus.rep1 + eminus.rep1)/2.
stds1 = delta1/1.96
mean1 = pd.Series([st.norm.rvs(loc=0, scale=std) if not np.isnan(std) else np.nan for std in stds1], index=stds1.index)

delta2 = (eplus.rep2 + eminus.rep2)/2.
stds2 = delta2/1.96
mean2 = pd.Series([st.norm.rvs(loc=0, scale=std) if not np.isnan(std) else np.nan for std in stds2], index=stds1.index)

np.abs(mean1 - mean2) < np.sqrt(delta1**2 + delta2**2)


# plot all errors vs n

plt.figure(figsize=(5,3))
grouped = combined.groupby('rep1_n')
error = grouped['error_rep1'].mean().reset_index()
index = error.rep1_n >= 5
x = error.reset_index
eminus = pd.Series(index=error.rep1_n)
eplus = pd.Series(index=error.rep1_n)
for name, group in grouped['error_rep1']:
    print name
    try:
        lowerbound, upperbound = bootstrap.ci(group, n_samples=100)
        eminus.loc[name] = group.mean() - lowerbound
        eplus.loc[name] = -group.mean() + upperbound
    except:
        print 'issue with %d'%name
plt.errorbar(error.loc[index].rep1_n, error.loc[index].error_rep1,
             yerr=[eminus.loc[error.loc[index].rep1_n], eplus.loc[error.loc[index].rep1_n]],
             fmt=',', ecolor='k', linewidth=1)
plt.scatter(error.loc[index].rep1_n, error.loc[index].error_rep1,
            label='%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx]),
            marker=marker_styles[i], c=colors[i], s=10)
plt.xlim(0, 80)
plt.ylim(0, 1.2)
plt.ylabel('width of confidence interval (kcal/mol)')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])


        
plt.figure(figsize=(5,3))
grouped = combined.groupby('rep2_n')
error = grouped['error_rep2'].mean().reset_index()
index = error.rep2_n >= 15
x = error.reset_index
eminus = pd.Series(index=error.rep2_n)
eplus = pd.Series(index=error.rep2_n)
for name, group in grouped['error_rep1']:
    print name
    try:
        lowerbound, upperbound = bootstrap.ci(group, n_samples=100)
        eminus.loc[name] = group.mean() - lowerbound
        eplus.loc[name] = -group.mean() + upperbound
    except:
        print 'issue with %d'%name
plt.errorbar(error.loc[index].rep2_n, error.loc[index].error_rep2,
             yerr=[eminus.loc[error.loc[index].rep2_n], eplus.loc[error.loc[index].rep2_n]],
             fmt=',', ecolor='k', linewidth=1)
plt.scatter(error.loc[index].rep2_n, error.loc[index].error_rep2,
            label='%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx]),
            marker=marker_styles[i], c=colors[0], s=10)
plt.xlim(0, 200)
plt.ylim(0, 2)
plt.ylabel('width of confidence interval (kcal/mol)')
plt.xlabel('number of measurements')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()
ax.set_position([0.2, 0.25, 0.5, 0.65])

# for at least 5 measurements, what is  error in dG bins
binedges = [-15, -11.5, -11, -10.5, -10, -9.5, -9,.0 -8.5, -8.0, -7.5, -7.0, -4]
binlabels = np.array(['min', '-11.5', '-11.0', '-10.5',  '-10.0', ' -9.5', ' -9.0', ' -8.5', ' -8.0', ' -7.5', ' -7.0', 'max']).astype(str)
combined.loc[:, 'rep1_bin'] = np.digitize(combined.loc[:, 'rep1'], binedges)
grouped = combined.loc[combined.rep1_n>=5].groupby('rep1_bin')['error_rep1']
x = grouped.mean().index
eminus = pd.Series(index=x)
eplus = pd.Series(index=x)
for name, group in grouped:
    print name
    try:
        lowerbound, upperbound = bootstrap.ci(group.dropna(), n_samples=100)
        eminus.loc[name] = group.mean() - lowerbound
        eplus.loc[name] = -group.mean() + upperbound
    except:
        print 'issue with %d'%name
plt.figure(figsize=(3,4))    
plt.errorbar(x, grouped.mean(),
             yerr=[eminus, eplus],
             fmt=',', ecolor='k', linewidth=1, capsize=0, capthick=0)
plt.scatter(x, grouped.mean(),
            marker=marker_styles[i], c=colors[1], s=10)
plt.xlim(0, len(x))
plt.ylim(0, 1.2)
plt.xticks(x, ['%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx])
                for bin_idx in np.arange(1, len(binlabels))], rotation=90)
plt.ylabel('width of confidence interval (kcal/mol)')
plt.xlabel('measured affinity')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()

# for at least 5 measurements, what is  error in dG bins
binedges = [-15, -11.5, -11, -10.5, -10, -9.5, -9,.0 -8.5, -8.0, -7.5, -7.0, -4]
binlabels = np.array(['min', '-11.5', '-11.0', '-10.5',  '-10.0', ' -9.5', ' -9.0', ' -8.5', ' -8.0', ' -7.5', ' -7.0', 'max']).astype(str)
combined.loc[:, 'rep2_bin'] = np.digitize(combined.loc[:, 'rep2'], binedges)
grouped = combined.loc[combined.rep2_n>=15].groupby('rep2_bin')['error_rep2']
x = grouped.mean().index
eminus = pd.Series(index=x)
eplus = pd.Series(index=x)
for name, group in grouped:
    print name
    try:
        lowerbound, upperbound = bootstrap.ci(group.dropna(), n_samples=100)
        eminus.loc[name] = group.mean() - lowerbound
        eplus.loc[name] = -group.mean() + upperbound
    except:
        print 'issue with %d'%name
plt.figure(figsize=(3,4))    
plt.errorbar(x, grouped.mean(),
             yerr=[eminus, eplus],
             fmt=',', ecolor='k', linewidth=1, capsize=0, capthick=0)
plt.scatter(x, grouped.mean(),
            marker=marker_styles[i], c=colors[1], s=10)
plt.xlim(0, len(x))
plt.ylim(0, 2)
plt.xticks(x, ['%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx])
                for bin_idx in np.arange(1, len(binlabels))], rotation=90)
plt.ylabel('width of confidence interval (kcal/mol)')
plt.xlabel('measured affinity')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()


# for at least 5 measurements, what is  error in dG bins
binedges = [-15, -11.5, -11, -10.5, -10, -9.5, -9,.0 -8.5, -8.0, -7.5, -7.0, -4]
binlabels = np.array(['min', '-11.5', '-11.0', '-10.5',  '-10.0', ' -9.5', ' -9.0', ' -8.5', ' -8.0', ' -7.5', ' -7.0', 'max']).astype(str)
combined.loc[:, 'rep1_bin'] = np.digitize(combined.loc[:, 'rep1'], binedges)
grouped = combined.loc[(combined.rep2_n>=5)&(combined.rep1_n>=5)].groupby('rep1_bin')['within_bound']
x = grouped.mean().index
eminus = pd.Series(index=x)
eplus = pd.Series(index=x)
for name, group in grouped:
    print name
    try:
        lowerbound, upperbound = bootstrap.ci(group.dropna(), n_samples=100)
        eminus.loc[name] = group.mean() - lowerbound
        eplus.loc[name] = -group.mean() + upperbound
    except:
        print 'issue with %d'%name
plt.figure(figsize=(3,4))    
plt.errorbar(x, grouped.mean(),
             yerr=[eminus, eplus],
             fmt=',', ecolor='k', linewidth=1, capsize=0, capthick=0)
plt.scatter(x, grouped.mean(),
            marker=marker_styles[i], c=colors[1], s=10)
plt.xlim(0, len(x))
plt.ylim(0.55, 0.95)
plt.xticks(x, ['%s$\leq \Delta G \leq$%s'%(binlabels[bin_idx-1], binlabels[bin_idx])
                for bin_idx in np.arange(1, len(binlabels))], rotation=90)
plt.ylabel('fraction not different from replicate')
plt.xlabel('measured affinity')
ax = plt.gca()
ax.tick_params(top='off', right='off')
plt.tight_layout()