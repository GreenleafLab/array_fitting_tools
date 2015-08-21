
# run findSeqDistribution on old library with new chip
# %run ~/array_image_tools_SKD/scripts/findSeqDistribution.py -b 150608_barcode_mapping_lib2/tecto_lib2.sort.unique_barcodes -l 140929_library_design/allJunctions.noUs.noMB2.characterization -o previous_chip
lib_ilocs = np.unique(is_designed[is_designed != -1])

old_lib_variants = designed_library_unique.iloc[lib_ilocs].variant_number.values

seq_ilocs = np.array([np.where(is_designed == lib_iloc)[0][0] for lib_iloc in lib_ilocs])

# check
[consensus.iloc[seq].sequence.find(seqfun.reverseComplement(designed_library_unique.iloc[lib].sequence)) for lib, seq in itertools.izip(lib_ilocs, seq_ilocs)]

# run again
barcodeMapNew = pd.read_table('150605_onchip_binding/WC/binding_curves/reduced_signals/barcode_mapping/AG3EL_filtered_anyRNA_sorted.barcode_to_seq', index_col='barcode')
new_lib_variants = np.array([barcodeMapNew.loc[consensus.iloc[seq].barcode].variant_number for seq in seq_ilocs])

# save
variantDict = pd.DataFrame(np.column_stack([new_lib_variants, old_lib_variants]), columns=['new_lib', 'old_lib'])
variantDict.to_csv('150605_onchip_binding/variants.new_to_old.dat', index=False)

# now load both
dirname = '/raid1/lab/sarah/RNAarray/150605_onchip_binding/WC/'
file_old = os.path.join(dirname, '../../141111_miseq_run_tecto_TAL_VR/with_all_clusters/binding_curves_wc/reduced_signals/barcode_mapping/AAYFY_ALL_filtered_tecto_sorted.annotated.perVariant.CPfitted')
file_new = os.path.join(dirname,'binding_curves/reduced_signals/barcode_mapping/AG3EL_filtered_anyRNA_sorted.annotated.perVariant.CPfitted')

variant_table_old = pd.read_table(file_old, index_col='variant_number')
variant_table_new = pd.read_table(file_new, index_col='variant_number')
for variant_table in [variant_table_old, variant_table_new]:
    variant_table.loc[:, 'variant_number'] = variant_table.index

variantDict = pd.read_table('/raid1/lab/sarah/RNAarray/150605_onchip_binding/variants.new_to_old.dat')

compare = {}
cols = ['dG', 'dG_ub', 'dG_lb', 'numTests', 'fitFraction', 'pvalue', 'variant_number']
compare['new'] = pd.DataFrame(variant_table_new.loc[variantDict.dropna().new_lib, cols].values, columns=cols)
compare['old'] = pd.DataFrame(variant_table_old.loc[variantDict.dropna().old_lib, cols].values, columns=cols)
compare = pd.concat(compare, axis=1)

# plot
min_numtests = 5
index = (pd.concat([compare.loc[:, (key, 'numTests')]*compare.loc[:, (key, 'fitFraction')] >= min_numtests for key in ['new', 'old']], axis=1).all(axis=1)&
         pd.concat([compare.loc[:, (key, 'pvalue')] <= 0.05 for key in ['new', 'old']], axis=1).all(axis=1))

x = compare.loc[index, ('new', 'dG')]
y = compare.loc[index, ('old', 'dG')]
c = np.sqrt(np.column_stack([(compare.loc[index, (key, 'dG_ub')] - compare.loc[index, (key, 'dG_lb')])**2 for key in ['new', 'old']]).sum(axis=1))

slope, intercept, r_value, p_value, std_err = st.linregress(x,y)

fig = plt.figure(figsize=(3.5,3.5))
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(x, y, vmin=-1, vmax=1.5, cmap='coolwarm', edgecolors='k', facecolors='none', marker='.')
xlim = np.array([-11, -8])
ax.plot(xlim, xlim, 'k--', linewidth=1)
ax.plot(xlim, xlim*slope + intercept, 'r:', linewidth=1)
ax.set_xlim(xlim)
ax.set_ylim(xlim)
ax.set_xlabel('dG new (kcal/mol)')
ax.set_ylabel('dG old (kcal/mol)')
plt.tight_layout()
plt.savefig(os.path.join(figDirectory, 'rsquareds.pdf'))

# how much would we expect given confidence intervals?
key = 'old'
x = compare.loc[index, (key, 'dG')]

compare.loc[:, (key, 'sigma')] = (compare.loc[:, (key, 'dG_ub')] -compare.loc[:, (key, 'dG_lb')])/2/1.96
num_iter = 1000
rsqreds = np.zeros(num_iter)
for i in range(num_iter): 
    newy = np.random.normal(loc=compare.loc[index, (key, 'dG')],
                 scale=compare.loc[index, (key, 'sigma')])
    index2 = np.isfinite(newy)
    slope, intercept, r_value, p_value, std_err = st.linregress(x.loc[index2],newy[index2])
    rsqreds[i] = r_value**2

slope, intercept, r_value, p_value, std_err = st.linregress(x,newy)

fig = plt.figure(figsize=(3.5,3.5))
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(x, newy, vmin=-1, vmax=1.5, cmap='coolwarm', edgecolors='k', facecolors='none')
xlim = np.array([-11, -8])
ax.plot(xlim, xlim, 'k--', linewidth=1)
ax.plot(xlim, xlim*slope + intercept, 'r:', linewidth=1)
ax.set_xlim(xlim)
ax.set_ylim(xlim)