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

chip = 'AG3EL'
flow = 'WC_GAAAA'
dirname = '/lab/sarah/RNAarray/150605_onchip_binding/'

figDirectory = os.path.join(dirname, flow, 'figs_'+str(datetime.date.today()))
if not os.path.exists(figDirectory):
    os.mkdir(figDirectory)
    
outFile = os.path.join(dirname, flow, 'bindingCurves', '%s_Bottom_filtered_reduced'%chip)
variant_table = IMlibs.loadLibCharVariantTable(libCharFile, outFile + '.CPvariant')

variant_tables = [variant_table_wc, variant_table]

# plot e
    
index = ((pd.concat(variant_tables, axis=1).loc[:, 'dG'] < -7).all(axis=1)&
    ((pd.concat(variant_tables, axis=1).loc[:, 'numTests']) >=5).all(axis=1))
    
ddG = variant_table.dG - variant_table_wc.dG

colors =  ["#e74c3c", "#3498db", "#34495e", "#9b59b6","#2ecc71"]
fig = plt.figure(figsize=(4, 2.5))
ax = fig.add_subplot(111)
for i, length in enumerate([10, 9, 11]):
    index2 = index&(variant_tables[0].length == length)
    sns.distplot(ddG.loc[index2], label='%dbp'%length, color=colors[i])
plt.legend()
ax.tick_params(right='off', top='off')
plt.xlabel('$\Delta\Delta$G WC_GAAAA - WC (kcal/mol)')
plt.ylabel('probability')
plt.tight_layout()
