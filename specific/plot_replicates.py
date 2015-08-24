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