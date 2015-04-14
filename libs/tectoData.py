import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scikits.bootstrap import bootstrap
import os
import variantFun
import hjh.tecto_assemble
from  hjh.junction import Junction
import itertools

class Parameters():
    def __init__(self):
        # save the units of concentration given in the binding series
        self.RT = 0.582  # kcal/mol at 20 degrees celsius
        self.min_deltaG = -12
        self.max_deltaG = -3
        self.concentration_units = 1E-9
        self.concentrations = np.array([2000./np.power(3, i) for i in range(8)])[::-1]
        self.lims = {'qvalue':(2e-4, 1e0),
                    'dG':(-12, -4),
                    'koff':(1e-5, 1E-2),
                    'kon' :(1e1, 1e6),
                    'kobs':(1e-5, 1e-2)}

class AffinityData():
    def __init__(self, sub_table_affinity, sub_variant_table_affinity, sub_table_offrates=None, sub_variant_table_offrates=None,times=None): 
        # affinity params general
        self.concentrations = 2000./np.power(3, np.arange(8))[::-1]
        self.more_concentrations = np.logspace(-2, 4, 50)        
        self.RT = 0.582  # kcal/mol at 20 degrees celsius
        
        # affinity params specific
        index = (~np.isnan(sub_table_affinity.loc[:, ['0', '1']]).all(axis=1)&
                 sub_table_affinity.loc[:, 'barcode_good'])
        
        self.binding_curves = sub_table_affinity.loc[index, np.arange(8).astype(str)]
        self.normalized_binding_curves = self.binding_curves.copy()
        for i in np.arange(8).astype(str):
            self.normalized_binding_curves.loc[:, i] = (sub_table_affinity.loc[index, i] - sub_table_affinity.loc[index, 'fmin'])/ sub_table_affinity.loc[index, 'fmax']
        
        self.clusters = sub_table_affinity.index
        self.fit_clusters = index
        

        self.affinity_params = sub_variant_table_affinity.loc['dG':]
        self.affinity_params.loc['eminus'] = sub_variant_table_affinity.loc['dG'] - sub_variant_table_affinity.loc['dG_lb']
        self.affinity_params.loc['eplus'] = sub_variant_table_affinity.loc['dG_ub'] - sub_variant_table_affinity.loc['dG']
        self.affinity_params_cluster = sub_table_affinity.loc[index, 'fmax':'fmin_var']
        
        # sequnece parameters
        if 'ss' in sub_variant_table_affinity.index:
            self.seq_params =  sub_variant_table_affinity.loc['sequence':'ss']
        else:
            self.seq_params =  sub_variant_table_affinity.loc['sequence':'total_length']
            
        # off rates
        if sub_table_offrates is not None and sub_variant_table_offrates is not None and times is not None:
            index = (~np.isnan(sub_table_offrates.loc[:, ['0', '1']]).all(axis=1)&
                     sub_table_offrates.loc[:, 'barcode_good'])
            self.fit_clusters_offrates = index
            self.times = times
            self.times_cluster = pd.DataFrame(index=index.loc[index].index,
                                              data=times.loc[sub_table_offrates.loc[index, 'tile']].values,
                                              columns=times.columns)
            num_time_points = self.num_time_points = times.shape[1]
            self.offrate_curves = sub_table_offrates.loc[index, np.arange(num_time_points).astype(str)]
            
            self.normalized_offrate_curves = self.offrate_curves.copy()
            for i in np.arange(num_time_points).astype(str):
                self.normalized_offrate_curves .loc[:, i] = (sub_table_offrates.loc[index, i] - sub_table_offrates.loc[index, 'fmin'])/ sub_table_offrates.loc[index, 'fmax']
            
            self.offrate_params = sub_variant_table_offrates.loc['toff':]
            self.offrate_params.loc['eminus'] = sub_variant_table_offrates.loc['toff'] - sub_variant_table_offrates.loc['toff_lb']
            self.offrate_params.loc['eplus'] = sub_variant_table_offrates.loc['toff_ub'] - sub_variant_table_offrates.loc['toff']
            self.offrate_params_cluster = sub_table_offrates.loc[index, 'fmax':'fmin_var']
            self.more_times = np.linspace(0, times.max().max(), 100)
            
            # on rate params
            self.onrate_params = pd.Series(index=['kon', 'eminus', 'eplus'])
            self.onrate_params.loc['kon'] = self.findOnRate()
            for e in ['eminus', 'eplus']:
                self.onrate_params.loc[e] = self.findOnRateError(self.offrate_params.loc[e], self.affinity_params.loc[e])
    
    def findNormalizedError(self):
        ybounds = pd.DataFrame(index=np.arange(8).astype(str), columns=['fnorm_lb', 'fnorm_ub'])
        for i in range(8):
            ybounds.loc[str(i)] = bootstrap.ci(self.normalized_binding_curves.loc[:, str(i)], statfunction=np.median)
        yerr = pd.DataFrame(index=np.arange(8).astype(str), columns=['eminus', 'eplus'])
        yerr.loc[:, 'eminus'] = self.normalized_binding_curves.median(axis=0) - ybounds.loc[:, 'fnorm_lb']
        yerr.loc[:, 'eplus'] = ybounds.loc[:, 'fnorm_ub'] - self.normalized_binding_curves.median(axis=0) 
        
        return yerr
    
    def findNormalizedErrorOffrates(self ):
        maxTime = np.ceil(self.times.max().max())
        minTime = np.floor(self.times.min().min())
        timeBins = np.linspace(0, maxTime, self.num_time_points*0.75)
        #timeBins = np.logspace(-1, np.log10(maxTime), 20 )
        timeBinCenters = (timeBins[:-1] + timeBins[1:])*0.5

        ybounds = pd.DataFrame(index=timeBinCenters, columns=['fnorm_lb', 'fnorm_ub', 'fnorm']) 
        for binLeft, binRight in itertools.izip(timeBins[:-1], timeBins[1:]):
            loc = (binLeft + binRight)*0.5
            time_points = ((self.times_cluster < binRight)&(self.times_cluster >= binLeft))
            ybounds.loc[loc, 'fnorm'] = self.normalized_offrate_curves[time_points].mean(axis=1).median()
            if not np.isnan(ybounds.loc[loc, 'fnorm']):
                ybounds.loc[loc, ['fnorm_lb', 'fnorm_ub']] = bootstrap.ci(self.normalized_offrate_curves[time_points].mean(axis=1).dropna(), statfunction=np.median, method='pi')        
        return ybounds
    
    def plotHistogram(self, parameter=None, plot_kde=None, binsize=None):
        if parameter is None: parameter = 'dG'
        if binsize is None: binsize=0.25
        if plot_kde is None: plot_kde = False
        fig = plt.figure(figsize=(3.5, 3.25))
        ax = fig.add_subplot(111)

        if plot_kde:
            sns.kdeplot(self.affinity_params_cluster.loc[:, parameter],
                        kernel='gau',
                        shade=True,
                        color=sns.xkcd_rgb['vermillion'], ax=ax, linewidth=1)
            normed = True
        else: normed=False
        ax.hist(self.affinity_params_cluster.loc[:, parameter], bins = np.arange(-14, -2, binsize),
                 alpha=0.5, facecolor=sns.xkcd_rgb['vermillion'], normed=normed)
        ymax = np.max([ax.get_ylim()[-1], 5])
        
        ax.fill_between([self.affinity_params.loc['dG_lb'],
                         self.affinity_params.loc['dG_ub']],  0, ymax, color='0.5', alpha=0.5)
        ax.set_xlim(-12, -4)
        ax.set_ylim(0, ymax)
        ax.set_xlabel('dG (kcal/mol)')
        ax.set_ylabel('number of clusters')
        plt.tight_layout()
        return ax
        

    def plotMedianNormalizedClusters(self, plot_confidence_intervals=None, annotate=None):
        if plot_confidence_intervals is None:
            plot_confidence_intervals = True
        if annotate is None:
            annotate = True 

        fig = plt.figure(figsize=(3.75, 3.25))
        ax = fig.add_subplot(111)
        xvalues = self.concentrations
        yvalues = self.normalized_binding_curves.median(axis=0)
        yerr = self.findNormalizedError().transpose().astype(float)
        ax.errorbar(xvalues, self.normalized_binding_curves.median(axis=0), yerr=yerr.values,
                    fmt='o', elinewidth=1, capsize=2, capthick=1, alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
        if plot_confidence_intervals:
            ax.fill_between(self.more_concentrations,
                            self._bindingcurve(self.more_concentrations, self.affinity_params.loc['dG_lb']),
                            self._bindingcurve(self.more_concentrations, self.affinity_params.loc['dG_ub']),
                            edgecolor='w', facecolor=sns.xkcd_rgb['light grey'], alpha=0.5)
        ax.plot(self.more_concentrations,
                self._bindingcurve(self.more_concentrations, self.affinity_params.loc['dG']),
                color=sns.xkcd_rgb['vermillion'])
        
        ax = self._formataxis(ax)
        #sns.despine(trim=True)
        plt.tight_layout()
        if annotate:
            ax.annotate('dG (kcal/mol):\n%4.2f (%4.2f, %4.2f)'%(self.affinity_params.loc['dG'], self.affinity_params.loc['dG_lb'], self.affinity_params.loc['dG_ub']),
                        xy=(.9, 0.05),  xycoords='axes fraction',
                        horizontalalignment='right', verticalalignment='bottom')
        return ax
    
    def plotMedianNormalizedClustersOffrates(self, plot_confidence_intervals=None, annotate=None):
        if plot_confidence_intervals is None:
            plot_confidence_intervals = True
        if annotate is None:
            annotate = True

        ybounds = self.findNormalizedErrorOffrates()
        
        fig = plt.figure(figsize=(3.75, 3.25))
        ax = fig.add_subplot(111)
        xvalues = ybounds.index.tolist()
        yvalues = ybounds.loc[:, 'fnorm']   
        yerr = np.array([(ybounds.loc[:, 'fnorm'] - ybounds.loc[:, 'fnorm_lb']).values, (ybounds.loc[:, 'fnorm_ub'] - ybounds.loc[:, 'fnorm']).values], dtype=float)
        ax.errorbar(xvalues, yvalues, yerr=yerr,
                    fmt='o', elinewidth=1, capsize=2, capthick=1, alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
        if plot_confidence_intervals:
            ax.fill_between(self.more_times,
                            self._offratecurve(self.more_times, self.offrate_params.loc['toff_lb']),
                            self._offratecurve(self.more_times, self.offrate_params.loc['toff_ub']),
                            edgecolor='w', facecolor=sns.xkcd_rgb['light grey'], alpha=0.5)
        ax.plot(self.more_times,
                self._offratecurve(self.more_times, self.offrate_params.loc['toff']),
                color=sns.xkcd_rgb['vermillion'])
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, self.more_times[-1])
        ax.set_xlabel('time (s)')
        ax.set_ylabel('fraction bound')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        #sns.despine(trim=True)
        plt.tight_layout()
        if annotate:
            ax.annotate('half life (s):\n%4.0f (%4.0f, %4.0f)'%(self.offrate_params.loc['toff'], self.offrate_params.loc['toff_lb'], self.offrate_params.loc['toff_ub']),
                        xy=(.9, 0.9),  xycoords='axes fraction',
                        horizontalalignment='right', verticalalignment='top')
        return ax

    def findOnRate(self):
        return 1/(np.exp(self.affinity_params.loc['dG']/self.RT)*self.offrate_params.loc['toff']) # per molar per second
    
    def findOnRateError(self, delta_toff, delta_dG):
        x = self.offrate_params.loc['toff']
        y = self.affinity_params.loc['dG']
        RT = self.RT
        return np.sqrt( np.power(-np.exp(-y/RT)/(x**2)*delta_toff, 2) +
                        np.power(-np.exp(-y/RT)/(RT*x)*delta_dG, 2))
        
    def deltaGDoubleDagger(self, koff_f, koff_0):
        return -self.RT*np.log(koff_f/float(koff_0))

    def deltaGDoubleDaggerError(self, toff_f, toff_0, delta_toff_f, delta_toff_0):
        return np.sqrt(np.power(delta_toff_0/toff_0, 2) + np.power(delta_toff_f/toff_f, 2) )

    def findDeltaGDoubleDagger(self, table, variant_table, table_offrates, variant_table_offrates, variant=None):
        # assume reference is variant 0
        if variant is None: variant = 0
        b = loadTectoData(table, variant_table, variant, table_offrates, variant_table_offrates, self.times )
        ddG = pd.DataFrame(index = ['affinity', 'offrate', 'onrate'], columns=['dG', 'eminus', 'eplus'])
        ddG.loc['affinity', 'dG'] = self.affinity_params.loc['dG'] - b.affinity_params.loc['dG']
        for e in ['eminus', 'eplus']:
            ddG.loc['affinity', e] = np.sqrt(np.power(self.affinity_params.loc[e], 2) + np.power(b.affinity_params.loc[e], 2))
        
        # delta g double dagger from off rate
        ddG.loc['offrate', 'dG'] = self.deltaGDoubleDagger(1/self.offrate_params.loc['toff'], 1/b.offrate_params.loc['toff'])
        for e in ['eminus', 'eplus']:
            ddG.loc['offrate', e] = self.deltaGDoubleDaggerError(self.offrate_params.loc['toff'], b.offrate_params.loc['toff'],
                                                                 self.offrate_params.loc[e], b.offrate_params.loc[e])

        # delta G double dagger from on rate
        ddG.loc['onrate', 'dG'] = self.deltaGDoubleDagger(self.onrate_params.loc['kon'], b.onrate_params.loc['kon'])
        for e in ['eminus', 'eplus']:
            ddG.loc['onrate', e] = self.deltaGDoubleDaggerError(self.onrate_params.loc['kon'], b.onrate_params.loc['kon'],
                                                                 self.onrate_params.loc[e], b.onrate_params.loc[e])        
        
        return ddG 
    
    def plotEnergyDiagram(self, ddG):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xvalues = [0,1, 2,3]
        
        initial_energy = 0
        energy_barrier_forward = 1
        energy_barrier_backward = 3
        final_energy = -2
        transition_energy1 = initial_energy + energy_barrier_forward
        transition_energy2 = final_energy + energy_barrier_backward

        
        ref = [initial_energy, transition_energy1, transition_energy2, final_energy]
        
        final_energy_new = final_energy + ddG.loc['affinity', 'dG']
        energy_barrier_backward_new = energy_barrier + ddG.loc['offrate', 'dG'] 
        transition_energy2_new = final_energy_new + energy_barrier_backward_new
        
        energy_barrier_forward_new = energy_barrier_forward + ddG.loc['onrate', 'dG']
        transition_energy1_new = initial_energy + energy_barrier_forward_new
        
        test = [initial_energy, transition_energy1_new, transition_energy2_new, final_energy_new]
        
        
        ax.plot(xvalues, ref, marker='o')
        ax.errorbar(xvalues, test, [[0, ddG.loc['onrate', e], ddG.loc['offrate', e], ddG.loc['affinity', e]] for e in ['eminus', 'eplus']], fmt = 's-')        
    
    def plotClusters(self, indices=None):
        if indices is None: indices = self.normalized_binding_curves.index
        fig = plt.figure(figsize=(3.75, 3.25))
        ax = fig.add_subplot(111)
        for index in indices:
            ax.plot(self.concentrations, self.binding_curves.loc[index], 'o', alpha=0.5, color=sns.xkcd_rgb['charcoal'] )
            ax.plot(self.more_concentrations,
                    self._bindingcurve(self.more_concentrations, self.affinity_params_cluster.loc[index, 'dG'],
                                                                 fmin=self.affinity_params_cluster.loc[index, 'fmin'],
                                                                 fmax=self.affinity_params_cluster.loc[index, 'fmax']),
                    '-', alpha=0.5, color=sns.xkcd_rgb['vermillion'])
        ax.set_xscale('log')
        ax.set_xlabel('concentration (nM)')
        ax.set_ylabel('absolute fluorescence (a.u.)')
        ax.set_xlim(0.5, 2500)
        plt.tight_layout()
        return ax
    
    def reinterpretSequence(self):
        regions = pd.Series(index=np.arange(len(self.seq_params.loc['sequence'])), dtype=str)
        regions.iloc[:6] = 'base'
        regions.iloc[-6:] = 'base'
        
        # assuming R1
        regions.iloc[6:11] = 'receptor'
        regions.iloc[-12:-6] = 'receptor'
        
        # helix
        start_ind = 11
        end_ind = -12
        helix = self.seq_params.helix_sequence.split('_')
        regions.iloc[start_ind:start_ind+len(helix[0])] = 'helix'
        regions.iloc[end_ind-len(helix[0]):end_ind] = 'helix'
        
        # junction
        start_ind = start_ind+len(helix[0])
        end_ind = end_ind-len(helix[0])
        junction = self.seq_params.junction_sequence.split('_')
        regions.iloc[start_ind:(start_ind +len(junction[0]))] = 'junction'
        regions.iloc[end_ind-len(junction[1]):end_ind] = 'junction'
        
        # helix again
        start_ind = start_ind+len(junction[0])
        end_ind = end_ind - len(junction[1])
        regions.iloc[start_ind:start_ind+len(helix[1])] = 'helix'
        regions.iloc[end_ind-len(helix[1]):end_ind] = 'helix'
        
        # loop 
        start_ind = start_ind+len(helix[1])
        end_ind = end_ind - len(helix[1])     
        regions.iloc[start_ind:end_ind] = 'loop'
        return regions
    
    def getSecondaryStructure(self, name=None):
        tectoSeq = hjh.tecto_assemble.TectoSeq()
        regions = self.reinterpretSequence()
        cmd = tectoSeq.printVarnaCommand(tectoSeq=self.seq_params.sequence, indices=((regions=='helix')|(regions=='junction')).values, name=name, use_colormap=False)
        return cmd

    def getSecondaryStructureAll(self, name=None):
        tectoSeq = hjh.tecto_assemble.TectoSeq()
        cmd = tectoSeq.printVarnaCommand(tectoSeq=self.seq_params.sequence, name=name, use_colormap=False)
        return cmd
    
    def findAssociatedVariant(self, charDict, variant_table):
        b = self.seq_params.loc['topology':'total_length'].drop('helix_sequence')
        for key, value in charDict.items():
            b.loc[key] = value
        
        return variant_table.loc[(variant_table.loc[:, b.index] == b).all(axis=1)].index
    
    def checkHelixBasePaired(self, regions=None):
        if regions is None:
            regions = self.reinterpretSequence()
        helixSS = ''.join(np.array(list(self.seq_params.ss))[(regions=='helix').values])
        helixLength = int(self.seq_params.total_length - self.seq_params.junction_length)
        
        if helixSS == ''.join(['(']*helixLength + [')']*helixLength ):
            return True
        else:
            return False

    def checkBaseBasePaired(self, regions=None):
        if regions is None:
            regions = self.reinterpretSequence()
        baseSS = ''.join(np.array(list(self.seq_params.ss))[(regions=='base').values])
        helixLength = 6
        
        if baseSS == ''.join(['(']*helixLength + [')']*helixLength ):
            return True
        else:
            return False

    def checkJunctionUnPaired(self, regions=None):
        if regions is None:
            regions = self.reinterpretSequence()
            
        junctionSS = ''.join(np.array(list(self.seq_params.ss))[(regions=='junction').values])
        junctionLength = int(self.seq_params.junction_length)
        
        if junctionSS == ''.join(['.']*2*junctionLength):
            return True
        else:
            return False
    
    def checkSecondaryStructure(self, regions=None):
        if regions is None:
            regions = self.reinterpretSequence()
        success = True
        success = success and self.checkHelixBasePaired(regions=regions)
        success = success and self.checkBaseBasePaired(regions=regions)
        return success
        
    def _bindingcurve(self, concentrations, dG, fmax=None, fmin=None):
        if fmax is None:
            fmax = 1
        if fmin is None:
            fmin = 0
        parameters = Parameters()
        return fmax*concentrations/(concentrations + np.exp(dG/parameters.RT)/1E-9) + fmin
    
    def _formataxis(self,ax):
        ax.set_xscale('log')
        ax.set_xlabel('concentration (nM)')
        ax.set_ylabel('fraction bound')
        ax.set_xlim(0.5, 3000)
        ax.set_ylim(0, 1.1)
        return ax
    
    def _offratecurve(self, times, toff, fmax=None, fmin=None):
        if fmax is None:
            fmax = 1
        if fmin is None:
            fmin = 0
        parameters = Parameters()
        return fmax*np.exp(-(times/toff).astype(float)) + fmin


def loadTectoData(table, variant_table, variant, table_offrates=None, variant_table_offrates=None, times=None):
    
    if variant_table_offrates is not None and  table_offrates is not None and times is not None:
        affinityData = AffinityData(table.loc[table.loc[:, 'variant_number']==variant],
                                variant_table.loc[variant],
                                table_offrates.loc[table_offrates.loc[:, 'variant_number']==variant],
                                variant_table_offrates.loc[variant],
                                times)
    else:
        affinityData = AffinityData(table.loc[table.loc[:, 'variant_number']==variant],
                                    variant_table.loc[variant])
    return affinityData

def getSecondaryStructure(variant_table):
    variant_table.loc[:, 'ss'] = hjh.tecto_assemble.getAllSecondaryStructures(variant_table.loc[:, 'sequence'])
    return variant_table

def getVariantsAssociatedByLength(table, variant_table, seqs=None, motif=None, offset=None, helix_context=None, return_all=None):
    if helix_context is None: helix_context = 'rigid'
    if offset is None: offset = 0
    if return_all is None: return_all = False
    if seqs is None:
        if motif is None:
            print "need to define either list of junction sequences (i.e. 'A_') or motif (i.e. 'B1')"
            return
        else:
            if '_' in motif: motif = motif.split('_')
            if ',' in motif: motif = motif.split(',')
            else: motif = list(motif)
            seqs = ['_'.join(x) for x in Junction(tuple(motif)).sequences.values]
    maxoffset = np.abs(offset)+2
    variantDict = pd.DataFrame(index=seqs, columns=range(8, 13))
    for seq in seqs:
        print seq
        variants = variantFun.findVariantNumbers(table, {'helix_context':helix_context, 'loop':'goodLoop',
                                                                 'receptor':'R1','junction_sequence':seq,'offset':maxoffset})
        subtable = table.loc[np.in1d(table.loc[:, 'variant_number'], variants)]
        #subtable = pd.concat([subtable, subtable.affinity], axis=1)
        subtable.loc[:, 'offset'] = subtable.loc[:, 'helix_two_length'] - subtable.loc[:, 'helix_one_length']
        if seq != '_':
            pass
            subtable = subtable.loc[(subtable.loc[:, 'offset'] == offset)|(subtable.loc[:, 'offset']==offset-1)]
        variants =  np.unique(subtable.loc[:, 'variant_number']).astype(int)
        print seq, variants
        variantDict.loc[seq, variant_table.loc[variants, 'total_length']] = variants

    return variantDict

def getVariantsAssociatedByPosition(table, variant_table, seqs=None, motif=None, offset=None, helix_context=None):
    if helix_context is None: helix_context = 'rigid'
    if offset is None: offset = 0
    if return_all is None: return_all = False
    if seqs is None:
        if motif is None:
            print "need to define either list of junction sequences (i.e. 'A_') or motif (i.e. 'B1')"
            return
        else:
            if '_' in motif: motif = motif.split('_')
            if ',' in motif: motif = motif.split(',')
            else: motif = list(motif)
            seqs = ['_'.join(x) for x in Junction(tuple(motif)).sequences.values]
    for seq in seqs:
        variants = variantFun.findVariantNumbers(table, {'helix_context':helix_context, 'loop':'goodLoop',
                                                                 'receptor':'R1','junction_sequence':seq, 'total_length':10})
            
    return

def plotPositionChangesHistogram(table, variant_table, motifs, helix_context=None):
    if helix_context is None: helix_context = 'rigid'
    total_length = 10

    maxoffset = 2
    variants = []
    bulgetype = []
    for motif in motifs:
        v = variantFun.findVariantNumbers(table, {'helix_context':helix_context, 'loop':'goodLoop', 'topology':motif,
                                                                 'receptor':'R1', 'total_length':total_length, 'offset':maxoffset})
        variants.append(v)
        if motif in ['B1', 'B1_B1', 'B1_B1_B1', 'M_B1', 'M_M_B1', 'M_B1_B1']:
            bulgetype.append(['B1']*len(v))
        elif motif in ['B2', 'B2_B2', 'B2_B2_B2', 'B2_M', 'B2_M_M', 'B2_B2_M']:
            bulgetype.append(['B2']*len(v))
        else:
            bulgetype.append(['M']*len(v))
    # simple secondary structure testing
    variant_subtable = variant_table.loc[np.hstack(variants)]
    variant_subtable.loc[:, 'bulgetype'] =  np.hstack(bulgetype)
    variant_subtable.loc[:, 'ss_correct'] = [s.find('(....)') > -1 for s in variant_subtable.loc[:, 'ss']]
    
    variant_subtable = variant_subtable.loc[variant_subtable.loc[:, 'ss_correct']&((variant_subtable.loc[:, 'helix_one_length']==4)|(variant_subtable.loc[:, 'helix_one_length']==5))]
    cols = ['dG', 'helix_one_length']
    variant_subtable.loc[:, cols] = variant_subtable.loc[:, cols].astype(float)
    bins=np.linspace(-12, -6, 30)
    pal = {5:sns.xkcd_rgb["vermillion"], 4:sns.xkcd_rgb["light grey"]}
    
    g = sns.FacetGrid(variant_subtable, row="topology", hue='helix_one_length', size=2, aspect=2, palette=pal)
    g.map(plt.hist, "dG", bins=bins,histtype="stepfilled", alpha=0.75);
    g.set_axis_labels("dG (kcal/mol)", "number of variants");
    g.set(xlim=(-12, -6) )
    #g.fig.subplots_adjust(left=0.15, wspace=.05, right=0.5);
    g.add_legend();
    
    return 


def plotAllRigid(variant_table):
    variant_table.loc[variant_table.loc[:, 'topology'].astype(str) == "nan", 'topology'] = ''
    helix_context = 'rigid'
    variants = variantFun.findVariantNumbers(table, {'helix_context':helix_context, 'loop':'goodLoop',
                                                             'receptor':'R1','offset':1})
    variant_subtable = variant_table.loc[variants]

    index = ((variant_subtable.loc[:, 'topology'] == '')|
             (variant_subtable.loc[:, 'topology'] == 'M')|
             (variant_subtable.loc[:, 'topology'] == 'M_M'))

    fig = plt.figure(figsize=(3.5, 3.25))
    ax = fig.add_subplot(111)
    ax = sns.violinplot( x='total_length', y='dG', data=variant_subtable.loc[index],order=[8,9,10,11,12], hue='topology', palette='Set2')
    ax = sns.stripplot(x="total_length", y="dG", data=variant_subtable.loc[index],order=[8,9,10,11,12], hue='topology',palette='Set2',
                        size=3, jitter=0.25, edgecolor="gray", alpha=0.25, marker=".")
    
    ax.set_ylim(-12, -5)
    #ax.set_xlim(7.5, 12.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figDirectory, 'junciton_M_MM.bylength.pdf'))
    
def plotBadLoop(variant_table):
    variant_table.loc[variant_table.loc[:, 'topology'].astype(str) == "nan", 'topology'] = ''
    helix_context = 'rigid'
    variants_good = variantFun.findVariantNumbers(table, {'helix_context':helix_context, 'loop':'goodLoop',
                                                             'receptor':'R1','offset':1})
    variants_bad = variantFun.findVariantNumbers(table, {'helix_context':helix_context, 'loop':'badLoop',
                                                             'receptor':'R1','offset':1})
    
    variant_subtable = variant_table.loc[np.append(variants_good,variants_bad) ]
    variant_subtable.loc[:, 'cutoff'] = variant_subtable.loc[:, 'qvalue']<0.075
    index_to_change = variant_subtable.loc[:, 'dG'] > -6
    variant_subtable.loc[index_to_change, 'dG'] = -6
    variant_subtable.loc[index_to_change, 'dG_lb'] = -6
    bins=np.linspace(-12, -5.5, 100)
    pal = {1:sns.xkcd_rgb["vermillion"], 0:sns.xkcd_rgb["light grey"]}
    g = sns.FacetGrid(variant_subtable, row="loop", hue='cutoff', size=2, aspect=2, palette=pal)
    g.map(plt.hist, "dG", bins=bins,histtype="stepfilled", alpha=0.75);
    g.set_axis_labels("dG (kcal/mol)", "number of variants");
    g.set(yticks=np.arange(0, 400,100), xlim=(-12, -5.5) )
    g.fig.subplots_adjust(left=0.15, wspace=.05, right=0.5);
    g.add_legend();
    
    
    #g.map(sns.distplot, "dG", kde=False, norm_hist=False);
    # plot
    xlim =  [-11.5, -4]
    ylim = [-11.5, -4]
    
    # plot histograms
    sns.distplot(data);

    fig = plt.figure(figsize=(3.5,3.25))
    gs = gridspec.GridSpec(1, 2,
                       width_ratios=[1,3],
                       wspace=0.025)
    ax1 = fig.add_subplot(111, aspect='equal')
    im = ax1.hexbin(data.loc[:, ('good', 'dG')], data.loc[:, ('bad', 'dG')], cmap='Greys', 
                    extent=[xlim[0], xlim[1], ylim[0], ylim[1]], gridsize=50)
    ax1.set_xlabel('GGAA loop dG (kcal/mol)')
    ax1.set_ylabel('GAAA loop dG (kcal/mol)')
    plt.tight_layout()
    plt.colorbar(im)
    
    xlim =  [-12, -5]
    ylim = [-12, -5]
    index = data.loc[:, ('good', 'dG')] < -8
    color = 'vermillion'
    name1 = ('good', 'dG'); name2 = ('bad', 'dG')
    g = sns.JointGrid(data.loc[:, name1], data.loc[:, name2], size=3.75, ratio=7, space=0, dropna=True, xlim=xlim, ylim=xlim)
    g.plot_marginals(sns.kdeplot, color=sns.xkcd_rgb[color], shade=True)
    #g.plot_joint(plt.hexbin, cmap='Greys', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], gridsize=40)
    #g.plot_joint(sns.kdeplot, cmap='Greys', shade=True, gridsize=50, clip=[(xlim[0], xlim[1]), (ylim[0], ylim[1])])
    g.plot_joint(plt.scatter, alpha=0.5, edgecolors=sns.xkcd_rgb['charcoal'], facecolors='none')
    g.set_axis_labels('GGAA loop dG (kcal/mol)', 'GAAA loop dG (kcal/mol)')
    #.annotate(st.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");
    
    
def findOtherVariants(variant_table, table, variants):
    iterables = [['good', 'bad'], ['dG', 'eminus', 'eplus', 'variant']]
    index = pd.MultiIndex.from_product(iterables, names=['loop', 'value'])
    data = pd.DataFrame(index=variants, columns=index)
    
    for variant in variants:
        a = tectoData.loadTectoData(table, variant_table, variant)
        data.loc[variant, 'bad'] = np.append(a.affinity_params.loc[['dG', 'eminus', 'eplus']].values, variant).astype(float)
        new_variants = a.findAssociatedVariant({'loop':'goodLoop'}, variant_table)
        if len(new_variants) == 1:
            b = tectoData.loadTectoData(table, variant_table, new_variants[0])
            data.loc[variant, 'good'] = np.append(b.affinity_params.loc[['dG', 'eminus', 'eplus']].values,  new_variants[0]).astype(float)     
    return data

def plotMultipleBindingCurves(table, variant_table, variants_good, variants_bad):

    fig = plt.figure(figsize=(3.75, 3.25))
    ax = fig.add_subplot(111)
    
    for variant in variants_good:

        a = tectoData.loadTectoData(table, variant_table, variant)
        #normValues.loc[variant] = a.normalized_binding_curves.median(axis=0)
        #ax.errorbar(a.concentrations, a.normalized_binding_curves.median(axis=0), yerr=a.findNormalizedError().transpose().astype(float).values,
        #            fmt='o', elinewidth=1, capsize=2, capthick=1, alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
        ax.scatter(a.concentrations, a.normalized_binding_curves.median(axis=0),  edgecolors=sns.xkcd_rgb['vermillion'], facecolors='none')
        ax.plot(a.more_concentrations, a._bindingcurve(a.more_concentrations, a.affinity_params.dG), color=sns.xkcd_rgb['vermillion'])
        ax.fill_between(a.more_concentrations,
                        a._bindingcurve(a.more_concentrations, a.affinity_params.loc['dG_lb']),
                        a._bindingcurve(a.more_concentrations, a.affinity_params.loc['dG_ub']),
                        edgecolor='w', facecolor=sns.xkcd_rgb['vermillion'], alpha=0.1)
        
    for variant in variants_bad:

        a = tectoData.loadTectoData(table, variant_table, variant)
        #normValues.loc[variant] = a.normalized_binding_curves.median(axis=0)
        #ax.errorbar(a.concentrations, a.normalized_binding_curves.median(axis=0), yerr=a.findNormalizedError().transpose().astype(float).values,
        #            fmt='o', elinewidth=1, capsize=2, capthick=1, alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
        ax.scatter(a.concentrations, a.normalized_binding_curves.median(axis=0),  edgecolors=sns.xkcd_rgb['cerulean'], facecolors='none')
        ax.plot(a.more_concentrations, a._bindingcurve(a.more_concentrations, a.affinity_params.dG), color=sns.xkcd_rgb['cerulean'])
        ax.fill_between(a.more_concentrations,
                        a._bindingcurve(a.more_concentrations, a.affinity_params.loc['dG_lb']),
                        a._bindingcurve(a.more_concentrations, a.affinity_params.loc['dG_ub']),
                        edgecolor='w', facecolor=sns.xkcd_rgb['cerulean'], alpha=0.1)
        
    ax.set_xscale('log')
    ax.set_xlabel('concentration (nM)')
    ax.set_ylabel('fraction bound')
    ax.set_xlim(0.5, 3000)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

def getAllVariantNumbers(table, variant_table, motifs, helix_context=None):

    variantDict = {}
    for motif in motifs:
        index = (variant_table.loc[:, 'total_length'] == 8)&(variant_table.loc[:, 'topology'] == motif)
        seqs = np.unique(variant_table.loc[index, 'junction_sequence'])
        variantDict[motif] = getVariantsAssociatedByLength(table.loc[table.loc[:, 'topology']==motif],
                                                                        variant_table,
                                                                        seqs=np.unique(variant_table.loc[index, 'junction_sequence']),
                                                                        offset=0,
                                                                        helix_context=helix_context)
    #variantDict['_'] = pd.DataFrame( index=['_'], columns=range(8, 13))
    #variantDict['_'].loc['_'] = [21736,21739,0,21744,21747]
    variantDict = pd.concat(variantDict)
    return pd.DataFrame(index=variantDict.index, data=variantDict.values, columns=['8', '9', '10', '11', '12'])



def parseVariantDeltaGsByLength(table, variant_table, motif,variantDict ):
    iterables = [['dG', 'eminus', 'eplus', 'bp_correct', 'junction_bp'], np.arange(8, 13).astype(str)]
    yvaluesAll = pd.DataFrame(index=variantDict.loc[motif].index, columns=pd.MultiIndex.from_product(iterables, names=['param', 'length']))
    for seq in yvaluesAll.index.tolist():
        for variant in variantDict.loc[(motif, seq)]:
            if not np.isnan(variant):
                a = loadTectoData(table, variant_table, variant)
                length = '%d'%a.seq_params.total_length
                for param in ['dG', 'eminus', 'eplus']:
                    yvaluesAll.loc[seq, (param, length)] = a.affinity_params.loc[param]
                regions = a.reinterpretSequence()
                yvaluesAll.loc[seq, ('bp_correct', length)] = a.checkSecondaryStructure(regions=regions)
                yvaluesAll.loc[seq, ('junction_bp', length)] = a.checkJunctionUnPaired(regions=regions)
    return yvaluesAll

def plotOverLength(yvaluesAll, errorbar=None, plotlines=None, how_to_index=None):
    # plot
    if errorbar is None: errorbar = False
    if plotlines is None: plotlines = False
    if how_to_index is None or how_to_index=='all':
        index = yvaluesAll.loc[:, 'bp_correct'].all(axis=1)
    elif how_to_index == 'any':
        index = yvaluesAll.loc[:, 'bp_correct'].any(axis=1)
    else:
        index = yvaluesAll.loc[:, 'bp_correct'].sum(axis=1) >= how_to_index
        
    fig = plt.figure(figsize=(3.25, 3.25)); ax = fig.add_subplot(111)
    jitter = np.linspace(-0.1, 0.1,index.sum()); count=0
    
    for seq in yvaluesAll.loc[index].index:
        lengths = yvaluesAll.loc[seq, 'dG'].dropna().index.tolist()
        yvalues = yvaluesAll.loc[seq, 'dG'].dropna()
        yerr = [yvaluesAll.loc[seq, e].dropna() for e in ['eminus' , 'eplus']]
        index = yvaluesAll.loc[seq, 'bp_correct'].loc[lengths].astype(bool)
        if errorbar:
            if plotlines: fmt='o-'
            else: fmt = 'o'
            ax.errorbar(lengths+jitter[count], yvalues, yerr=yerr,
                        fmt=fmt, elinewidth=1, capsize=1, capthick=1, alpha=0.5, marker='s', markersize=4, color=sns.xkcd_rgb['charcoal'], linewidth=1)
        else:
            ax.scatter(np.array(index.loc[index].index, dtype=float)+jitter[count], yvalues.loc[index], s=16, marker='o',
                        edgecolors=sns.xkcd_rgb['charcoal'], alpha=0.75, facecolors='none')
            if plotlines:
                ax.plot(np.array(index.loc[index].index, dtype=float)+jitter[count], yvalues.loc[index], linewidth=1,
                            edgecolors=sns.xkcd_rgb['charcoal'], alpha=0.1)
        count+=1
    lengths = [8,9,10,11,12]
    plt.xlim(7.5, 12.5)
    plt.ylim(-12, -6)
    plt.xticks(lengths, ['%d'%i for i in lengths])
    plt.xlabel('lengths (bp)')
    plt.ylabel('dG (kcal/mol)')
    plt.tight_layout()
    return ax

def getSecondaryStructureDiagrams(table, variant_table, variants, figDirectory):
    for variant in variants:
        a = loadTectoData(table, variant_table, variant)
        cmd = a.getSecondaryStructure(name=os.path.join(figDirectory, 'variant_%d.ss.eps'%variant))
        os.system(cmd)
    return cmd

def makeOffRateBarGraph(table, variant_table, variants, table_kinetics, variant_table_kinetics, times):
    lengths = np.array([8,9,10,11,12])
    yvalues = pd.DataFrame(index=lengths, columns=['eminus', 'koff', 'eplus', 'ss', 'fold'])
    for variant in variants:
        a = loadTectoData(table, variant_table, variant, table_kinetics, variant_table_kinetics, times)
        length = a.seq_params.total_length
        yvalues.loc[length, 'koff'] = 1/(a.offrate_params.loc['toff'].astype(float))
        for e in ['eminus', 'eplus']: yvalues.loc[length, e] = (a.offrate_params.loc[e]/a.offrate_params.loc['toff']**2)
        yvalues.loc[length, 'fold'] = a.offrate_curves.loc[:, ['0']].mean().values[0]/a.offrate_curves.loc[:, ['39']].mean().values[0]
        yvalues.loc[length, ['ss']] = a.checkSecondaryStructure()
    fig = plt.figure(figsize=(3.25, 3.25))
    ax = fig.add_subplot(111)
    ax.set_xlim(7.5, 12.5)
    ax.set_ylim(0, 1.5E-3)
    ax.set_ylabel('k off (per s)')
    ax.set_xlabel('lengths (bp)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    index = (yvalues.loc[lengths, 'fold'] >= 10).values.astype(bool)
    ax.bar(lengths[index]-0.4, yvalues.loc[lengths[index], 'koff'], 0.8, facecolor='r',
           yerr=[yvalues.loc[lengths[index], 'eminus'], yvalues.loc[lengths[index], 'eplus']],
           ecolor=sns.xkcd_rgb['charcoal'], linewidth=1)
    
    index = ((yvalues.loc[lengths, 'fold'] >= 6)& (yvalues.loc[lengths, 'fold'] < 10)).values.astype(bool)
    ax.bar(lengths[index]-0.4, yvalues.loc[lengths[index], 'koff'], 0.8, facecolor='0.5',
           yerr=[yvalues.loc[lengths[index], 'eminus'], yvalues.loc[lengths[index], 'eplus']],
           ecolor=sns.xkcd_rgb['charcoal'], linewidth=1)

    index = ((yvalues.loc[lengths, 'fold'] < 6)).values.astype(bool)
    ax.bar(lengths[index]-0.4, yvalues.loc[lengths[index], 'koff'], 0.8, facecolor='0.75',
           yerr=[yvalues.loc[lengths[index], 'eminus'], yvalues.loc[lengths[index], 'eplus']],
           ecolor=sns.xkcd_rgb['charcoal'], linewidth=1)
    return ax
    
    
        
def makePerSeqLengthDiagram(table, variant_table, variants, figDirectory=None, colorBySs=None, colorByBinding=None):
    if colorBySs is None:
        colorBySs = False
    if colorByBinding:
        colorByBinding = False
    lengths = np.arange(8, 13)

    yvalues = pd.DataFrame(index=lengths, columns=['eminus', 'dG', 'eplus', 'ss', 'q'])
    for variant in variants:
        
        a = loadTectoData(table, variant_table, variant)
        length = a.seq_params.total_length
        yvalues.loc[length, 'dG'] = (a.affinity_params.loc['dG'].astype(float))
        yvalues.loc[length, ['eminus', 'eplus']] = (a.affinity_params.loc[['eminus', 'eplus']].astype(float).values)
        if colorBySs:
            yvalues.loc[length, ['ss']] = a.checkSecondaryStructure()
            yvalues.loc[length, ['q']] = a.affinity_params.loc['qvalue']
    yvalues.dropna(axis=0, how='all', inplace=True)
    lengths = np.array(yvalues.index)
    plt.figure(figsize=(3.25, 3.25))
    if not colorBySs:
        plt.errorbar(lengths, yvalues.loc[lengths, 'dG'], yerr=yvalues.loc[lengths, ['eminus', 'eplus']].transpose().values,
                     fmt='o-', elinewidth=1, capsize=2, capthick=1, alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
    else:
        plt.errorbar(lengths, yvalues.loc[lengths, 'dG'], yerr=yvalues.loc[lengths, ['eminus', 'eplus']].transpose().values,
                     fmt='-', elinewidth=1, capsize=2, capthick=1, alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
        index = (yvalues.loc[:, 'ss'] &  (yvalues.loc[:, 'q'] < 0.05)).values.astype(bool)
        plt.plot(lengths[index], yvalues.loc[lengths[index], 'dG'], 'o', alpha=0.5, color=sns.xkcd_rgb['charcoal'], linewidth=1)
        
        index = (yvalues.loc[:, 'ss'] &  (yvalues.loc[:, 'q'] > 0.05)).values.astype(bool)
        plt.scatter(lengths[index], yvalues.loc[lengths[index], 'dG'], s=36, alpha=0.5, edgecolors=sns.xkcd_rgb['charcoal'], linewidth=1, facecolors='none')
        
        index = (~yvalues.loc[:, 'ss'] &  (yvalues.loc[:, 'q'] > 0.05)).values.astype(bool)
        plt.scatter(lengths[index], yvalues.loc[lengths[index], 'dG'], s=36, alpha=0.5, edgecolors=sns.xkcd_rgb['red'], linewidth=1, facecolors='none')
        
        index = (~yvalues.loc[:, 'ss'] &  (yvalues.loc[:, 'q'] < 0.05)).values.astype(bool)
        plt.scatter(lengths[index], yvalues.loc[lengths[index], 'dG'], s=36, alpha=0.5, edgecolors=sns.xkcd_rgb['red'], linewidth=1, facecolors=sns.xkcd_rgb['red'])
                
        
    plt.xlim(7.5, 12.5)
    plt.ylim(-12, -6)
    plt.xticks(lengths, ['%d'%i for i in lengths])
    plt.xlabel('lengths (bp)')
    plt.ylabel('dG (kcal/mol)')
    plt.tight_layout()
    if figDirectory is not None:
        plt.savefig(os.path.join(figDirectory, 'junction.%s.by_length.pdf'%variant_table.loc[variants[0], 'junction_sequence']))
    return