import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cmx
from matplotlib import gridspec
import seaborn as sns
from scikits.bootstrap import bootstrap
import os
import variantFun
import hjh.tecto_assemble
from  hjh.junction import Junction
import itertools
import statsmodels.api as sm
import scipy.stats as st
import seqfun
import IMlibs
import scipy as scp

class Parameters():
    def __init__(self, concentrations=None):
        # save the units of concentration given in the binding series
        self.RT = 0.582  # kcal/mol at 20 degrees celsius
        self.min_deltaG = -12
        self.max_deltaG = -3
        self.concentration_units = 1E-9
        if concentrations is None:
            self.concentrations = np.array([2000./np.power(3, i) for i in range(8)])[::-1]
        else:
            self.concentrations = concentrations
        self.lims = {'qvalue':(2e-4, 1e0),
                    'dG':(-12, -4),
                    'koff':(1e-5, 1E-2),
                    'kon' :(1e1, 1e6),
                    'kobs':(1e-5, 1e-2)}

class AffinityData():
    def __init__(self, sub_table_affinity, sub_variant_table_affinity, sub_table_offrates=None, sub_variant_table_offrates=None,times=None, concentrations=None): 
        # affinity params general
        if concentrations is None:
            concentrations = 2000./np.power(3, np.arange(8))[::-1]
        self.concentrations = concentrations
        self.numPoints = numPoints = len(self.concentrations)
        self.more_concentrations = np.logspace(-2, 4, 50)        
        self.RT = 0.582  # kcal/mol at 20 degrees celsius
        
        # affinity params specific
        index = IMlibs.filterStandardParameters(sub_table_affinity, self.concentrations).index
        
        self.binding_curves = sub_table_affinity.loc[index, IMlibs.formatConcentrations(concentrations)]
        self.normalized_binding_curves = self.binding_curves.copy()
        for i in IMlibs.formatConcentrations(concentrations):
            self.normalized_binding_curves.loc[:, i] = (sub_table_affinity.loc[index, i] - sub_table_affinity.loc[index, 'fmin'])/ sub_table_affinity.loc[index, 'fmax']
        
        self.clusters = sub_table_affinity.index
        self.fit_clusters = index

        self.affinity_params = sub_variant_table_affinity.loc['dG':]
        self.affinity_params.loc['eminus'] = sub_variant_table_affinity.loc['dG'] - sub_variant_table_affinity.loc['dG_lb']
        self.affinity_params.loc['eplus'] = sub_variant_table_affinity.loc['dG_ub'] - sub_variant_table_affinity.loc['dG']
        self.affinity_params_cluster = sub_table_affinity.loc[index, 'fmax':'fmin_stde']
        
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
        
        ybounds = pd.DataFrame(index=np.arange(self.numPoints).astype(str), columns=['fnorm_lb', 'fnorm_ub'])
        for i in IMlibs.formatConcentrations(self.concentrations):
            ybounds.loc[str(i)] = bootstrap.ci(self.normalized_binding_curves.loc[:, str(i)], statfunction=np.median)
        yerr = pd.DataFrame(index=np.arange(self.numPoints).astype(str), columns=['eminus', 'eplus'])
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
    
    def plotClustersAll(self, fmax_lb=None):
        max_num_plots = 16
        fig = plt.figure(figsize=(9, 6))
        gs = gridspec.GridSpec(4, 4)
        
        # sort dG and plot in this order
        vec = self.affinity_params_cluster.dG.copy()
        vec.sort()       
        if len(self.binding_curves) > max_num_plots:
            index = vec.iloc[np.linspace(0, len(vec)-1, max_num_plots).astype(int)].index
        else:
            index = vec.index
        
        for i, idx in enumerate(index):
            ax = fig.add_subplot(gs[i])
            self.plotCluster(idx, ax, fmax_lb=fmax_lb)

        plt.annotate('absolute fluorescence (a.u.)', xy=(0.2, 0.5),
                xycoords='figure fraction',
                rotation=90,
                horizontalalignment='right', verticalalignment='center',
                fontsize=12)
           
        plt.annotate('concentration (nM)', xy=(0.5, 0.025),
                xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='top',
                fontsize=12)

        plt.annotate(('# tests: %d\n'+
                      'Pass fit filter: %3.1f%%\n'+
                      'dG: %4.2f (%4.2f, %4.2f)')
                      %(self.affinity_params.numTests,
                        self.affinity_params.fitFraction*100,
                        self.affinity_params.dG, self.affinity_params.dG_lb, self.affinity_params.dG_ub),
                       xy=(.025, 0.975), xycoords='figure fraction',
                      horizontalalignment='left', verticalalignment='top',
                      fontsize=12)
        gs.update(wspace=0.5, hspace=0.5, left=0.25, bottom=0.1, top=0.975, right=0.975)
        pass
    
    def plotCluster(self, index, ax, fmax_lb=None ):
        ax.plot(self.concentrations, self.binding_curves.loc[index], 'o', alpha=0.5, color=sns.xkcd_rgb['charcoal'] )
        ax.plot(self.more_concentrations,
                self._bindingcurve(self.more_concentrations, self.affinity_params_cluster.loc[index, 'dG'],
                                                             fmin=self.affinity_params_cluster.loc[index, 'fmin'],
                                                             fmax=self.affinity_params_cluster.loc[index, 'fmax']),
                '-', alpha=0.5, color=sns.xkcd_rgb['vermillion'])
        if fmax_lb is not None:
            ax.plot([0.5, 2500], [fmax_lb, fmax_lb], 'k:', linewidth=1)
        ax.set_xscale('log')
        ax.set_xlim(0.5, 2500)
        pass
    
    
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


def loadTectoData(table, variant_table, variant, table_offrates=None, variant_table_offrates=None, times=None, concentrations=None):
    
    if variant_table_offrates is not None and  table_offrates is not None and times is not None:
        affinityData = AffinityData(table.loc[table.loc[:, 'variant_number']==variant],
                                variant_table.loc[variant],
                                table_offrates.loc[table_offrates.loc[:, 'variant_number']==variant],
                                variant_table_offrates.loc[variant],
                                times,
                                concentrations=concentrations)
    else:
        affinityData = AffinityData(table.loc[table.loc[:, 'variant_number']==variant],
                                    variant_table.loc[variant], concentrations=concentrations)
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

def filterVariants(subtable):
    numClustersFilter = (subtable.numTests >= 5)

    
    index = numClustersFilter
    
    return subtable.loc[index]


def plotSequenceJoe(variant_table, helices=None, mismatches=None):
    if helices is None:
        helices = True
    if mismatches is None:
        mismatches = False
    subtable = variant_table.copy()
    #index = (subtable.length == 10)&(subtable.numTests >= 5)&(subtable.dG_ub - subtable.dG_lb < 1)
    index = (subtable.length == 10)
    subtable = subtable.loc[index]
    #subtable.sort('dG', inplace=True)
    #plt.figure();
    #plt.scatter(np.arange(len(subtable)), subtable.dG, facecolors='none', edgecolors='k')
    
    if helices:
        predicted = pd.read_table('/home/sarah/JunctionLibrary/seq_params/exhustive_helices.results', sep=' ', header=None, usecols=[0, 4], names=['seq', 'ddG'])
        predicted.index = [s.replace('U', 'T') for s in predicted.seq]
    if mismatches:
        predicted = pd.read_table('/home/sarah/JunctionLibrary/seq_params/double_double_sequences.predictions.txt', index_col=0)
    
    
    subtable.index = subtable.sequence
    A = pd.concat([subtable.loc[predicted.index].dG, predicted.ddG], axis=1).dropna()
    x = A.dG
    y = A.ddG   
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, alpha=0.5, marker='.', s=20, c='k')
    
    # fit with OLS
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    
    xlim_max = np.array([-12, -4])
    #ax.plot(xlim_max,  results.params.dG*xlim_max + results.params.const, 'r:')
    
    
    # robust least squares
    model = sm.RLM(y, X )
    results = model.fit()

    xlim_max = np.array([-12, -4])
    ax.plot(xlim_max,  results.params.dG*xlim_max + results.params.const, 'r:', label='best fit')
       
    # line of slope 1
    origin = np.array([(A.dG - A.ddG).mean(), 0]) - np.array([2]*2)
    ax.plot([origin[0], origin[0]+5], [origin[1], origin[1]+5], '--', color=sns.xkcd_rgb['turquoise'], label='slope 1')
    
    ax.set_xlim(-12.5, -9.5)
    ax.set_ylim(-1.5, 2)
    ax.set_xlabel('$\Delta$G observed (kcal/mol)')
    ax.set_ylabel('$\Delta$$\Delta$G predicted (kcal/mol)')
    ax.tick_params(top='off', right='off')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # also plot histograms of error
    plt.figure(figsize=(3.5, 3.5))
    plt.hist(results.resid.values, bins=np.arange(-1.5, 1.5, .1), alpha=0.5, color='grey')
    plt.xlabel('residual ddG (kcal/mol)')
    plt.tight_layout()

    plt.figure(figsize=(3.5, 3.5))
    plt.hist((subtable.dG_ub - subtable.dG_lb).values, bins=np.arange(0, 3, 0.1), alpha=0.5, color='grey')  
    plt.xlabel('95% confidence (kcal/mol)')
    plt.tight_layout()

def plotVarianceDataset(table):
    bins = np.linspace(0, 2, 50)
    fig = plt.figure(figsize=(4.5,3.75))
    index = table.loc[:, 'barcode_good'].astype(bool)&(table.loc[:, 'qvalue']<=0.05)
    index2 = table.loc[:, 'barcode_good'].astype(bool)&(table.loc[:, 'qvalue']>0.05)
    plt.hist(table.loc[index, 'dG_var'].dropna().astype(float).values, normed=False, bins=bins, color=sns.xkcd_rgb['vermillion'], alpha=0.5, label='q<=0.05')
    plt.hist(table.loc[index2, 'dG_var'].dropna().astype(float).values, normed=False, bins=bins, color=sns.xkcd_rgb['light grey'], alpha=0.5, label='q>0.05')
    plt.xlabel('variance in fit dG, per cluster (kcal/mol)')
    plt.ylabel('number of clusters')
    plt.legend()
    plt.tight_layout()
    return

def plotConfIntervals(variant_table, min_num_tests=None):
    if min_num_tests is None: min_num_tests = 5
    
    variant_table.loc[:, 'conf_int_median'] = variant_table.loc[:, 'dG_ub'] - variant_table.loc[:, 'dG_lb']
    variant_table.loc[~np.isfinite(variant_table.loc[:, 'conf_int_median']), 'conf_int_median'] = np.nan

    fig = plt.figure(figsize=(4.5,3.75))
    bins = np.linspace(0, 4, 50)
    index = (variant_table.loc[:, 'numTests'] >= min_num_tests)&(variant_table.loc[:, 'qvalue'] < 0.05)
    plt.hist(variant_table.loc[index, 'conf_int_median'].dropna().astype(float).values,
             normed=False, bins=bins, color=sns.xkcd_rgb['vermillion'], alpha=0.5, label='q>=0.5')
    index = (variant_table.loc[:, 'numTests'] >= min_num_tests)&(variant_table.loc[:, 'qvalue'] > 0.05)
    plt.hist(variant_table.loc[index, 'conf_int_median'].dropna().astype(float).values,
             normed=False, bins=bins, color=sns.xkcd_rgb['light grey'], alpha=0.5, label='q>0.05')
    plt.xlabel('width of 95% confidence interval \nin the median fit dG (kcal/mol)')
    plt.ylabel('number of variants')
    plt.legend()
    plt.tight_layout()
    pass

def histogramSublibrary(variant_table, sublibrary):
    subtable = variant_table.loc[variant_table.sublibrary == sublibrary]
    min_num_tests = 5
    
    fig = plt.figure(figsize=(4.5,3.75))
    bins = np.arange(-12, -5, 0.2)
    index = (subtable.loc[:, 'numTests'] >= min_num_tests)&(subtable.loc[:, 'qvalue'] > 0.05)
    plt.hist(subtable.loc[index, 'dG'].dropna().astype(float).values,
             normed=False, bins=bins, color=sns.xkcd_rgb['light grey'], alpha=0.5, label='q>0.05')
    
    index = (subtable.loc[:, 'numTests'] >= min_num_tests)&(subtable.loc[:, 'qvalue'] < 0.05)
    plt.hist(subtable.loc[index, 'dG'].dropna().astype(float).values,
             normed=False, bins=bins, color=sns.xkcd_rgb['vermillion'], alpha=0.5, label='q>=0.5')
    plt.xlabel('dG (kcal/mol)')
    plt.ylabel('number of variants')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
def historgram_ddG_Tert(variant_table, tert1, tert2):
    subtable = variant_table.loc[variant_table.sublibrary == 'tertiary_contacts']
    subtable.loc[:, 'did_bind'] = subtable.qvalue < 0.05
    min_num_tests = 5
    index_t1 = (subtable.numTests >= 5)&(subtable.dG_ub - subtable.dG_lb < 1)&(subtable.qvalue< 0.05)&(subtable.receptor== tert1)
    index_t2 = (subtable.numTests >= 5)&(subtable.dG_ub - subtable.dG_lb < 1)&(subtable.qvalue< 0.05)&(subtable.receptor== tert2)
    table_t1 = subtable[index_t1]
    table_t2 = subtable[index_t2]
    index = np.arange(0,len(table_t1))
    table_t2_sorted = pd.DataFrame(index=np.arange(0,len(table_t1)), columns = table_t2.columns)
    
    for i, row in enumerate(table_t1.iterrows()):
        indx_t2_keep = (table_t2.junction_seq == row[1].junction_seq)&(table_t2.effective_length == row[1].effective_length)&(table_t2.helix_one_length == row[1].helix_one_length)
        if not table_t2[indx_t2_keep].empty:
            table_t2_sorted.iloc[i] = table_t2[indx_t2_keep].iloc[0]
    inds = (~pd.isnull(table_t2_sorted)).any(1).nonzero()[0]
    if len(inds) > 1:
        x = table_t1.iloc[inds]['dG'].values
        y = table_t2_sorted.iloc[inds]['dG'].values
        ddG = y-x
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        bins = np.arange(1, 3, 0.1)
        #ax.hist(ddG,bins=bins, alpha=0.5, histtype='stepfilled')
        ax.hist(ddG , bins=bins, color=sns.xkcd_rgb['vermillion'], histtype='stepfilled', alpha=0.5)
        plt.title(r'$\Delta\Delta$' + 'G A225U')
        plt.xlabel(r'$\Delta\Delta$' + 'G (kcal/mol)')
        plt.ylabel('Count')
        if tert2 is 'A225U':
            y = np.linspace(0,35)
            h = plt.plot(np.ones(50)*2.1, y)
            
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
    return ddG
def plotTertiaryContacts(variant_table, length=None):
    if length is None: length = 10
    subtable = variant_table.loc[variant_table.sublibrary == 'tertiary_contacts']
    subtable.loc[:, 'did_bind'] = subtable.qvalue < 0.05
    min_num_tests = 5
    bins = np.arange(-12, -5, 0.2)
    
    grouped = subtable.groupby('receptor')
    receptors = grouped.first().index
    order = ((grouped['did_bind'].sum() < grouped['did_bind'].count())&
        (grouped['did_bind'].sum() > 0))

    index = ((subtable.numTests >= min_num_tests) &
        (subtable.length == length) &
        (subtable.loop == "GGAA") &
        (pd.Series(np.in1d(subtable.receptor, receptors[order]), index=subtable.index)))
    

    g = sns.FacetGrid(subtable.loc[index], col="receptor", hue="did_bind",
                      col_wrap=3,
                      size=2,
                      margin_titles=True,
                      hue_kws={"color":[sns.xkcd_rgb['light grey'], sns.xkcd_rgb['vermillion']]})
    g.map(plt.hist, "dG", bins=bins, alpha=0.5, histtype='stepfilled')
    g.set_axis_labels("dG (kcal/mol)", "count");
    g.fig.subplots_adjust(wspace=.02, hspace=.2, left=0.15);
    
    
def plotTertvsTert(variant_table, tert1, tert2):
    subtable = variant_table.loc[variant_table.sublibrary == 'tertiary_contacts']
    subtable.loc[:, 'did_bind'] = subtable.qvalue < 0.05
    min_num_tests = 5
    index_t1 = (subtable.numTests >= 5)&(subtable.dG_ub - subtable.dG_lb < 1)&(subtable.qvalue< 0.05)&(subtable.receptor== tert1)
    index_t2 = (subtable.numTests >= 5)&(subtable.dG_ub - subtable.dG_lb < 1)&(subtable.qvalue< 0.05)&(subtable.receptor== tert2)
    table_t1 = subtable[index_t1]
    table_t2 = subtable[index_t2]
    index = np.arange(0,len(table_t1))
    table_t2_sorted = pd.DataFrame(index=np.arange(0,len(table_t1)), columns = table_t2.columns)
    
    for i, row in enumerate(table_t1.iterrows()):
        indx_t2_keep = (table_t2.junction_seq == row[1].junction_seq)&(table_t2.effective_length == row[1].effective_length)&(table_t2.helix_one_length == row[1].helix_one_length)
        if not table_t2[indx_t2_keep].empty:
            table_t2_sorted.iloc[i] = table_t2[indx_t2_keep].iloc[0]
    inds = (~pd.isnull(table_t2_sorted)).any(1).nonzero()[0]
    if len(inds) > 1:
        x = table_t1.iloc[inds]['dG'].values
        y = table_t2_sorted.iloc[inds]['dG'].values
        slope, intercept, r_value, p_value, std_err = scp.stats.linregress(x,y)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, aspect='equal')
        

        #ax2 = sns.jointplot(x, y, kind="reg", size=8)
        ax.scatter(x, y, marker='.', c='k', alpha=0.8)
        # plot line of slope one
        # line of slope 1
        origin = np.array([-12, -12-(x-y).mean()])
        ax.set_xlim([-12, -7])
        ax.set_ylim([-12, -7])
        ax.plot([-12, -7], [-12, -7], '--', color=sns.xkcd_rgb['vermillion'], label='yisx')
        sns.set_style("darkgrid")
 
        ax.plot([origin[0], origin[0]+5], [origin[1], origin[1]+5], '--', color=sns.xkcd_rgb['amber'], label='slope 1')
        ax.set_xlabel(tert1,fontsize=15)
        ax.set_ylabel(tert2,fontsize=15)
        ax = sns.regplot(x,y)
        #xlim_max = np.array([-12, -4])
        stringprint = ('R^2 = ' + str(round(r_value**2,2)) + '\n'
                       'slope = ' + str(round(slope,2)) + '\n')
                       #'y-int = '+ str(round(intercept,2)) + '\n')
                       
        
        ax.text(-11.75,-8, stringprint, fontsize=13)
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='major', color='w', linewidth=1.0)
        ax.grid(b=True, which='minor', color='w', linewidth=0.5)

    #X = sm.add_constant(x)
    
    #
    ## robust least squares
    #model = sm.RLM(y, X)
    #results = model.fit()
    

def makeTertMatrix(variant_table,figDirectory):
    #want to compare every tertiary contact to every other
    inds_r = [0, 1, 2, 3, 6, 7, 9, 10]
    subtable = variant_table.loc[variant_table.sublibrary == 'tertiary_contacts']
    subtable.loc[:, 'did_bind'] = subtable.qvalue < 0.05
    min_num_tests = 5
    grouped = subtable.groupby('receptor')
    receptors = grouped.first().index
    receptors = receptors[inds_r]
    
    for pair in map(','.join, itertools.combinations(receptors, 2)):
        terts = pair.split(',')
        if not(terts[0] == terts[1]):
            print pair
            tectoData.plotTertvsTert(variant_table,terts[0], terts[1])
            print 'scatter.%svs.%s.pdf'%(terts[0],terts[1])
            plt.savefig(os.path.join(figDirectory, 'scatter.%svs.%s.pdf'%(terts[0],terts[1])))
            plt.close()
    
    #fig = plt.figure(figsize=(7,7))
    #ax.scatter(x, y, marker='.', c='k', alpha=0.8)
    #
    #X = sm.add_constant(x)
    #xlim_max = np.array([-12, -4])
    #
    ## robust least squares
    #model = sm.RLM(np.asarray(y), np.asarray(x) )
    #results = model.fit()
    #ax.plot(xlim_max,  results.params.dG*xlim_max + results.params.const, 'r:', label='best fit')
    
def plotAbsFluorescence(a, index, concentrations=None):
    
    if concentrations is None:
        concentrations = ['%4.2fnM'%d for d in 2000*np.power(3., np.arange(0, -8, -1))][::-1]
    numconcentrations = len(concentrations)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=numconcentrations-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=False))
    with sns.axes_style('white'):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        for i in range(numconcentrations):
            counts, xbins, patches = ax.hist(a.loc[index, i].dropna().values, bins=np.arange(0, 2000, 10), histtype='stepfilled', alpha=0.1,
                     color = scalarMap.to_rgba(i))
            if (counts == 0).sum() > 0:
                index2 = np.arange(0, np.ravel(np.where(counts == 0))[0])
            else:
                index2 = np.arange(len(counts))
            ax.plot(((xbins[:-1] + xbins[1:])*0.5)[index2], counts[index2], color=scalarMap.to_rgba(i), label=concentrations[i])
        ax.set_yscale('log')
        ax.set_ylim(1, 10**5)
        plt.legend()
        ax.set_ylabel('number of clusters')
        ax.set_xlabel('absolute fluorescence')
        plt.tight_layout()
    pass

def plotDeltaAbsFluorescence(a, filterName, concentrations=None):
    
    a = a.copy()
    a.loc[:, 'null'] = ''
    a.loc[[str(s).find(filterName) == -1 for s in a.loc[:, 'filter']], 'null'] = 'yes'
    a.loc[[str(s).find(filterName) > -1 for s in a.loc[:, 'filter']], 'null'] = 'no'
    

    if concentrations is None:
        concentrations = ['%4.2fnM'%d for d in 2000*np.power(3., np.arange(0, -8, -1))][::-1]
    numconcentrations = len(concentrations)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=numconcentrations-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=False))
    with sns.axes_style('white'):
        fig = plt.figure(figsize=(5,8))
        bins = np.arange(0, 4000, 10)
        for i in range(numconcentrations):
            ax = fig.add_subplot(8, 1, i+1)
            for cat in ['yes', 'no']:
                if cat == 'no': color = scalarMap.to_rgba(i)
                else: color = '0.5'
                index = a.null == cat
                counts, xbins, patches = ax.hist(a.loc[index, i].dropna().values, bins=bins, histtype='stepfilled', alpha=0.1,
                         color = color)
                if (counts == 0).sum() > 0:
                    index2 = np.arange(0, np.ravel(np.where(counts == 0))[0])
                else:
                    index2 = np.arange(len(counts))
                ax.plot(((xbins[:-1] + xbins[1:])*0.5)[index2], counts[index2], color=color, label=concentrations[i])
            ax.set_yscale('log')
            ax.set_ylim(1, 10**5)
            ax.set_yticks(np.power(10, range(0, 5)))
            #ax.set_ylabel('number of clusters')
            if i == numconcentrations-1:
                ax.set_xlabel('absolute fluorescence')
            else:
                ax.set_xticklabels([])
        plt.subplots_adjust(hspace=0.02, bottom=0.1, top=0.95, left=0.15)
    pass

def plotThreeWayJunctions(variant_table):
    subtable = variant_table.loc[variant_table.sublibrary == 'three_way_junctions']
    subtable.loc[:, 'did_bind'] = subtable.qvalue < 0.05
    
    min_num_tests = 5
    bins = np.arange(-12, -5, 0.2)
    
    seqs = np.unique(subtable.junction_seq)
    for seq in np.array_split(seqs, 2):
        print seq
        index = (subtable.numTests >= min_num_tests)&np.in1d(subtable.junction_seq, seq)
        g = sns.FacetGrid(subtable.loc[index], col="loop", row="junction_seq", hue="did_bind",
                          size=2,
                          margin_titles=True,
                          aspect=1.5,
                          hue_kws={"color":[sns.xkcd_rgb['vermillion'], sns.xkcd_rgb['light grey']]})
        g.map(plt.hist, "dG", bins=bins, alpha=0.5, histtype='stepfilled')
        g.set_axis_labels("\DeltaG (kcal/mol)", "count");
        g.fig.subplots_adjust(wspace=.1, hspace=.05, left=0.15);
        
def plotThreeWayJunctions_specific(variant_table):
    subtable = variant_table.loc[variant_table.sublibrary == 'three_way_junctions']
    subtable.loc[:, 'did_bind'] = subtable.qvalue < 0.05
    
    min_num_tests = 5
    bins = np.arange(-12, -5, 0.2)
    
    seqs = np.unique(subtable.junction_seq)
    seqs_specific = seqs[[4,0,6]]
    index = (subtable.numTests >= min_num_tests)&np.in1d(subtable.junction_seq, seqs_specific)
    sns.set(font_scale=2)
    g = sns.FacetGrid(subtable.loc[index], col="loop", row="junction_seq", hue="did_bind",
                      size=5,
                      margin_titles=True,
                      aspect=1.3,
                      row_order = seqs_specific, 
                      hue_kws={"color":[sns.xkcd_rgb['vermillion'], sns.xkcd_rgb['light grey']]})
    g = (g.map(plt.hist, "dG", bins=bins, alpha=0.5, histtype='stepfilled')).set_titles("diners")
    g.set_axis_labels(r'$\Delta$' + "G (kcal/mol)", "count");
    g.fig.subplots_adjust(wspace=.2, hspace=.15, left=0.15);
    plt.setp(g.fig.texts, text="")
    g.set_titles(row_template="", col_template="")
    
    #g.col_names.remove    
def plotThreeWaySwitch(variant_table, figDirectory):
    subtable = variant_table.loc[variant_table.sublibrary == 'three_way_junctions']
    subtable.loc[:, 'did_bind'] = subtable.qvalue < 0.05
    min_num_tests = 5
    bins = np.arange(-12, -5, 0.2)
    topos = np.unique(subtable.junction_seq)
    table_threeway_all = pd.DataFrame(columns = subtable.columns)
    for topo in topos:
        index_t = (subtable.junction_seq== topo)&(subtable.numTests >= 5)&(~np.isnan(subtable.dG))
        table_t = subtable[index_t]
        table_GGAA = pd.DataFrame(columns=table_t.columns)
        table_UUCG = pd.DataFrame(columns=table_t.columns)
        for flank in np.unique(subtable.flank):
            index_GGAA = (table_t.flank== flank)&(table_t.loop== 'GGAA_UUCG')
            if not table_t[index_GGAA].empty:
                for i, row in enumerate(table_t[index_GGAA].iterrows()):
                    #print row
                    index_UUCG = (table_t.helix_one_length == row[1].helix_one_length)&(table_t.loop == 'UUCG_GGAA')&(table_t.flank== flank)
                    if not table_t[index_UUCG].empty:
                       table_GGAA = table_GGAA.append(table_t[index_GGAA].iloc[i])
                       table_UUCG = table_UUCG.append(table_t[index_UUCG])
                    
        x = table_GGAA['dG'].values
        y = table_UUCG['dG'].values
        KGGAA = np.exp(-x/0.58)
        KUUCG = np.exp(-y/0.58)
        Kbind = KUUCG + KGGAA
        Kconf = KGGAA/KUUCG
        dGbind = -0.58*np.log(Kbind)
        dGconf = -0.58*np.log(Kconf)
        table_GGAA.loc[:,'KGGAA'] = KGGAA
        table_GGAA.loc[:,'KUUCG'] = KUUCG
        table_GGAA.loc[:,'Kbind'] = Kbind
        table_GGAA.loc[:,'Kconf'] =  Kconf
        table_GGAA.loc[:,'dGbind'] = dGbind
        table_GGAA.loc[:,'dGconf'] = dGconf
        table_threeway_all = table_threeway_all.append(table_GGAA)
        
        
        slope, intercept, r_value, p_value, std_err = scp.stats.linregress(x,y)
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, aspect='equal')
        

        #ax2 = sns.jointplot(x, y, kind="reg", size=8)
        ax.scatter(x, y, marker='.', c='k', alpha=0.8)
        # plot line of slope one
        # line of slope 1
        #origin = np.array([-12, -12-(x-y).mean()])
        ax.set_xlim([-12, -7])
        ax.set_ylim([-12, -7])
        #ax.plot([-12, -7], [-12, -7], '--', color=sns.xkcd_rgb['vermillion'], label='yisx')
        sns.set_style("darkgrid")
 
        #ax.plot([origin[0], origin[0]+5], [origin[1], origin[1]+5], '--', color=sns.xkcd_rgb['amber'], label='slope 1')
        ax.set_xlabel('GGAA',fontsize=15)
        ax.set_ylabel('UUCG',fontsize=15)
        ax = sns.regplot(x,y)
        #xlim_max = np.array([-12, -4])
        stringprint = ('R^2 = ' + str(round(r_value**2,2)) + '\n'
                       'slope = ' + str(round(slope,2)) + '\n')
                       #'y-int = '+ str(round(intercept,2)) + '\n')
                       
        
        ax.text(-11.75,-8, stringprint, fontsize=13)
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='major', color='w', linewidth=1.0)
        ax.grid(b=True, which='minor', color='w', linewidth=0.5)
        fig.suptitle(topo)
        plt.savefig(os.path.join(figDirectory, 'threewayUUCGvsGGAA.topology%s.pdf'%(topo)))
        plt.close('all')
        
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0,0])
        
        ax1.hist(Kconf , bins=np.linspace(0, 200, 50 ), color=sns.xkcd_rgb['vermillion'], histtype='stepfilled', alpha=0.5)
        plt.xlabel('Kconf')
        plt.ylabel('count')
        plt.tight_layout()
        ax2 = fig.add_subplot(gs[0,1])
        ax2.hist(dGconf, bins=np.linspace(-3.5, 3.5, 40), color=sns.xkcd_rgb['vermillion'], histtype='stepfilled', alpha=0.5,label="dGconf")
        ax2.hist(dGbind, bins=np.linspace(-12.5, -5.5, 40), color=sns.xkcd_rgb['amber'], histtype='stepfilled', alpha=0.5, label="dGbind")
        ax2.legend()
        plt.xlabel('dG (kcal/mol)')
        plt.ylabel('count')
        
        ax3 = fig.add_subplot(gs[1,0])
        ax3 = sns.boxplot(table_GGAA.dGbind, groupby=np.sort(table_GGAA.helix_one_length))
        plt.ylabel('dGbind (kcal/mol)')
        ax3.set_ylim([-12.5,-5.5])
        ax4 = fig.add_subplot(gs[1,1])
        ax4 = sns.boxplot(table_GGAA.dGconf, groupby=np.sort(table_GGAA.helix_one_length))
        ax4.set_ylim([-3.5, 3.5])
        plt.ylabel('dGconf (kcal/mol)')
        fig.suptitle(topo)
        #grouped = subtable.groupby('receptor')
        plt.savefig(os.path.join(figDirectory, 'threeway.topology%s.energetics.pdf'%(topo)))
    return table_threeway_all   
    #need to color by length

def compareTopos(variant_table, figDirectory):
    subtable = plotThreeWaySwitch(variant_table, figDirectory)
    subtable = subtable.sort('flank')
    topos = np.unique(subtable.junction_seq)
    lengths = np.unique(subtable.helix_one_length)
    #topology of no insertions
    
    #topo_ = topos[4]
    counter = 1
    for topo_, topo, in itertools.combinations(topos, 2):
        print counter, topo_, topo
        counter = counter +1
        if topo_ is not topo:
            table_ = subtable[subtable.junction_seq== topo_]
            dGconf_ = np.array([])
            dGconf_2 = np.array([])
            for length in lengths:
                subtable_ = table_[(table_.helix_one_length == length)]
                flanks_ = subtable_.flank
                index_t2 = (subtable.junction_seq== topo)&(subtable.helix_one_length== length)
                subtable_t2 = subtable[index_t2]
                index_keep =  np.in1d(subtable[index_t2].flank, flanks_)
                index_keep_ =  np.in1d(flanks_,subtable[index_t2].flank)
                dGconf_ = np.append(dGconf_,  subtable_[index_keep_].dGconf)
                dGconf_2 = np.append(dGconf_2,  subtable_t2[index_keep].dGconf)
            slope, intercept, r_value, p_value, std_err = scp.stats.linregress(dGconf_,dGconf_2)    
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, aspect='equal')
            ax.scatter(dGconf_, dGconf_2, marker='.', c='k', alpha=0.8)
            # plot line of slope one
            # line of slope 1
            origin = np.array([-2.5, -2.5-(dGconf_-dGconf_2).mean()])
            ax.set_xlim([-2.5, 2])
            ax.set_ylim([-2.5, 2])
            ax.plot([-2.5, 2], [-2.5,2], '--', color=sns.xkcd_rgb['vermillion'], label='yisx')
            ax.plot([origin[0], origin[0]+5], [origin[1], origin[1]+5], '--', color=sns.xkcd_rgb['amber'], label='slope 1')
            ax = sns.regplot(dGconf_,dGconf_2)
            plt.xlabel('dGconf %s'%(topo_))
            plt.ylabel('dGconf %s'%(topo))
            plt.title('Compare three way junctions of topology %s vs %s'%(topo,topo_))
            sns.set_style("darkgrid")
            plt.savefig(os.path.join(figDirectory, 'threeway.dGconf_topology%svs%s.pdf'%(topo, topo_)))
            plt.close()
        
  

    
def plotScatterplotDelta(variant_tables):
    index = pd.concat([(variant_table.numTests >= 5)&
        (variant_table.dG_ub - variant_table.dG_lb <= 1)
        &(variant_table.qvalue < 0.05)
        for variant_table in variant_tables], axis=1).all(axis=1)
    x = variant_tables[0].loc[index, 'dG']
    y = variant_tables[1].loc[index, 'dG']

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(x, y, marker='.', c='k', alpha=0.1)

    X = sm.add_constant(x)
    xlim_max = np.array([-12, -4])
    
    # robust least squares
    model = sm.RLM(y, X )
    results = model.fit()
    ax.plot(xlim_max,  results.params.dG*xlim_max + results.params.const, 'r:', label='best fit')
    
    # plot line of slope one
    # line of slope 1
    origin = np.array([-12, -12-(x-y).mean()])
    ax.plot([origin[0], origin[0]+5], [origin[1], origin[1]+5], '--', color=sns.xkcd_rgb['turquoise'], label='slope 1')
    

    pass

def plotSequenceJoeLengthChanges(variant_table):
    subtable = variant_table.loc[variant_table.sublibrary=='sequences']
    grouped = subtable.groupby('helix')
    helices = grouped['dG'].count().loc[(grouped['dG'].count() > 1)].index
    subtable = subtable.loc[np.in1d(subtable.helix, helices)]
    
    newtable = pd.DataFrame(index=helices, columns=[9, 10, '11G', '11T'])
    for name, group in subtable.groupby('helix'):
        sub = subtable.loc[subtable.helix==name]
        index = (sub.numTests >= 5)&(sub.dG_ub - subtable.dG_lb < 1)&(sub.length != 11)
        newtable.loc[name, sub.loc[index].length] = sub.loc[index].dG.values
        
        # 11 p is more complicated
        index = (sub.numTests >= 5)&(sub.dG_ub - subtable.dG_lb < 1)&(sub.length == 11)
        for variant in index.loc[index].index:
            if sub.loc[variant, 'sequence'][15] == 'T':
                newtable.loc[name, '11T'] = sub.loc[variant].dG
            elif sub.loc[variant, 'sequence'][15] == 'G':
                newtable.loc[name, '11G'] = sub.loc[variant].dG
            
    
    fig1 = plt.figure(figsize=(4,3.5))
    fig2 = plt.figure(figsize=(4,3.5))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    for helix in helices:
        ax1.plot([9, 10, 11], newtable.loc[helix, [9, 10, '11T']], 'o-', alpha=0.1, c='k')
        ax1.set_title('11T')
        ax2.plot([9, 10, 11], newtable.loc[helix, [9, 10, '11G']], 'o-', alpha=0.1, c='k')
        ax2.set_title('11T')
    for ax in [ax1, ax2]:
        ax.set_xlim(7.5, 12.5)
        ax.set_ylim(-12, -5)
        ax.set_xlabel('length (bp)')
        ax.set_ylabel('dG (kcal/mol)')
        ax.set_xticks([8, 9, 10, 11, 12])
    
    for fig in [fig1, fig2]:
        fig.subplots_adjust(bottom=0.15, left=0.15)
    
def plotSequencesLengthChanges(variant_table):

    subtable = variant_table.loc[variant_table.sublibrary=='sequences_tiled']
    flanks = np.unique(subtable.flank)
    newtable = pd.DataFrame(index= flanks, columns=[8, 9, 10, 11, 12])
    for name, group in subtable.groupby('flank'):
        sub = subtable.loc[subtable.flank==name]
        index = (sub.numTests >= 5)
        newtable.loc[name, sub.loc[index].length] = sub.loc[index].dG.values

    fig = plt.figure(figsize=(4,3.5))
    ax = fig.add_subplot(111)
    for flank in flanks:
        ax.plot([8, 9, 10, 11, 12], newtable.loc[flank], 'o-', alpha=0.1, c='k')
    ax.set_xlim(7.5, 12.5)
    ax.set_ylim(-12, -5)
    ax.set_xlabel('length (bp)')
    ax.set_ylabel('dG (kcal/mol)')
    ax.set_xticks([8, 9, 10, 11, 12])
    fig.subplots_adjust(bottom=0.15, left=0.15)
    
def compareSeqTiled(variant_table, length=None):
    if length is None:
        length = 10
        
    index = (variant_table.length==length)&(variant_table.numTests >=5)&(variant_table.dG_ub - variant_table.dG_lb < 1)    
    index2 = (variant_table.sublibrary == 'sequences')&index
    index1 = (variant_table.sublibrary == 'sequences_tiled')&index
    
    bins = np.arange(-12, -5, 0.1)
    plt.figure(figsize=(4,4));

    sns.distplot(variant_table.loc[index1, 'dG'].values, hist_kws={'histtype':'stepfilled'}, color='seagreen', label='tiled')
    sns.distplot(variant_table.loc[index2, 'dG'].values, hist_kws={'histtype':'stepfilled'}, color='grey', label='random')
    
    plt.xlabel('dG (kcal/mol)')
    plt.ylabel('probability density')
    plt.tight_layout()
    
def plotScatterplotReplicates(variant_tables, labels=None, index=None):
    if labels is None:
        labels = ['rep1', 'rep2']
    if index is None:
        index = pd.concat([(vtable.pvalue <= 0.05)&(vtable.numTests >= 5) for vtable in variant_tables], axis=1).all(axis=1)
    subtable = pd.concat([vtable.loc[index, 'dG'] for vtable in variant_tables], axis=1).dropna()
    subtable.columns = labels
    
    xlim = [-11.75, -5.75]
    plt.figure(figsize=(4,4))
    plt.hexbin(subtable.loc[:, labels[0]], subtable.loc[:, labels[1]], bins='log')
    plt.plot(xlim, xlim, 'r:')
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.xlabel('dG (kcal/mol) %s'%labels[0])
    plt.ylabel('dG (kcal/mol) %s'%labels[1])
    r, pvalue = st.pearsonr(subtable.loc[:, labels[0]], subtable.loc[:, labels[1]])
    plt.annotate('rsq = %4.3f\nsigma= %4.3f'%(r**2, (subtable.loc[:, labels[0]]- subtable.loc[:, labels[1]]).std()), xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top',
                    fontsize=12)
    plt.tight_layout()
    
    # plot residuals
    plt.figure(figsize=(4,4))
    plt.hist((subtable.loc[:, labels[0]]- subtable.loc[:, labels[1]]).values , bins=np.linspace(-1, 1, 100), color=sns.xkcd_rgb['vermillion'], histtype='stepfilled', alpha=0.5)
    plt.xlabel('ddG (kcal/mol) %s - %s'%(labels[0], labels[1]))
    plt.ylabel('count')
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.plot([0, 0], ylim, 'k:')
    plt.tight_layout()
    return

def plotLoopFlankers(variant_table):
    
    subtable = variant_table.loc[variant_table.sublibrary == 'loop_flanking']
    subtable.loc[:, 'side'] = [s[0] for s in subtable.helix]
    index = (subtable.numTests >=5)&(subtable.dG_ub - subtable.dG_lb < 1)
    bins = np.arange(-12, -8, 0.2)
    g = sns.FacetGrid(subtable.loc[index], row="side", col="no_flank",
                      size=2,
                      margin_titles=True,
                     )
    g.map(plt.hist, "dG", bins=bins, alpha=0.5, histtype='stepfilled')
    g.set_axis_labels("dG (kcal/mol)", "count");
    g.set_xticklabels([-12, '', -11, '', -10, '', -9, '', ''])
    g.fig.subplots_adjust(wspace=.02, hspace=.2, left=0.15);
    
def plotLengthChangesPerSequence(variant_table):
    
    # for all base paired sequences, no_flank is all base paired sequences
    seqs = []
    for x, y, in itertools.product('AGCU', 'AGCU'):
        seq = ''.join([x, y])
        seqs.append('_'.join([seq, seqfun.reverseComplement(seq, rna=True)]))
    
    index = (variant_table.numTests>=5)&(variant_table.sublibrary=='junction_conformations')
    subtable = variant_table.loc[index & np.in1d(variant_table.no_flank, seqs)]
    
    byLength = {}
    for seq in np.unique(subtable.junction_seq):
        byLength[seq] = subtable.loc[subtable.junction_seq == seq].pivot(index='offset', columns='length', values='dG').loc[[0]]
    byLength = pd.concat(byLength)
    
    fig = plt.figure(figsize=(4,3.5))
    ax = fig.add_subplot(111)
    for flank in np.unique(subtable.junction_seq):
        ax.plot([8, 9, 10, 11, 12], byLength.loc[(flank,0)], 'o-', alpha=0.1, c='k')
    ax.set_xlim(7.5, 12.5)
    ax.set_ylim(-12, -5)
    ax.set_xlabel('length (bp)')
    ax.set_ylabel('dG (kcal/mol)')
    ax.set_xticks([8, 9, 10, 11, 12])
    plt.tight_layout()
    
def plotFmaxBoxplots(variant_table, table):

    # do it also in bins of dG
    binedges = np.arange(-12.1, -6, 0.5)
    variant_table.loc[:, 'binned_dGs'] = np.digitize(variant_table.dG, binedges)
    index = variant_table.numTests >= 5
    
    plt.figure();
    sns.boxplot(x="binned_dGs", y="fmax", data=variant_table.loc[index],
                order=np.unique(variant_table.binned_dGs),
                palette="Blues_d")
    plt.ylim(0, 5000)
    plt.xticks(np.arange(len(binedges)), ['%3.1f:%3.1f'%(i, j) for i, j in itertools.izip(binedges[:-1], binedges[1:])]+['>%4.1f'%binedges[-1]], rotation=90)
    plt.ylabel('median of single cluster fit fmax')
    plt.tight_layout()
    
    # plot median last binding point
    grouped = table.loc[table.barcode_good].groupby('variant_number')
    variant_table.loc[:, 'all_cluster_signal'] = grouped['all_cluster_signal'].median()

    plt.figure();
    sns.boxplot(x="binned_dGs", y="all_cluster_signal", data=variant_table.loc[index],
                order=np.unique(variant_table.binned_dGs),
                palette="Blues_d")
    plt.ylim(0, 5000)
    plt.xticks(np.arange(len(binedges)), ['%3.1f:%3.1f'%(i, j) for i, j in itertools.izip(binedges[:-1], binedges[1:])]+['>%4.1f'%binedges[-1]], rotation=90)
    plt.ylabel('median fluorescence in red')
    plt.tight_layout()
    
    # plot normalized fmax
    variant_table.loc[:, 'norm_fmax'] = variant_table.fmax/variant_table.all_cluster_signal

    plt.figure();
    sns.boxplot(x="binned_dGs", y="norm_fmax", data=variant_table.loc[index],
                order=np.unique(variant_table.binned_dGs),
                palette="Blues_d")
    plt.ylim(0, 2)
    plt.xticks(np.arange(len(binedges)), ['%3.1f:%3.1f'%(i, j) for i, j in itertools.izip(binedges[:-1], binedges[1:])]+['>%4.1f'%binedges[-1]], rotation=90)
    plt.ylabel('median of single cluster fit fmax / fluorescence in red')
    plt.tight_layout()
    
    # plot single cluster fit fmaxs
    table.loc[:, 'binned_dGs'] = np.digitize(table.dG, binedges)

    plt.figure();
    sns.boxplot(x="binned_dGs", y="fmax", data=table,
                order=np.unique(table.binned_dGs),
                palette="Blues_d")
    plt.ylim(0, 5000)
    plt.xticks(np.arange(len(binedges)), ['%3.1f:%3.1f'%(i, j) for i, j in itertools.izip(binedges[:-1], binedges[1:])]+['>%4.1f'%binedges[-1]], rotation=90)
    plt.ylabel('single cluster fit fmax')
    plt.tight_layout()