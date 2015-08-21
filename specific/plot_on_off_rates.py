from fitOnOffRates import checkFits

# load results on/off
#offrate_file = '/lab/sarah/RNAarray/150605_onchip_binding/WC/offRates/AG3EL_Bottom_filtered_reduced.CPresults'
#onrate_file  = '/lab/sarah/RNAarray/150605_onchip_binding/WC/onRates/AG3EL_Bottom_filtered_reduced.CPresults'

offrate_file = '/lab/sarah/RNAarray/141111_miseq_run_tecto_TAL_VR/with_all_clusters/WC/offRates/AAYFY_ALL_Bottom_filtered_reduced.CPresults'
#libCharFile = '/lab/sarah/RNAarray/150311_library_v2/all_10expts.library_characterization.txt'
libCharFile = '/lab/sarah/RNAarray/140929_library_design/allJunctions.noUs.noMB2.characterization'
#variant_table = IMlibs.loadLibCharVariantTable(libCharFile, 'bindingCurves/AG3EL_Bottom_filtered_reduced.CPvariant')
variant_table = IMlibs.loadLibCharVariantTable(libCharFile, '/lab/sarah/RNAarray/141111_miseq_run_tecto_TAL_VR/with_all_clusters/WC/bindingCurves/AAYFY_ALL_Bottom_filtered_reduced.CPvariant')

results_off = pd.read_table(offrate_file, index_col=0)
results_on  = pd.read_table(onrate_file, index_col=0)

# plot kobs versus other parameters
index_on = checkFits(results_on, fittype='on')
fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(1, 2, bottom=0.25, wspace=0.3, right=0.95)

ax = fig.add_subplot(gs[0,0])
ax.scatter(np.log10(results_on.loc[~index_on].kobs.astype(float)), results_on.loc[~index_on].fmin, marker='.', alpha=0.5,
           facecolors='0.5', edgecolors='none')
ax.scatter(np.log10(results_on.loc[index_on].kobs.astype(float)),  results_on.loc[index_on].fmin, marker='.', alpha=0.5,
           facecolors='r', edgecolors='none')
ax.set_xlabel('log$_{10}$$(kobs)$')
ax.set_ylabel('fmin')
ax.tick_params(right='off', top='off')
ax.set_xlim(-6, 0)
ax.set_ylim(0, .6)

ax = fig.add_subplot(gs[0,1])
ax.scatter(np.log10(results_on.loc[~index_on].kobs.astype(float)), results_on.loc[~index_on].fmax, marker='.', alpha=0.5,
           facecolors='0.5', edgecolors='none')
ax.scatter(np.log10(results_on.loc[index_on].kobs.astype(float)),  results_on.loc[index_on].fmax, marker='.', alpha=0.5,
           facecolors='r', edgecolors='none')
ax.set_xlabel('log$_{10}$$(kobs)$')
ax.set_ylabel('fmax')
ax.tick_params(right='off', top='off')
ax.set_xlim(-6, 0)
ax.set_ylim(0, 0.6)

index_off = checkFits(results_off, fittype='off')
fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(1, 2, bottom=0.25, wspace=0.3, right=0.95)

ax = fig.add_subplot(gs[0,0])
ax.scatter(np.log10(results_off.loc[~index_off].koff.astype(float)), results_off.loc[~index_off].fmin, marker='.', alpha=0.5,
           facecolors='0.5', edgecolors='none')
ax.scatter(np.log10(results_off.loc[index_off].koff.astype(float)),  results_off.loc[index_off].fmin, marker='.', alpha=0.5,
           facecolors='r', edgecolors='none')
ax.set_xlabel('log$_{10}$$(koff)$')
ax.set_ylabel('fmin')
ax.tick_params(right='off', top='off')
ax.set_xlim(-6, 0)
ax.set_ylim(0, .6)

ax = fig.add_subplot(gs[0,1])
ax.scatter(np.log10(results_off.loc[~index_off].koff.astype(float)), results_off.loc[~index_off].fmax, marker='.', alpha=0.5,
           facecolors='0.5', edgecolors='none')
ax.scatter(np.log10(results_off.loc[index_off].koff.astype(float)),  results_off.loc[index_off].fmax, marker='.', alpha=0.5,
           facecolors='r', edgecolors='none')
ax.set_xlabel('log$_{10}$$(koff)$')
ax.set_ylabel('fmax')
ax.tick_params(right='off', top='off')
ax.set_xlim(-6, 0)
ax.set_ylim(0, 3)

# plot koff versus kobs
variants = pd.concat([index_off, index_on], axis=1, keys=['off', 'on']).all(axis=1)
c = 2.743484225
kds1 = results_off.koff*c/(results_on.kobs - results_off.koff)
kds2 = parameters.find_Kd_from_dG(variant_table.dG.astype(float))

plt.figure(figsize=(4,4))
plt.scatter(kds2.loc[~variants], kds1.loc[~variants], marker='.', alpha=0.5,
           facecolors='0.5', edgecolors='none')
plt.scatter(kds2.loc[variants], kds1.loc[variants], marker='.', alpha=0.5,
           facecolors='r', edgecolors='none')
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(right='off', top='off')
plt.xlim(5E-1, 1E3)
plt.ylim(5E-1, 1E3)
plt.xlabel('$K_d$ (nM) binding curves')
plt.ylabel('$K_d$ (nM) on/off rates')
plt.tight_layout()

#plot koff versus obs versus binding equilibria
c = 2.743484225
kds1 = results_off.koff*c/(results_on.kobs - results_off.koff)
e_kds1 = (fitFun.errorPropagationKdFromKoffKobs(results_off.koff, results_on.kobs, c,
                              results_off.koff-results_off.koff_lb,
                                 results_on.kobs-results_on.kobs_lb)+
          fitFun.errorPropagationKdFromKoffKobs(results_off.koff, results_on.kobs, c,
                                 results_off.koff_ub-results_off.koff,
                                 results_on.kobs_ub-results_on.kobs))/2.
kds2 = parameters.find_Kd_from_dG(variant_table.dG.astype(float))
e_kds2 = (fitFun.errorProgagationKdFromdG(variant_table.dG,
                                   variant_table.dG-variant_table.dG_lb)+
          fitFun.errorProgagationKdFromdG(variant_table.dG,
                                   variant_table.dG_ub-variant_table.dG))/2.
index = variants&(e_kds1<5)&(e_kds2<5)

plt.figure(figsize=(3,3))
#plt.scatter(kds2.loc[index], kds1.loc[index], marker='.', alpha=0.5,
#           facecolors='k', edgecolors='none')
sns.kdeplot(kds2.loc[variants], kds1.loc[variants], clip=[[0, 20], [0, 20]],
            shade=True, cmap='binary', n_levels=25)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(right='off', top='off')
plt.xlim(.5, 20)
plt.ylim(.5, 20)
plt.xlabel('$K_d$ (nM) binding curves')
plt.ylabel('$K_d$ (nM) on/off rates')
plt.tight_layout()

# delta
variant = 34936

dG = variant_table.dG
dG_dagger = parameters.find_dG_from_Kd(results_off.koff.astype(float))

kds = parameters.find_Kd_from_dG(dG.astype(float))
dG_dagger_on = parameters.find_dG_from_Kd((results_off.koff/kds).astype(float))
x = (dG - dG.loc[variant]).loc[variants]
y = (dG_dagger - dG_dagger.loc[variant]).loc[variants]
z = -(dG_dagger_on - dG_dagger_on.loc[variant]).loc[variants]

fig = plt.figure(figsize=(3,3));
ax = fig.add_subplot(111, aspect='equal')
ax.tick_params(top='off', right='off')
#plt.scatter(x, y, marker='.', alpha=0.5)
#plt.scatter(x, y,  marker='.', alpha=0.5, c='k')
im = plt.hexbin(x, y,  extent=[-0.5, 2, -0.5, 2], gridsize=100, bins='log')
#sns.kdeplot(x, z,  cmap="Blues", shade=True, shade_lowest=False)
slope, intercept, r_value, p_value, std_err = st.linregress(x,y)

xlim = np.array(ax.get_xlim())
plt.plot(xlim, xlim*slope + intercept, 'c--', linewidth=1)
plt.plot(xlim, xlim, 'r:', linewidth=1)
plt.xlabel('$\Delta \Delta G$')
plt.ylabel('$\Delta \Delta G_{off}\dagger$')

plt.xlim(-0.1, 1.7)
plt.ylim(-0.1, 1.5)
#plt.colorbar(im)
plt.tight_layout()

fig = plt.figure(figsize=(3,3));
ax = fig.add_subplot(111, aspect='equal')
ax.tick_params(top='off', right='off')
#plt.scatter(x, y, marker='.', alpha=0.5)
#plt.scatter(x, y,  marker='.', alpha=0.5, c='k')
im = plt.hexbin(x, z,  extent=[-0.5, 2, -0.5, 2], gridsize=100, bins='log')
#sns.kdeplot(x, z,  cmap="Blues", shade=True, shade_lowest=False)
slope, intercept, r_value, p_value, std_err = st.linregress(x,z)

xlim = np.array(ax.get_xlim())
plt.plot(xlim, xlim*slope + intercept, 'c--', linewidth=1)
plt.plot(xlim, xlim, 'r:', linewidth=1)
plt.xlabel('$\Delta \Delta G$')
plt.ylabel('$\Delta \Delta G_{on}\dagger$')

plt.xlim(-0.1, 1.7)
plt.ylim(-0.1, 1.5)
#plt.colorbar(im)
plt.tight_layout()