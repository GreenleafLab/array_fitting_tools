import itertools
import pandas as pd
import os
import numpy as np
import IMlibs
import fitFun
parameters = fitFun.fittingParameters()

def errorPropagateAverage(sigmas, weights):
    sigma_out = np.sqrt(np.sum([np.power(weight*sigma/weights.sum(), 2)
                                if weight != 0
                                else 0
                                for sigma, weight in itertools.izip(sigmas, weights)]))
    return sigma_out

def errorPropagateAverageAll(sigmas, weights, index=None):
    if index is None:
        index = sigma_dGs.index
    subweights = weights.loc[index].astype(float)
    subsigmas = sigmas.loc[index].astype(float)
    
    sigma_out = pd.Series([errorPropagateAverage(subsigmas.loc[i], weights.loc[i])
                           for i in subsigmas.index], index=subsigmas.index)
    
    return sigma_out

def weightedAverage(values, weights):
    average = (np.sum([value*weight
                       if weight != 0
                       else 0
                       for value, weight in itertools.izip(values, weights)])/
                weights.sum())
    return average

def weightedAverageAll(values, weights, index=None):
    if index is None:
        index = values.index
    
    subvalues = values.loc[index]
    subweights = weights.loc[index]
    average = pd.Series([weightedAverage(subvalues.loc[i], weights.loc[i])
                           for i in subvalues.index], index=subvalues.index)
    return average


if __name__ == '__main__':
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
       
    outFile = os.path.join(dirname, flow, 'bindingCurves', '%s_Bottom_filtered_reduced'%chip)
    variant_table = IMlibs.loadLibCharVariantTable(libCharFile, outFile + '.CPvariant')
    
    variant_tables = [variant_table_wc, variant_table]
    
    # make flags
    
    # 0x1 data from rep 1
    # 0x2 data from rep 2
    # 0x4 data not measured in rep 1
    # 0x8 data not measured in rep 2
    
    
    index = ((pd.concat(variant_tables, axis=1).loc[:, 'numTests'] >=5).all(axis=1)&
             (pd.concat(variant_tables, axis=1).loc[:, 'dG'] < parameters.cutoff_dG).all(axis=1))
    values = pd.concat([table.dG for table in variant_tables],
                       axis=1, keys=['0', '1'])
    numTests = pd.concat([table.numTests for table in variant_tables],
                       axis=1, keys=['0', '1'])
    numTests.fillna(0, inplace=True)
    
    # correct for offset between 1st and 2nd measurement
    offset = 0.24
    values.iloc[:, 1] -= offset
    
    # find error measurements
    eminus = pd.concat([(table.dG - table.dG_lb) for table in variant_tables],
                         axis=1, keys=values.columns)
    eplus  = pd.concat([(table.dG_ub - table.dG) for table in variant_tables],
                         axis=1, keys=values.columns)
    variance = ((eminus + eplus)/2)**2
    weights = pd.DataFrame(data=1, index=variance.index, columns=variance.columns)
    index = (variance > 0).all(axis=1)
    weights.loc[index] = 1/variance.loc[index]
    
    # final variant table
    cols =  ['numTests1', 'numTests2', 'weights1', 'weights2', 'numTests', 'dG', 'eminus', 'eplus', 'flag']
    
    results = pd.DataFrame(index=variant_table.index, columns=cols)
    results.loc[:, ['numTests1', 'numTests2', 'numTests', 'flag']] = 0
    
    # if one of the measurements has less than 5 clusters, use only the other measurement
    indexes = [(numTests.iloc[:, 0] < 5)&(numTests.iloc[:, 1] >= 5),
               (numTests.iloc[:, 0] >= 5)&(numTests.iloc[:, 1] < 5),
               (numTests >= 5).all(axis=1)]
    weights.loc[indexes[0], '0'] = 0
    weights.loc[indexes[1], '1'] = 0
    
    for i, index in enumerate(indexes):
        results.loc[index, 'dG']     = weightedAverageAll(values, weights, index=index)
        results.loc[index, 'eminus'] = errorPropagateAverageAll(eminus, weights, index=index)
        results.loc[index, 'eplus']  = errorPropagateAverageAll(eplus, weights, index=index)
        results.loc[index, 'flag']  += np.power(2, i)
        results.loc[index, ['numTests1', 'numTests2']] = numTests.loc[index].values
        results.loc[index, ['weights1', 'weights2']]   = weights.loc[index].values
        results.loc[index, 'numTests'] = weightedAverageAll(numTests, weights, index=index)
    


