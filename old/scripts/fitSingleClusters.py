#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF               
sns.set_style('white', )

### MAIN ###

################ Parse input parameters ################

#set up command line argument parser
parser = argparse.ArgumentParser(description='fit single clusters to binding curve')
parser.add_argument('reducedCPseq', 
                    help='CPsignal file to fit')
parser.add_argument('-m', '--mapCPfluors',
                    help='map file giving the dir names to look for CPfluor files')
parser.add_argument('-n','--num_cores', type=int, default=1,
                    help='maximum number of cores to use. default=1')
parser.add_argument('-nc', '--null_column', type=int, default=-1,
                    help='point in binding series to use for null scores (Default is '
                    'last concentration. -2 for second to last concentration)', )

if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

# define functions
def main(reducedCPsignalFile, concentrations, null_column=None):
    print 'Fitting single cluster fits "%s"...'%fittedBindingFilename
    # get binding series
    print '\tLoading binding series and all RNA signal:'
    bindingSeries, allClusterSignal = IMlibs.loadBindingCurveFromCPsignal(reducedCPsignalFile, concentrations)
    
    # make normalized binding series
    IMlibs.boundFluorescence(allClusterSignal, plot=True)   # try to reduce noise by limiting how big/small you divide by
    bindingSeriesNorm = np.divide(bindingSeries, np.vstack(allClusterSignal))
    
    # find null scores and max signal
    fabs_green_max = bindingSeriesNorm.iloc[:, args.null_column]
    null_scores = IMlibs.loadNullScores(signalNamesByTileDict, filterPos=filterPos, filterNeg=filterNeg, binding_point=args.null_column)
    
    # get binding estimation and choose 10000 that pass filter
    ecdf = ECDF(pd.Series(null_scores).dropna())
    qvalues = pd.Series(1-ecdf(bindingSeries.iloc[:, args.null_column].dropna()), index=bindingSeries.iloc[:, args.null_column].dropna().index)
    qvalues.sort()
    index = qvalues.iloc[:1E4].index # take top 10K binders

    # fit first round
    print '\tFitting best binders with no constraints...'
    parameters = fittingParameters.Parameters(concentrations, fabs_green_max.loc[index])
    fitUnconstrained = IMlibs.splitAndFit(bindingSeriesNorm, 
                                          concentrations, parameters.fitParameters, numCores, index=index, mod_fmin=True)
    
    # reset fitting parameters based on results
    maxdG = parameters.find_dG_from_Kd(parameters.find_Kd_from_frac_bound_concentration(0.9, concentrations[args.null_column])) # 90% bound at 
    param = 'fmax'
    parameters.fitParameters.loc[:, param] = IMlibs.plotFitFmaxs(fitUnconstrained, maxdG=maxdG, param=param)
    plt.savefig(os.path.join(os.path.dirname(fittedBindingFilename), 'constrained_%s.pdf'%param))
    
    param = 'fmin'
    parameters.fitParameters.loc[:, param] = IMlibs.findProbableFmin(bindingSeriesNorm, qvalues)
    plt.savefig(os.path.join(os.path.dirname(fittedBindingFilename), 'constrained_%s.pdf'%param))
        
    # now refit all remaining clusters
    print 'Fitting all with constraints on fmax (%4.2f, %4.2f, %4.2f)'%(parameters.fitParameters.loc['lowerbound', 'fmax'], parameters.fitParameters.loc['initial', 'fmax'], parameters.fitParameters.loc['upperbound', 'fmax'])
    print 'Fitting all with constraints on fmin (%4.4f, %4.4f, %4.4f)'%(parameters.fitParameters.loc['lowerbound', 'fmin'], parameters.fitParameters.loc['initial', 'fmin'], parameters.fitParameters.loc['upperbound', 'fmin'])
    
    # save fit parameters
    fitParametersFilename = os.path.join(os.path.dirname(fittedBindingFilename),
                                         'bindingParameters.%s.fp'%datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    parameters.fitParameters.to_csv(fitParametersFilename, sep='\t')
    fitConstrained = pd.DataFrame(index=bindingSeriesNorm.index, columns=fitUnconstrained.columns)
    
    # sort by qvalue to try to get groups of equal distributions of binders/nonbinders
    index = pd.concat([bindingSeriesNorm, pd.DataFrame(qvalues, columns=['qvalue'])], axis=1).sort('qvalue').index
    index_all = bindingSeriesNorm.loc[index].dropna(axis=0, thresh=4).index
    fitConstrained.loc[index_all] = IMlibs.splitAndFit(bindingSeriesNorm, concentrations,
                                                       parameters.fitParameters, numCores, index=index_all)
    fitConstrained.loc[:, 'qvalue'] = qvalues
    
    return fitParameters

if __name__=="__main__":    
    args = parser.parse_args()

    tmp, tmp, concentrations = IMlibs.loadMapFile(args.mapCPfluors)



    # save fittedBindingFilename
    #fitParametersFilename = IMlibs.getFitParametersFilename(annotatedSignalFilename)
    #IMlibs.saveDataFrame(fitConstrained, fitParametersFilename, index=False, float_format='%4.3f')
    table = IMlibs.makeFittedCPsignalFile(fitConstrained,annotatedSignalFilename, fittedBindingFilename, bindingSeriesNorm, allClusterSignal)
