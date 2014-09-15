import numpy as np
import scipy.stats as spst
from scipy.optimize import leastsq

class NLS:
    ''' This provides a wrapper for scipy.optimize.leastsq to get the relevant output for nonlinear least squares. Although scipy provides curve_fit for that reason, curve_fit only returns parameter estimates and covariances. This wrapper returns numerous statistics and diagnostics'''

    #import numpy as np
    #import scipy.stats as spst
    #from scipy import optimize

    def __init__(self, func, p0, xdata, ydata, parmNames):
        # Check the data
        if len(xdata) != len(ydata):
            msg = 'The number of observations does not match the number of rows for the predictors'
            raise ValueError(msg)

        ## Check parameter estimates
        #if type(p0) != dict:
        #    msg = "Initial parameter estimates (p0) must be a dictionry of form p0={'a':1, 'b':2, etc}"
        #    raise ValueError(msg)

        self.func = func
        self.inits = p0
        self.xdata = xdata
        self.ydata = ydata
        self.nobs = len( ydata )
        self.nparm= len( self.inits )

        self.parmNames = parmNames

        #for i in range( len(self.parmNames) ):
        #    if len(self.parmNames[i]) > 5:
        #        self.parmNames[i] = self.parmNames[i][0:4]

        # Run the model
        self.mod1 = leastsq(self.func, self.inits, args = (self.xdata, self.ydata), full_output=1)

        # Get the parameters
        #self.parmEsts = np.round( self.mod1[0], 4 )
        self.parmEsts = self.mod1[0]

        # Get the Error variance and standard deviation
        #self.residuals = self.mod1[2]['fvec']
        #self.RSS = np.sum( self.residuals**2 )

        self.RSS = np.sum( self.mod1[2]['fvec']**2 )
        self.df = self.nobs - self.nparm
        self.MSE = self.RSS / self.df
        self.RMSE = np.sqrt( self.MSE )
 
        # Get the covariance matrix
        if self.mod1[1] is  None:
            """
            mod1 will be none is singular matrix is encountered (i.e. very low variance)
            """
            self.jacobian_exists = False
        else:
            self.jacobian_exists = True
            self.cov = self.MSE * self.mod1[1]

            # Get parameter standard errors
            self.parmSE = np.diag( np.sqrt( self.cov ) )
    
            # Calculate the t-values
            self.tvals = self.parmEsts/self.parmSE
    
            # Get p-values
            self.pvals = (1 - spst.t.cdf( np.abs(self.tvals), self.df))*2
    
            # Get biased variance (MLE) and calculate log-likehood
            self.s2b = self.RSS / self.nobs
            self.logLik = -self.nobs/2 * np.log(2*np.pi) - self.nobs/2 * np.log(self.s2b) - 1/(2*self.s2b) * self.RSS
            
        del(self.mod1)
        #del(self.s2b)
        del(self.inits)

    # Get AIC. Add 1 to the df to account for estimation of standard error
    def AIC(self, k=2):
        return -2*self.logLik + k*(self.nparm + 1)

    #del(np)
    #del(optimize)

    # Print the summary
    def summary(self):
        print
        print 'Non-linear least squares'
        print 'Model: ' + self.func.func_name
        print 'Parameters:'
        print "      Estimate Std. Error t-value P(>|t|)"
        for i in range( len(self.parmNames) ):
                print "% -5s % 5.4f % 5.4f % 5.4f % 5.4f" % tuple( [self.parmNames[i], self.parmEsts[i], self.parmSE[i], self.tvals[i], self.pvals[i]] )
        print
        print 'Residual Standard Error: % 5.4f' % self.RMSE
        print 'Df: %i' % self.df

    def predictBindingCurve(self, concentration):
        concentration = np.array(concentration)
        predictions = self.parmEsts[0]*concentration/(exp(self.parmEsts[1])+concentration)
        return predictions
