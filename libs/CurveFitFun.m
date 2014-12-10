% Class for fitting binding curves with constraints

classdef CurveFitFun
     methods (Static)
         
         % function to minimize to find kd. Least squares fit.
         function mse = findKd(params, concentrations, fraction_bound)
             mse = sum((fraction_bound - CurveFitFun.findBindingCurve(params, concentrations)).^2);
         end
         
         % function to plot binding curve
         function fracbound = findBindingCurve(x, concentration)
             fmax = x(1);
             dG   = x(2);
             fmin = x(3);
             fracbound = fmax*concentration./(concentration+exp(dG/0.582)/1e-9)+fmin;
         end
         
         % function to find qvalue
         function q = findFDR(score, null_scores)
             q = sum(abs(null_scores) > abs(score))/length(null_scores);
         end
             
         % function to find offrate
         function mse = findOffRate(params, concentrations, fraction_bound)
         end
     end
end