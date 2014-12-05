% Class for fitting binding curves with constraints

classdef FitFun
     methods (Static)
         
         % function to minimize to find kd. Least squares fit.
         function mse = findKd(params, concentrations, fraction_bound)
             mse = sum((fraction_bound - FitFun.findBindingCurve(concentrations, params(1), params(2), params(3))).^2);
         end
         
         % function to plot binding curve
         function fracbound = findBindingCurve(concentration, fmax, kd, fmin)
            fracbound = fmax*concentration./(concentration+kd)+fmin;
         end
         
         % function to find offrate
         function mse = findOffRate(params, concentrations, fraction_bound)
         end
     end
end