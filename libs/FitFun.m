% Class for fitting binding curves with constraints

classdef FitFun
     methods (Static)
         
         % function to minimize to find kd. Least squares fit.
         function mse = findKd(params, concentrations, fraction_bound)
             mse = sum((fraction_bound - concentrations*params(1)./(params(2)+concentrations)).^2);
         end
         
         % function to plot binding curve
         function fracbound = findBindingCurve(concentration, fmax, kd)
            fracbound = fmax*concentration./(concentration+kd);
         end
         
         % function to find offrate
         function mse = findOffRate(params, concentrations, fraction_bound)
         end
     end
end