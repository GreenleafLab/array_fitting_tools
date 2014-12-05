% use this function to return the fivt parameters in a new file

function fitBindingCurve(bindingCurveFilename, min_constraints, max_constraints, outputFitFilename, initial_points)
    %%
    % load binding curves
    load(bindingCurveFilename);
    
    % line by line fit with contraints
    [numtottest, ~] = size(binding_curves);
    
    % initiate fits
    params = ones(numtottest, 3)*nan;
    rmse = ones(numtottest, 1)*nan;
    fit_successful = zeros(numtottest, 1);
    rsq = ones(numtottest, 1)*nan;
    
    % set default initial guess
    if ~exist('initial_points', 'var');
        initial_points = [0.5,400, 0];
    end
    
    %% cycle through each row and fit
    for i=1:numtottest;
        frac_bound = binding_curves(i,:);
        indx = find(~isnan(frac_bound));
        if length(indx) < 4 || ~isfinite(FitFun.findKd(initial_points,concentrations(indx),frac_bound(indx)));
            fprintf('Skipping iteration %d of %d', i, numtottest)
            continue
        else
            f = @(x)FitFun.findKd(x,concentrations(indx),frac_bound(indx));
            options = optimset('Algorithm', 'interior-point');
            [x,fval, exitflag,output,lambda, grad, hessian] = fmincon(f, initial_points, [], [], [], [],min_constraints, max_constraints, [], options);

            % save parameters
            params(i, 1) = x(1);
            params(i, 2) = x(2);
            params(i, 3) = x(3);

            % save fitting parameters
            rmse(i) = sqrt(fval/(length(indx)-length(x)));
            SSresid = fval;
            SStotal = sum((frac_bound(indx) - mean(frac_bound(indx))).^2);
            rsq(i) = 1 - SSresid/SStotal;
            if rsq(i) > 0 && exitflag==1;
                fit_successful(i) = 1;
            end
            fprintf('Completed iteration %d of %d', i, numtottest)
        end
    end
    %final_to_save = [params, fit_successful, rsq, rmse];
    %dlmwrite(outputFitFilename, final_to_save, 'delimiter','\t','precision',6)
    save(outputFitFilename, 'params', 'fit_successful', 'rsq', 'rmse' )
end