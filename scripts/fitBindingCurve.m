% use this function to return the fit parameters in a new file

function fitBindingCurve(bindingCurveFilename, xvaluesFilename, min_constraints, max_constraints, outputFitFilename, initial_points)
    %%
    % load binding curves
    binding_curves = dlmread(bindingCurveFilename);
    
    % load concentrations
    concentrations = transpose(dlmread(xvaluesFilename));
    
    % line by line fit with contraints
    %numtottest = length(binding_curves);
    numtottest = 100;
    
    % initiate fits
    params = zeros(numtottest, 2);
    rmse = zeros(numtottest, 1);
    exitflags = zeros(numtottest, 1);
    
    % set default initial guess
    if ~exist('initial_points', 'var');
        initial_points = [0.5, 400];
    end
    
    %% cycle through each row and fit
    for i=1:numtottest;
        %%
        frac_bound = binding_curves(i,:);
        indx = find(~isnan(frac_bound));
        f = @(x)FitFun.findKd(x,concentrations(indx),frac_bound(indx));
        options = optimset('Algorithm', 'interior-point');
        [x,fval, exitflag,output,lambda, grad, hessian] = fmincon(f, initial_points, [], [], [], [],min_constraints, max_constraints, [], options);
        params(i, 1) = x(1);
        params(i, 2) = x(2);
        rmse(i) = sqrt(fval);
        exitflags(i) = exitflag;
    end
    final_to_save = [params, rmse, exitflags];
    dlmwrite(outputFitFilename, final_to_save, 'delimiter','\t','precision',6)
end