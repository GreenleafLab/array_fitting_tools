% use this function to return the fit parameters in a new file

function fitOffRateCurve(bindingCurveFilename, min_constraints, max_constraints, outputFitFilename, initial_points, scale_factor, fittype)
    %%
    % load binding curves
    load(bindingCurveFilename);
    
    % line by line fit with contraintsvz
    [numtottest, ~] = size(binding_curves);
    
    %% initiate fits
    params = ones(numtottest, 3)*nan;
    rmse = ones(numtottest, 1)*nan;
    exit_flag = ones(numtottest, 1)*nan;
    rsq = ones(numtottest, 1)*nan;
    params_var = ones(numtottest, 3)*nan;
    qvalue = ones(numtottest, 1)*nan;
    options = optimset('Display', 'off');
    
    % set default initial guess
    if ~exist('scale_factor', 'var');
        scale_factor = 1;
    end
    
    fmax_pos = 1;
    toff_pos = 2;
    fmin_pos = 3;
    
    min_fmin_upperbound = max_constraints(fmin_pos);
    min_fmax_upperbound = max_constraints(fmax_pos);

    %% cycle through each row and fit
    for i=1:numtottest;
        frac_bound = binding_curves(i,:);
        time = times(i, :);
        indx = find(~isnan(frac_bound));
        if strcmp(fittype, 'onrate');
            f = @CurveFitFun.findOnRate;
            initial_points(fmin_pos) = nanmin(frac_bound);
        else
            f = @CurveFitFun.findOffRate;
        end
        
        % fine tune initial parameters
        max_constraints(fmin_pos) = max(min(frac_bound)*2, min_fmin_upperbound);    % upper bound of fmin is either twice the minimum frac bund, or the pre-defined minimum of upperbound of fmin
        max_constraints(fmax_pos) = max(nanmax(frac_bound)*2, min_fmax_upperbound); % upperbound of fmax is either twice the maximum fracbound, or the predefined minimum of upperbound of fmax
        initial_points(fmax_pos) = nanmax(frac_bound);
 
        if length(indx) < 3 || ~isfinite(sum((f(initial_points, time(indx)) - frac_bound(indx)).^2));
            fprintf('Skipping iteration %d of %d\n', i, numtottest)
            continue
        else
            % get qvalue
            if strcmp(fittype, 'onrate');
                qvalue(i) = CurveFitFun.findFDR(binding_curves(i, end), null_scores);
            else
                qvalue(i) = CurveFitFun.findFDR(binding_curves(i, 1), null_scores);
            end
            % fit
            [x,fval,residual,exitflag,~, ~, jacobian] = lsqcurvefit(f, initial_points, time(indx), frac_bound(indx), min_constraints, max_constraints, options);
            
            % save parameters
            params(i, :) = x;

            % save fitting parameters
            rmse(i) = sqrt(fval/(length(indx)-length(x)));
            SSresid = fval;
            SStotal = sum((frac_bound(indx) - mean(frac_bound(indx))).^2);
            rsq(i) = 1 - SSresid/SStotal;
            exit_flag(i) = exitflag;
            [~,R] = qr(jacobian,0);
            mse = sum(abs(residual).^2)/(size(jacobian,1)-size(jacobian,2));
            Rinv = inv(R);
            Sigma = Rinv*Rinv'*mse;
            params_var(i, :) = full(sqrt(diag(Sigma)));
            if mod(i, 100)==0;
                fprintf('Completed iteration %d of %d\n', i, numtottest)
            end
        end
    end
    %final_to_save = [params, fit_successful, rsq, rmse];
    %dlmwrite(outputFitFilename, final_to_save, 'delimiter','\t','precision',6)
    save(outputFitFilename, 'params', 'exit_flag', 'rsq', 'rmse', 'qvalue', 'params_var')
end