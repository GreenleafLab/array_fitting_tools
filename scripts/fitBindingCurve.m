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
    exit_flag = ones(numtottest, 1)*nan;
    rsq = ones(numtottest, 1)*nan;
    params_var = ones(numtottest, 3)*nan;
    qvalue = ones(numtottest, 1)*nan;
    options = optimset('Display', 'off');
    
    % set default initial guess
    if ~exist('initial_points', 'var');
        initial_points = [0.5,400, 0];
    end
    
    fmax_pos = 1;
    dG_pos = 2;
    fmin_pos = 3;

    %% cycle through each row and fit
    for i=1:numtottest;
        frac_bound = binding_curves(i,:)./all_cluster(i);
        qvalue(i) = CurveFitFun.findFDR(binding_curves(i, end), null_scores);
        indx = find(~isnan(frac_bound));
        f = @CurveFitFun.findBindingCurve;
        
        % fine tune initial parameters
        max_constraints(fmin_pos) = min(min(frac_bound)*2, max_constraints(fmin_pos))
 
        if length(indx) < 3 || ~isfinite(sum((f(initial_points, concentrations(indx)) - frac_bound(indx)).^2));
            fprintf('Skipping iteration %d of %d', i, numtottest)
            continue
        else
            % fit
            [x,fval,residual,exitflag,~, ~, jacobian] = lsqcurvefit(f, initial_points, concentrations(indx), frac_bound(indx), min_constraints, max_constraints, options);
            
            % save parameters
            params(i, :) = x;

            % save fitting parameters
            rmse(i) = sqrt(fval/(length(indx)-length(x)));
            SSresid = fval;
            SStotal = sum((frac_bound(indx) - mean(frac_bound(indx))).^2);
            rsq(i) = 1 - SSresid/SStotal;
            exit_flag(i) = exitflag
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