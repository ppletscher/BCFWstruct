function [ primal ] = primal_objective( param, maxOracle, model, lambda )
% [ avg_loss ] = average_loss(param, maxOracle, model)
%
% Computes the primal SVM objective value for a given model.w on
% input data param.patterns
% primal = lambda/2 * ||w||^2 + 1/n \sum_i max_y (L_i(y) - w*psi_i(y))
% hinge_losses is an n x 1 vector containing the terms in the sum over i
%
% This function is not used by our BCFW solver, but can be used to monitor
% progress of primal solvers, such a the srochastic subgradient method (SSG)
%
% This function is expensive, as it requires a full decoding pass over all
% examples (so it costs as much as n BCFW iterations).

    patterns = param.patterns;
    labels = param.labels;
    phi = param.featureFn;
    loss = param.lossFn;

    hinge_losses = 0;
    for i=1:numel(patterns)
        % solve the loss-augmented inference for point i
        ystar_i = maxOracle(param, model, patterns{i}, labels{i});
        
        loss_i = loss(param, labels{i}, ystar_i);
        
        % feature map difference
        psi_i =   phi(param, patterns{i}, labels{i}) ...
                - phi(param, patterns{i}, ystar_i);
            
        % hinge loss for point i
        hinge_loss_i = loss_i - model.w'*psi_i;
        assert(hinge_loss_i >= 0);
        
        hinge_losses = hinge_losses + hinge_loss_i;
    end
    primal = lambda/2*(model.w'*model.w) + hinge_losses/numel(patterns);
    
end % primal_objective
