function [ avg_loss ] = average_loss( param, maxOracle, model )
% [ avg_loss ] = average_loss( param, maxOracle, model)
%
% Return the average loss for the predictions of model.w on
% input data param.patterns. See solverBCFW for interface of param.
%
% This function is expensive, as it requires a full decoding pass over all
% examples (so it costs as much as n BCFW iterations).

    patterns = param.patterns;
    labels = param.labels;
    loss = param.lossFn;

    loss_term = 0;
    for i=1:numel(patterns)
        ystar_i = maxOracle(param, model, patterns{i}); % standard decoding (not loss-augmented) as no input label
        loss_term = loss_term + loss(param, labels{i}, ystar_i);
    end
    avg_loss = loss_term / numel(patterns);
    
end % average_loss
