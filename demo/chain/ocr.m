% Applies the structured SVM to the OCR dataset by Ben Taskar. The structured
% model considered here is the standard chain graph, with the pixel values of
% the digit as unary features and a transition matrix of size num_states^2 as
% a pairwise potential. Additionally, we include a unary bias term for the first
% and last symbol in the sequence.

addpath(genpath('../../solvers/'));
addpath('helpers');

% We support two different settings for the dataset (ocr: only one fold in
% training set, ocr2: all but one fold in training set
% -- ocr2 is the one that we have used in our experiments in the 
% ICML 2013 paper)
data_name = 'ocr';
[patterns_train, labels_train, patterns_test, labels_test] = loadOCRData(data_name, '../../data/');

%% == run one of the solvers on the problem

% create problem structure:
param = [];
param.patterns = patterns_train;
param.labels = labels_train;
param.lossFn = @chain_loss;
param.oracleFn = @chain_oracle;
param.featureFn = @chain_featuremap;

% options structure:
options = [];
options.lambda = 1e-2;
options.num_passes = 10;
options.do_line_search = 1;
options.debug = 1;

%% run the solver
[model, progress] = solverBCFW(param, options);
%[model, progress] = solverFW(param, options);
%[model, progress] = solverSSG(param, options);

%% loss on train set
avg_loss = 0;
for i=1:numel(patterns_train)
    ypredict = chain_oracle(param, model, patterns_train{i}); % standard prediction as don't give label as input
    avg_loss = avg_loss + chain_loss(param, labels_train{i}, ypredict);
end
avg_loss = avg_loss / numel(patterns_train);
fprintf('average loss on the training set: %f.\n', avg_loss);

% loss on test set
avg_loss = 0;
for i=1:numel(patterns_test)
    ypredict = chain_oracle(param, model, patterns_test{i});
    avg_loss = avg_loss + chain_loss(param, labels_test{i}, ypredict);
end
avg_loss = avg_loss / numel(patterns_test);
fprintf('average loss on the test set: %f.\n', avg_loss);

% plot the progress of the solver
plot(progress.eff_pass, progress.primal, 'r-'); % primal
hold on;
plot(progress.eff_pass, progress.dual, 'b--'); % dual
hold off;
xlabel('effective passes');
