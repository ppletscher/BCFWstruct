function [model, progress] = solverSSG(param, options)
% [model, progress] = solverSSG(param, options)
%
% Solves the structured support vector machine (SVM) using stochastic 
% subgradient descent (=SGD,=Pegasos) on the SVM primal problem.
% The code here follows a similar notation as in the paper (Lacoste-Julien,
% Jaggi, Schmidt, Pletscher; ICML 2013).
% Each step of the method calls the decoding oracle (i.e. the 
% loss-augmented predicion) only for a single point.
%
% The structured SVM has the form
% min_{w} 0.5*\lambda*||w||^2+ 1/n*\sum_{i=1}^n H_i(w) [see (3) in paper]
%   where H_i(w) is the structured hinge loss on the i^th example:
%         H_i(w) = max_{y in Y} L(y_i,y) - <w, \psi_i(y)> [(2) in paper]
%
% We use a calling interface very similar to version 1.1 of svm-struct-matlab 
% developped by Andrea Vedaldi (see vedaldi/code/svm-struct-matlab.html).
% svm-struct-matlab is a Matlab wrapper interface to the widely used
% SVM^struct code by Thorsten Joachims (http://svmlight.joachims.org/svm_struct.html) 
% which implements a cutting plane algorithm. 
% 
% If your code was using:
%    model = svm_struct_learn(command_args,param)
%
% You can replace it with:
%    model = solverSSG(param, options)
% 
% with the same param structure and letting options.lambda = 1/C
% to solve the *same* optimization problem. [Make sure your code only 
% uses model.w as we currently don't return the dual variables, etc. in the
% model structure unlike svm-struct-learn].
% 
% Inputs:
%   param: a structure describing the problem with the following fields:
%
%     patterns  -- patterns (x_i)
%         A cell array of patterns (x_i). The entries can have any
%         nature (they can just be indexes of the actual data for
%         example).
%     
%     labels    -- labels (y_i)
%         A cell array of labels (y_i). The entries can have any nature.
%
%     lossFn    -- loss function callback
%         A handle to the loss function L(ytruth, ypredict) defined for 
%         your problem. This function should have a signature of the form:
%           scalar_output = loss(param, ytruth, ypredict) 
%         It will be given an input ytruth, a ground truth label;
%         ypredict, a prediction label; and param, the same structure 
%         passed to solverSSG.
% 
%     oracleFn  -- loss-augmented decoding callback
%         [Can also be called constraintFn for backward compatibility with
%          code using svm_struct_learn.]
%         A handle to the 'maximization oracle' (equation (2) in paper) 
%         which solves the loss-augmented decoding problem. This function
%         should have a signature of the form:
%           ypredict = decode(param, model, x, y)
%         where x is an input pattern, y is its ground truth label,
%         param is the input param structure to solverSSG and model is the
%         current model structure (the main field is model.w which contains
%         the parameter vector).
%
%     featureFn  feature map callback
%         A handle to the feature map function \phi(x,y). This function
%         should have a signature of the form:
%           phi_vector = feature(param, x, y)
%         where x is an input pattern, y is an input label, and param 
%         is the usual input param structure. The output should be a vector 
%         of *fixed* dimension d which is the same
%         across all calls to the function. The parameter vector w will
%         have the same dimension as this feature map. In our current
%         implementation, w is sparse if phi_vector is sparse.
% 
%  options:    (an optional) structure with some of the following fields to
%              customize the behavior of the optimization algorithm:
% 
%   lambda      The regularization constant (default: 1/n).
%   num_passes  Number of iterations (passes through the data) to run the 
%               algorithm. (default: 50)
%   debug       Boolean flag whether to track the primal objective etc.
%               (default: 0)
%   do_weighted_averaging
%               Boolean flag whether to use weighted averaging of the iterates.
%               *Recommended -- it made a big difference in test error in
%               our experiments.*
%               (default: 1)
%   time_budget Number of minutes after which the algorithm should terminate.
%               Useful if the solver is run on a cluster with some runtime
%               limits. (default: inf)
%   rand_seed   Optional seed value for the random number generator.
%               (default: 1)
%   sample      Sampling strategy for example index, either a random permutation
%               ('perm') or uniform sampling ('uniform').
%               (default: 'uniform')
%   debug_multiplier
%               If set to 0, the algorithm computes the objective after each full
%               pass trough the data. If in (0,100) logging happens at a
%               geometrically increasing sequence of iterates, thus allowing for
%               within-iteration logging. The smaller the number, the more
%               costly the computations will be!
%               (default: 0)
%   test_data   Struct with two fields: patterns and labels, which are cell
%               arrays of the same form as the training data. If provided the
%               logging will also evaluate the test error.
%               (default: [])
%
% Outputs:
%   model       model.w contains the parameters;
%   progress    Primal objective, duality gap etc as the algorithm progresses,
%               can be used to visualize the convergence.
%
% Authors: Simon Lacoste-Julien, Martin Jaggi, Mark Schmidt, Patrick Pletscher
% Web: https://github.com/ppletscher/BCFWstruct
%
% Relevant Publication:
%       S. Lacoste-Julien, M. Jaggi, M. Schmidt, P. Pletscher,
%       Block-Coordinate Frank-Wolfe Optimization for Structural SVMs,
%       International Conference on Machine Learning, 2013.

% == geting the problem description:
phi = param.featureFn; % for \phi(x,y) feature mapping

if isfield(param, 'constraintFn')
    % for backward compatibility with svm-struct-learn
    maxOracle = param.constraintFn;
else
    maxOracle = param.oracleFn; % loss-augmented decoding function
end

patterns = param.patterns; % {x_i} cell array
labels = param.labels; % {y_i} cell array
n = length(patterns); % number of training examples

% == parse the options
options_default = defaultOptions(n);
if (nargin >= 2)
    options = processOptions(options, options_default);
else
    options = options_default;
end

% general initializations
lambda = options.lambda;
phi1 = phi(param, patterns{1}, labels{1}); % use first example to determine dimension
d = length(phi1); % dimension of feature mapping
using_sparse_features = issparse(phi1);
progress = [];

% === Initialization ===
% set w to zero vector
if using_sparse_features
    model.w = sparse(d,1);
else
    model.w = zeros(d,1);
end

if (options.do_weighted_averaging)
    wAvg = model.w; % called \bar w in the paper -- contains weighted average of iterates
end

% logging
if (options.debug_multiplier == 0)
    debug_iter = n;
    options.debug_multiplier = 100;
else
    debug_iter = 1;
end
progress.primal = [];
progress.eff_pass = [];
progress.train_error = [];
if (isstruct(options.test_data) && isfield(options.test_data, 'patterns'))
    progress.test_error = [];
end

fprintf('running SSG on %d examples. The options are as follows:\n', length(patterns));
options

rand('state',options.rand_seed);
randn('state',options.rand_seed);
tic();


% === Main loop ====
k=0; % same k as in paper
for p=1:options.num_passes

    perm = [];
    if (isequal(options.sample, 'perm'))
        perm = randperm(n);
    end

    for dummy = 1:n
        % 1) Picking random example:
        if (isequal(options.sample, 'uniform'))
            i = randi(n); % uniform sampling
        else
            i = perm(dummy); % random permutation
        end
    
        % 2) solve the loss-augmented inference for point i
        ystar_i = maxOracle(param, model, patterns{i}, labels{i});
                
        % 3) get the subgradient
        % ***
        % [the non-standard notation below is by analogy to the BCFW
        % algorithm -- but you can convince yourself that we are just doing
        % the standard subgradient update:
        %    w_(k+1) = w_k - stepsize*(\lambda*w_k + 1/n psi_i(ystar_i))
        % with stepsize = 1/(\lambda*(k+1))
        % ***
        %
        % [note that lambda*w_s is subgradient of 1/n*H_i(w) ]
        % psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
        psi_i = phi(param, patterns{i}, labels{i})-phi(param, patterns{i}, ystar_i);
        w_s = 1/(n*lambda) * psi_i;
        
        % 4) step-size gamma:
        gamma = 1/(k+1);
        
        % 5) finally update the weights
        model.w = (1-gamma)*model.w + gamma*n * w_s; % stochastic subgradient update (notice the factor n here)
                    
        % 6) Optionally, update the weighted average:
        if (options.do_weighted_averaging)
            rho = 2/(k+2); % resuls in each iterate w^(k) weighted proportional to k
            wAvg = (1-rho)*wAvg + rho*model.w;
        end
        
        k=k+1;
        
        % debug: compute objective and duality gap. do not use this flag for
        % timing the optimization, since it is very costly!
        if (options.debug && k == debug_iter)
            if (options.do_weighted_averaging)
                model_debug.w = wAvg;
            else
                model_debug.w = model.w;
            end
            primal = primal_objective(param, maxOracle, model_debug, lambda);
            train_error = average_loss(param, maxOracle, model_debug);
            fprintf('pass %d (iteration %d), SVM primal = %f, train_error = %f \n', ...
                             p, k, primal, train_error);

            progress.primal = [progress.primal; primal];
            progress.eff_pass = [progress.eff_pass; k/n];
            progress.train_error = [progress.train_error; train_error];
            if (isstruct(options.test_data) && isfield(options.test_data, 'patterns'))
                param_debug = param;
                param_debug.patterns = options.test_data.patterns;
                param_debug.labels = options.test_data.labels;
                test_error = average_loss(param_debug, maxOracle, model_debug);
                progress.test_error = [progress.test_error; test_error];
            end

            debug_iter = min(debug_iter+n,ceil(debug_iter*(1+options.debug_multiplier/100))); 
        end

        % time-budget exceeded?
        t_elapsed = toc();
        if (t_elapsed/60 > options.time_budget)
            fprintf('time budget exceeded.\n');
            if (options.do_weighted_averaging)
                model.w = wAvg; % return the averaged version
            end
            return
        end
    end
end

if (options.do_weighted_averaging)
    model.w = wAvg; % return the averaged version
end

end % solverSSG


function options = defaultOptions(n)

options = [];
options.num_passes = 50;
options.do_weighted_averaging = 1;
options.time_budget = inf;
options.debug = 0;
options.rand_seed = 1;
options.sample = 'uniform'; % sampling strategy in {'uniform', 'perm'}
options.debug_multiplier = 0; % 0 corresponds to logging after each full pass
options.lambda = 1/n;
options.test_data = [];

end % defaultOptions
