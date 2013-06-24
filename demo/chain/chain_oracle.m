function label = chain_oracle(param, model, xi, yi)
% do loss-augmented decoding on a given example (xi,yi) using
% model.w as parameter. Param is ignored (included for standard
% interface). The loss used is normalized Hamming loss.
% 
% If yi is not given, then standard prediction is done (i.e. MAP decoding
% without the loss term).

w = model.w;

% problem dimensions
num_dims = size(xi.data,1);
num_vars = size(xi.data,2);
num_states = xi.num_states;

% map current w to a cell array which is easier to deal with
weight = weightVec2Cell(w, num_states, num_dims);

% build score for the chain graph
if (issparse(w))
    theta_unary = zeros(num_states, num_vars);
    for i=1:num_vars
        idx_x = find(xi.data(:,i));
        for j=1:num_states
            offset = (j-1)*num_dims;
            theta_unary(j,i) = w(idx_x+offset)'*xi.data(idx_x,i);
        end
    end
else
    theta_unary = weight{1}'*xi.data;
end
theta_unary(:,1) = theta_unary(:,1) + weight{2}; % first position has a bias 
theta_unary(:,end) = theta_unary(:,end) + weight{3}; % last position has a bias
theta_pair = weight{4};

% add loss-augmentation to the score (normalized Hamming distance used for loss)
if nargin > 3
    L = numel(yi); % length of chain
    for i=1:num_vars
        theta_unary(:,i) = theta_unary(:,i) + 1/L;
        idx = yi(i)+1;
        theta_unary(idx, i) = theta_unary(idx,i) - 1/L;
    end
end

% solve inference problem
label = chain_logDecode(theta_unary', theta_pair);
label = label'-1;

end % chain_oracle


function weight = weightVec2Cell(w, num_states, d)

idx = (num_states*d);
if (issparse(w))
    weight{1} = [];
else
    weight{1} = reshape(w(1:idx), [d num_states]);
end
weight{2} = w((idx+1):(idx+num_states));
idx = idx+num_states;
weight{3} = w((idx+1):(idx+num_states));
idx = idx+num_states;
weight{4} = reshape(w((idx+1):end), [num_states num_states]);

end % weightVec2Cell
