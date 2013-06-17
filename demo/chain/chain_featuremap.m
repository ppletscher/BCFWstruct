function phi = chain_featuremap(param, x, y)
% phi = chain_featuremap(param, x, y)
% computes the joint feature map phi(x,y). [param is ignored]
% 
% It is important that the implementation is consistent with the chain_oracle.m!
% The model is a standard chain graph, with discrete unaries and pairwise features. Additionally we
% include unary bias terms for the start and end of the sequence.

num_dims = size(x.data,1);
num_vars = size(x.data,2);
num_edges = num_vars-1;
num_states = x.num_states;

phi = zeros(num_states*num_dims+2*num_states+num_states^2, 1);

% unaries
for i=1:num_vars
    idx = y(i)*num_dims;
    phi((idx+1):(idx+1+num_dims-1)) = phi((idx+1):(idx+1+num_dims-1)) + x.data(:,i);
end
phi(num_states*num_dims+y(1)+1) = 1; % bias for first letter
phi(num_states*num_dims+num_states+y(end)+1) = 1; % bias for last letter

% pairwise
offset = num_states*num_dims+2*num_states;
for i=1:(num_vars-1)
    idx = y(i)+num_states*y(i+1);
    phi(offset+idx+1) = phi(offset+idx+1) + 1;
end
