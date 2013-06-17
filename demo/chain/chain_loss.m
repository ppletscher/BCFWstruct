function loss = chain_loss(param, ytruth, ypredict)
% loss = chain_loss(param, ytruth, ypredict)
% Returns the normalized Hamming distance of predicted label ypredict to true
% label ytruth. param is ignored (just there for standard interface).
% 
% It is important that it is consistent with the loss function used in the
% loss-augmented decoding function (chain_oracle.m)!

loss = sum(ypredict~=ytruth) / numel(ytruth);
