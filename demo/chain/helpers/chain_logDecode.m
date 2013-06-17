function [y] = chain_logDecode(logNodePot,logEdgePot)

[nNodes,nStates] = size(logNodePot);

% Forward Pass
alpha = zeros(nNodes,nStates);
alpha(1,:) = logNodePot(1,:);
for n = 2:nNodes % Forward Pass
	tmp = repmat(alpha(n-1,:)',1,nStates) + logEdgePot;
	alpha(n,:) = logNodePot(n,:) + max(tmp);
	[dummy, mxState(n,:)] = max(tmp);
end

% Backward Pass
y = zeros(nNodes,1);
[dummy, y(nNodes)] = max(alpha(nNodes,:));
for n = nNodes-1:-1:1
	y(n) = mxState(n+1,y(n+1));
end
