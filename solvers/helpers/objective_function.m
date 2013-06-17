function [ f ] = objective_function( w, b_alpha, lambda )
%OBJECTIVE_FUNCTION returns the function value f(alpha)
%   returns the SVM dual objective function  f(alpha) , which is
%   -equation (4) in the paper.
%   The arguments are w = A*alpha and b_alpha = b'*alpha.

    f = lambda/2 * (w'*w) - b_alpha;

end % objective_function

