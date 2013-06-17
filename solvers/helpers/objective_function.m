function [ f ] = objective_function( w, b_alpha, lambda )
%OBJECTIVE_FUNCTION returns the function value f(alpha)
%   returns the SVM dual objective function  f(alpha) , which is
%   -equation (4) in the paper.
%   Here we compute this value but here only using the corresponding primal
%   variable vector  w = A * alpha  and  b*alpha  as parameters.

    f = lambda/2 * (w'*w) - b_alpha;

end % objective_function

