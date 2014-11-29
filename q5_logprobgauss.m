function logprob = q5_logprobgauss(x, mu, sigma)
% Calculates the log-probability density value for the input example under a given multivariate Gaussian,
% i.e. log(P(x ; mu, sigma))
% 
% INPUT:
%  x: [1 x n] vector, representing an input example
%  mu: [n x 1] vector representing the mean of a Gaussian
%  sigma: [n x n] covariance matrix for the Gaussian
%
% OUTPUT:
%  logprob: [1 x 1] scalar value representing the log of the probability density value

n = length(x);

logprob = (-n/2).*log(2*pi) - (1/2).*log(det(sigma)) ...
    - (1/2).*(x - mu')*inv(sigma)*(x'-mu);

end