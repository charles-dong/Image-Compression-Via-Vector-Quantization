function [prob_c, free_energy_e, likelihood_e] = q5_GM_Expectation(X, mus, sigmas, priors)
% Executes the Expectation-step for the learning of a GMM.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  mus: [n x K] matrix containing the n-dimensional means of the K Gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                           covariance matrix of the i-th Gaussian
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.
%
% OUTPUT:
%  prob_c: [K x m] matrix, containing the posterior probabilities over the K Gaussians for the m examples.
%          Specifically, prob_c(j, i) represents the probability that the
%          i-th example belongs to the j-th Gaussian, 
%          i.e., P(z^(i) = j | X^(i,:))
%  free_energy_e: [1 x 1] scalar value representing the free energy value
%  likelihood_e: [1 x 1] scalar value representing the log-likelihood value

[m, n] = size(X);
K = length(priors);
prob_c = zeros(K,m);

for i = 1 : m
    % prob_c(j,i) = P(z^(i)=j | X^(i,:)) = P(Xi | zi = j)P(zi = j) / sum
        % from l = 1 to K of P(xi | zi = l)P(zi = l)
    for j = 1 : K %gets numerator
        prob_c(j,i) = (exp(q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j))).*priors(j));
    end
    
    %divides by denominator
    prob_c(:,i) = prob_c(:,i)./sum(prob_c(:,i));
end

%calc free energy
free_energy_e = 0;
for i = 1 : m
    for j = 1 : K
         if ( prob_c(j,i) ~= 0)
%             free_energy_e = free_energy_e + prob_c(j,i).*log( ...
%                 exp(q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j))).*priors(j)./prob_c(j,i) );

         free_energy_e = free_energy_e + prob_c(j,i).*( ...
             q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j)) + log(priors(j)) ...
             - log(prob_c(j,i)));
        end
    end
end

%calc log lik
likelihood_e = 0;
for i = 1 : m
    temp = 0;
    for j = 1 : K
        temp = temp + exp(q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j))).*priors(j);
    end
    likelihood_e = likelihood_e + log(temp);
end


end