function [mus, sigmas, priors, free_energy_m, likelihood_m] = q5_GM_Maximization(X, prob_c)
% Executes Maximization-step for the learning of a GMM.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  prob_c: [K x m] matrix, containing the the posterior probabilities over the K Gaussians for the m examples.
%          Specifically, prob_c(j, i) represents the probability that the
%          i-th example belongs to the j-th Gaussian, 
%          i.e., P(z^(i) = j | X^(i,:))
%
% OUTPUT:
%  mus: [n x K] matrix containing the n-dimensional means of the K gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                           covariance matrix of the i-th Gaussian.
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.
%  free_energy_m: [1 x 1] scalar value representing the free energy value
%  likelihood_m: [1 x 1] scalar value representing the log-likelihood value

[m, n] = size(X);
K = size(prob_c,1);

%get priors
priors = zeros(1,K);
for j = 1 : K
        priors(j) = sum(prob_c(j,:))./m;
end

%get mus
mus = zeros(K, n);
for j = 1 : K %for each cluster
    for i = 1 : m
        %numerator
        mus(j,:) = mus(j,:) + prob_c(j,i).*X(i,:);
    end
    mus(j,:) = mus(j,:)./sum(prob_c(j,:)); %denom
end
mus = mus';

%get sigmas
sigmas = zeros(n,n,K);
for j = 1 : K %for each cluster
    for i = 1 : m
        %numerator
        sigmas(:,:,j) = sigmas(:,:,j) + prob_c(j,i).* ...
            (X(i,:)' - mus(:,j)) * (X(i,:)' - mus(:,j))';
    end
    sigmas(:,:,j) = sigmas(:,:,j)./sum(prob_c(j,:)); %denom
end


%calc free energy
free_energy_m = 0;
for i = 1 : m
    for j = 1 : K
         if ( prob_c(j,i) ~= 0)
%             free_energy_e = free_energy_e + prob_c(j,i).*log( ...
%                 exp(q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j))).*priors(j)./prob_c(j,i) );

         free_energy_m = free_energy_m + prob_c(j,i).*( ...
             q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j)) + log(priors(j)) ...
             - log(prob_c(j,i)));
        end
    end
end

%calc log lik
likelihood_m = 0;
for i = 1 : m
    temp = 0;
    for j = 1 : K
        temp = temp + exp(q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j))).*priors(j);
    end
    likelihood_m = likelihood_m + log(temp);
end

end