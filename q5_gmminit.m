function [mus, sigmas, priors] = q5_gmminit(X, K, labels)
% Initializes a GMM model, given an initial clustering.

% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of gaussians for the GMM
%  labels: [m x 1] vector, containing the labels that the Kmeans algorithm assigned to the data.
%                  Each label l is an element of {1 ... K}, and it is associated with 
%                  the l-th gaussian.
% 
% OUTPUT:
%  mus: [n x K] matrix containing the n-dimensional means of the K gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                      covariance matrix for the i-th Gaussian
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.

[m,n] = size(X);

%get mus by using the hard labels
muCounters = zeros(m,1);
mus = zeros(K, n);
for i = 1 : m %for each example in X
    %add X to its appropriate mu and increment that mu's counter
    mus(labels(i),:) = mus(labels(i),:) + X(i,:);
    muCounters(labels(i)) = muCounters(labels(i)) + 1;
end
for j = 1 : K %for each Gaussian
    if (muCounters(j) ~= 0)
        %divide by the total number of examples it contains
       mus(j,:) = mus(j,:)./muCounters(j); 
    end
end
mus = mus';

%get priors
priors = zeros(1,K);
for i = 1 : m %get numerator
    priors(labels(i)) = priors(labels(i)) + 1;
end
priors = priors./m; %divide everything by m

%get sigmas
sigmas = zeros(n,n,K);
for i = 1 : m %get numerator
    sigmas(:,:,labels(i)) = sigmas(:,:,labels(i)) + ...
        (1).*(X(i,:)'-mus(:,labels(i)))*(X(i,:)'-mus(:,labels(i)))';
end
for j = 1 : K %divide each Gaussian's sigma by what we calc'ed in priors
    sigmas(:,:,j) = sigmas(:,:,j)./(priors(j).*m);
end







end

