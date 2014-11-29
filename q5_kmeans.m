function [labels, means, distortions] = q5_kmeans(X, K, seeds_idx)
% Executes Kmeans clustering algorithm, using euclidean distances.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of centroids (i.e. hyperparameter "K" in K-means)
%  seeds_idx: [1 x K] vector, containing the indices of the examples that 
%                     will be used as initial centroids.
% 
% OUTPUT:
%  labels: [m x 1] vector, containing the labels that the K-means algorithm assigned to the examples.
%                  labels(i) is an element of {1 ... K}, and it indicates the cluster ID associated to the i-th example
%  means: [n x K] matrix, containing the n-dimensional centroids of the K clusters.
%  distortions: [1 x num_iterations] vector, each element containing the total distortion at a particular iteration, i.e.
%                                    the sum of the squared Euclidean distances between the examples
%                                    and their associated centroids.
[m,n] = size(X);
num_iterations = 0;
centroids = X(seeds_idx,:);
labels = zeros(m,1);
distortions = [];

%while we've done more than 1 iteration and diff b/w consecutive
%distortions < 10^-6
while (num_iterations == 0 || num_iterations == 1 || abs(distortions(num_iterations) - distortions(num_iterations - 1)) > .000001)
    
    %calc dist b/w examples and centroids
    D = q5_dist2(X, centroids);
    
    %for each example, find closest centroid and update example's label
        %then calc distortion
    curr_distort = 0;
    for example = 1 : m
        
        [C,I] = min(D(example,:)); %I is index of smallest number 
        labels(example) = I; %update label
        curr_distort = curr_distort + (X(example,:) - centroids(I,:))*(X(example,:) - centroids(I,:))';
    end
    
    %assign distortion
    distortions = [distortions curr_distort];
    
    %update centroids by using new labels
    centroidCounters = zeros(m,1);
    centroids = zeros(K, n);
    for i = 1 : m %for each example in X
        %add X to its appropriate centroid and increment that centroid's
            %counter
        centroids(labels(i),:) = centroids(labels(i),:) + X(i,:);
        centroidCounters(labels(i)) = centroidCounters(labels(i)) + 1;
        
    end
    for j = 1 : K %for each centroid
        if (centroidCounters(j) ~= 0)
            %divide by the total number of examples it contains
           centroids(j,:) = centroids(j,:)./centroidCounters(j); 
        end
    end
    num_iterations = num_iterations + 1;
end

means = centroids';

end
