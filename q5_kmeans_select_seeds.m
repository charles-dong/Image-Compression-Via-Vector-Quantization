function seeds_idx = q5_kmeans_select_seeds(X, K, mode)
% Returns an initial set of centroids (i.e. a set of seeds) for the Kmeans algorithm. 
%
% INPUT:
%  X: [m x n] matrix, where each row is a d-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of centroids (i.e. hyperparameter "K" in K-means)
%  mode: string, indicating the type of initilization. It can be either 'random' or 'diverse_set'.
% 
% OUTPUT:
%  seeds_idx: [1 x K] vector, containing the indices of the examples that 
%                     will be used as initial centroids; seeds_idx(i)
%                     should be an integer number between 1 and m.

X = X';
[n, m] = size(X);
if strcmp(mode, 'random')
    % random initialization
    seeds_idx = randperm(m);
    seeds_idx = seeds_idx(1:K);
elseif strcmp(mode, 'diverse_set')
    % WRITE YOUR CODE HERE
    X = X';
   
    seeds_idx = [1];
    
    for k = 1 : K-1
%         prev_max = 0;
%         max_idx = 0;
        D = q5_dist2(X(seeds_idx(1,1:k),:), X);
         
        for example = 1 : m
            %totalDistance = 0;

            %sum distance
%             seeds_idx_size = size(seeds_idx,2);
%             for seed = 1 : seeds_idx_size
%                totalDistance = totalDistance + D(example,seeds_idx(seed));
%             end
            
            %if totalDistance of current example > prev_max,
                %make current example new max_idx
%             if (totalDistance > prev_max && isempty(find(seeds_idx == example)))
%                max_idx = example; 
%                prev_max = totalDistance;
%             end
            
            if ( k == 1 ) 
                [~, idx] = max(D);
                seeds_idx(1,k+1) = idx;
            else 
                [temp, ~] = min(D);
                [~,idx] = max(temp);
                seeds_idx(1,k+1) = idx;
                
            end
            
            

            
        end
        
    end
 
    
else
  error('parameter mode not recognized');
end

end