function D = q5_dist2(X1, X2)
% Calculates the *squared* Euclidean distance between two sets of points.
% You should be able to efficiently implement this function *without* using for-loops.
%
% INPUT:
%  X1: [m1 x n] matrix, where each row is an n-dimensional input example
%  X2: [m2 x n] matrix, where each row is an n-dimensional input example
%  
% OUTPUT:
%  D: [m1 x m2] matrix, where the element D(i,j) represent the squared Euclidean distance between
%               the i-th example of X1, and the j-th example of X2.

[m1,n] = size(X1);
[m2,n] = size(X2);

A2 = X1.^2*ones(n,1);
A2 = A2*ones(1,m2);

AB = (X1*X2').*(-2);

B2 = X2.^2*ones(n,1);
B2 = ones(m1,1)*B2';

D = A2 + AB + B2;

end