function q5b()
% This script requires the following functions to be implemented:
% q5_dist2
% q5_kmeans_select_seeds
% q5_kmeans
% q5_reconstructimgfromVQ
% q5_gmminit
% q5_logprobgauss
% q5_GM_Expectation
% q5_GM_Maximization
% q5_GaussianMixture

% Make sure you have followed the given directions (Note: do not remove this line of code)
assert(checking(mfilename)==0);

% read and visualize the image
I = double(rgb2gray(imread('dartmouthhall2.jpg')));
figure(2);
subplot(1,2,1);
imshow(uint8(I));
title('original image');

% split the image into tiles
tilesize = 8;
[X, num_x_tiles, num_y_tiles] =  q5_splitimgintiles(I, tilesize);

% execute Kmeans
K = 4;
seeds_idx = q5_kmeans_select_seeds(X, K, 'diverse_set');
[tileidx, prototypes, distortions] = q5_kmeans(X, K, seeds_idx);

% reconstruct the image
recI_kmeans = q5_reconstructimgfromVQ(prototypes, tilesize, tileidx, num_x_tiles, num_y_tiles);
ssd = sum((I(:)-recI_kmeans(:)).^2);
fprintf('SSD using K-means: %e\n', ssd);

% initialize the GMM
[mus_init, sigmas_init, priors_init] = q5_gmminit(X, K, tileidx);

% train the GMM
num_iterations = 10;
[mus, sigmas, priors, likelihood_e, free_energy_e, likelihood_m, free_energy_m ] = ...
                          q5_GaussianMixture(X, mus_init, sigmas_init, priors_init, num_iterations);

% calculate the posteriors of the examples, given the trained GMM model
[postprob, ~, ~] = q5_GM_Expectation(X, mus, sigmas, priors);

% calculate to which gaussians the tiles belong to, and reconstruct the image
[junk, labels] = max(postprob);
recI_GMM = q5_reconstructimgfromVQ(mus, tilesize, labels', num_x_tiles, num_y_tiles);
ssd_GMM = sum((I(:)-recI_GMM(:)).^2);
fprintf('SSD using GMM: %e\n', ssd_GMM);

% visualize the reconstructed image
figure(2);
subplot(1,2,2);
imshow(uint8(recI_GMM));
title(['ssd GMM = ' sprintf('%e',ssd_GMM)]);
% save the plot (Note: do not remove this line of code)
saveas(gcf, 'q5b.fig');

% visualize the Expectation plots
figure(3);
hold on;
plot(likelihood_e, '-*b', 'LineWidth', 2,'MarkerSize', 5);
plot(free_energy_e, ':sr', 'LineWidth', 2,'MarkerSize', 10);
legend('log likelihood', 'free energy');
xlabel( 'iteration' );
title('After E step');
saveas(gcf, 'q5b_estep.fig');

% visualize the Maximization plots
figure(4);
hold on;
plot(likelihood_m, '-*b', 'LineWidth', 2,'MarkerSize', 5);
plot(free_energy_m, ':sr', 'LineWidth', 2,'MarkerSize', 10);
legend('log likelihood', 'free energy');
xlabel(' iteration');
title('After M step');
saveas(gcf, 'q5b_mstep.fig');




end