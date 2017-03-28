function progAssign1_2HL

% % % load train images and targets
load('train_set_mnist_25_100feats','features','labels');

%% train neural network
eta = 0.002;
itr = 2000;
error = 1e-10; 
neurons = [50 30 1];
% [wh1 wh2 wo] = two_layer_percep(features, labels, eta, itr, error, neurons);
% load('model_2layer_100ShapeFeatsTANH_25','wh1','wh2','wo');
load('model_2layer_50ShapeFeats_01');
 
clearvars -except wh wo wh1 wh2 wo
% load test images and targets
load('test_set_mnist_25_100feats','features','labels');
% load('test_set_mnist_01');

% testing
[Osig] = two_layer_test(features,labels, wh1,wh2,wo);

disp(sum(labels==(Osig>0.5))/length(labels));
disp(sum((Osig>0.5))); %==labels));
disp('');


function [signals,PC,V] = pca2(data)
% PCA2: Perform PCA using SVD.
% data - MxN matrix of input data
% (M dimensions, N trials)
% signals - MxN matrix of projected data
% PC - each column is a PC

% V - Mx1 matrix of variances
[M,N] = size(data);
% subtract off the mean for each dimension
mn = mean(data,2);
data = data - repmat(mn,1,N);
% construct the matrix Y
Y = data' / sqrt(N-1);
% SVD does it all
[u,S,PC] = svd(Y);
% calculate the variances
S = diag(S);
V = S .* S;
% project the original data
signals = PC(:,1:100)' * data;

