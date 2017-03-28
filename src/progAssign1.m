function progAssign1

load('train_set_mnist_69_100feats','features','labels');
% load('train_set_mnist_25_100feats','features','labels');
% load('train_set_mnist_35_150feats','features','labels');
% load('train_set_mnist_01','features','labels');

%% train neural network
eta = 0.02;
itr = 1000; 
error = 1e-10; 
neurons = [15 1];
[features, mu,stddev] = featureNormalize(features(:,1:100));
[wh, wo] = single_layer_percep(features, labels, eta, itr, error, neurons);
% save('model_1layer_100ShapeFeatsTANH_35_5kitr','wh','wo');
 
clearvars -except wh wo wh1 wh2 wo 
load('test_set_mnist_69_100feats','features','labels');
% load('test_set_mnist_25_100feats','features','labels');
% load('test_set_mnist_35_150feats','features','labels');
% load('test_set_mnist_01','features','labels');

% testing
[features, mu,stddev] = featureNormalize(features(:,1:100));
[Osig,err] = single_layer_test(features,labels, wh,wo);

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

