function process_dataset

% training set
% img = loadMNISTImages('./train-images.idx3-ubyte');
% targets = loadMNISTLabels('./train-labels.idx1-ubyte');
load('train_set_full','img','targets')
[features,labels] = extract_pos(img,targets);
save('train_set_mnist_25_100feats','features','labels');

% testing set
% img = loadMNISTImages('./t10k-images.idx3-ubyte');
% targets = loadMNISTLabels('./t10k-labels.idx1-ubyte');
load('test_set_full','img','targets')
[features,labels] = extract_pos(img,targets);
save('test_set_mnist_25_100feats','features','labels');
disp('');

function [train_features,labels] = extract_pos(images,targets)
images = images';
c1 = 2;
c2 = 5;
samples = images(targets==c1 | targets==c2,:);
labels = targets(targets==c1 | targets==c2);
l = labels;
l(labels==c1) = 0;
l(labels==c2) = 1;
labels = l;

% [r,c] = find(samples);
% numfeat = 100;
% train_features = zeros(size(samples,1),numfeat);
% for i = 1:max(r)
%     val = c(r==i);
%     if(length(val) >= numfeat)
%         train_features(i,:) = val(1:numfeat);
%     else
%         train_features(i,:) = [val; zeros(numfeat-length(val),1)];
%     end
% end

% Extract shape features
numfeat = 100;
train_features = zeros(size(samples,1),numfeat);
for i = 1:size(samples,1)
    samp = samples(i,:);
    im = reshape(samp, 28,28);
    
    [r,c] = find(im);
    feat_one_sample = [r; c]';
    if(length(feat_one_sample) >= numfeat)
        train_features(i,:) = feat_one_sample(1:numfeat);
    else
        train_features(i,:) = [feat_one_sample; zeros(numfeat-length(feat_one_sample),1)];
    end
end

