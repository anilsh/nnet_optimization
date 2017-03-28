function [Osig,h1] = single_layer_output(x,wh,wo)

% Compute output of Ist layer
H1 = x * wh;
% Make inputs for next(output) layer
h1 = [sigmoid(H1), 1*ones(size(H1,1),1)];
% Compute output for output layer
O = h1 * wo;
% apply Activation function (sigmoid)
Osig = sigmoid(O);

% % Compute output of Ist layer
% H1 = x * wh;
% % Make inputs for next(output) layer
% h1 = [tanh(H1), 1*ones(size(H1,1),1)];
% % Compute output for output layer
% O = h1 * wo;
% % apply Activation function (sigmoid)
% Osig = tanh(O);