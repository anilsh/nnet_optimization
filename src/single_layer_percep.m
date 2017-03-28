function [wh, wo] = single_layer_percep(inputs, labels, eta, itr, error, neurons)
% Single Layer Neural Network with 1 hidden layer and 1 output layer

wh = 1/sqrt(size(inputs,2))*ones(size(inputs,2)+1, neurons(1))/1000; % input is along with bias
wo = 1/sqrt(neurons(1))*ones(neurons(1)+1, neurons(2))/1000;

e = [];
E = 10;
% iterate for 'itr' iterations
for it = 1:itr
    disp(['Epoch: ' num2str(it) '  Error: ' num2str(E)]);
    % check whether min error is reached 
    if(E < error)
        return;
    end
    
    % add bias term
    numI = floor(length(labels)/10);
    idx = randi(length(labels), numI,1);
    x = inputs(idx,:); 
    t = labels(idx); 
    x = [x'; ones(1,length(t))];   % add bias
    x = x';                        % make n X numFeatures

    % Compute output of Ist layer
    [Osig,h1] = single_layer_output(x,wh,wo);
    
    % find delta w's(change in weights) for both output & hidden layer
    del = errder(Osig,t).*errder_sigmoid(Osig); %[(t1-O1) (t2-O2)]';
    delwo = eta * del' * h1;
    delh = errder_sigmoid(h1(:,1:end-1)) .* (del * wo(1:end-1, :)');
    delwh = [eta * delh' *  x]';

%     % find delta w's(change in weights) for both output & hidden layer
%     del = errder(Osig,t).*(1-Osig.^2); %[(t1-O1) (t2-O2)]';
%     delwo = eta * del' * h1;
%     delh = (1-h1(:,1:end-1).^2) .* (del * wo(1:end-1, :)');
%     delwh = [eta * delh' *  x]';

    % update weights for both hidden and output layer
    wo = wo - delwo';
    wh = wh - delwh;
    
    % compute error for the current iteration
    err = errfun(Osig,t);
    e = [e err]; 

    % find error in the iteration
    E = err;
    %plot(e);
    %pause(0.001); 
end

function sigx = sigmoid(x)
sigx = 1 ./ (1 + exp(-1*x));

function err = errfun(O,t)
err = sum(1/2*(O-t).^2);

function der = errder(O,t)
der = ((O-t));

function der = errder_sigmoid(x)
der = x.*(1-x);