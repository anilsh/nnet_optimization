% 2 Hidden Layer Perceptron

function [wh1 wh2 wo] = two_layer_percep(inputs, outputs, eta, itr, error, neurons)

wh1 = 1/sqrt(size(inputs,2))*ones(size(inputs,2), neurons(1)); %- 0.5*rand(size(inputs,2), neurons(1));
wh2 = 1/sqrt(neurons(1))*ones(neurons(1)+1, neurons(2)); % - 0.5*rand(neurons(1)+1, neurons(2));
wo = 1/sqrt(neurons(2))*ones(neurons(2)+1, neurons(3)); % - 0.5*rand(neurons(2)+1, neurons(3));
% wh1 = [    0.7562    0.5259;
%     0.8844    0.0451;
%     0.8955    0.7966;
%     0.0781    0.3148;
%     0.01 0.01 ];
% wh2 = [    0.2355    0.1424;
%     0.2674    0.1939;
%     0.01 0.01];
% 
% wo =[    0.8334    0.7439;
%     0.7451    0.2659;
%      0.01 0.01];
% disp(wh1);disp(wh2);disp(wo);   
E = 10;
for it = 1:itr
    disp(['Epoch: ' num2str(it)]);
    if(E < error)
        return;
    end
    e = [];
    for i = 1:size(inputs,1)
        x = inputs(i, :); 
        x = x';
        t = outputs(i, :);
        
        H1 = wh1' * x; % hidden layer 1 output
        h1 = [H1; 1];
        %h1 = sigmoid(h1);  
        H2 = wh2' * h1; % hidden layer 2 output
        h2 = [H2; 1];
        %h2 = sigmoid(h2);
        O = wo' * h2;   % output layer values
%         O = tanh(O);
        O = (1 - exp(-1 * O)) ./ (1 + exp(-1 * O));
%         th = 0;
       
        odel = (O-t)^2/2;  % gradient at output layer
        
        e = [e odel]; %(t1 - O1)^2 + (t2-O2)^2 + (t3 - O3)^2 + (t4-O4)^2];  % error in this
        del = [O-t];
        delwo = eta * del' * h2';
        del2 = del * wo(1:end-1, :)';
        delwh2 = eta * del2' * h1';
        del1 = del2 * wh2(1:end-1, :)';
        delwh1 = eta * del1' * x';
        
        wo = wo + delwo';
        wh2 = wh2 + delwh2';
        wh1 = wh1 + delwh1';
    end    
    E = mean(e);
    wo(isnan(wo)) = 0.005;
    wh2(isnan(wh2)) = 0.005;
    wh1(isnan(wh1)) = 0.005;
end

end