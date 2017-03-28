function [pLabel] = two_layer_test(inputs,outputs, wh1,wh2,wo)


pLabel = [];
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
    
    pLabel = [pLabel; O];
end    

