function [Osig,err] = single_layer_test(inputs,labels, wh,wo)
%

x = inputs; 
t = labels;

% add bias term
x = [x'; 1*ones(1,size(labels,1))]; % add bias
x = x';                             % make nXnumFeatures

Osig = single_layer_output(x,wh,wo);

% compute error for the current iteration
err = errfun(Osig,t);



function err = errfun(O,t)
err = sum(1/2*(O-t).^2);

