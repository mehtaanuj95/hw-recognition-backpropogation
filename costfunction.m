function [J grad] = costfunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, Theta1, Theta2)

m = size(X, 1);
J = 0;

%% calculating the sigmoid
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
hThetaX = a3;

%% Mapping values from y to yVec for each training set.
yVec = zeros(m,num_labels);
for i = 1:m
    yVec(i,y(i)) = 1;
end

%% Cost function for neural network
J = 1/m * sum(sum(-1 * yVec .* log(hThetaX)-(1-yVec) .* log(1-hThetaX)));

%% Regularization
regularator = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

%% Total cost = cost + regularization
J = J + regularator;