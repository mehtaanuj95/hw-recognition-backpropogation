function [J grad] = costfunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, Theta1, Theta2)

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

%% Back-Propogation Algorithm

for t = 1:m

	% For the input layer, where l=1:
	a1 = [1; X(t,:)'];

	% For the hidden layers, where l=2:
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	yy = ([1:num_labels]==y(t))';
	% For the delta values:
	delta_3 = a3 - yy;

	delta_2 = (Theta2' * delta_3) .* [1; sigmoidgradient(z2)];
	delta_2 = delta_2(2:end); % Taking of the bias row

	% delta_1 is not calculated because we do not associate error with the input    

	% Big delta update
	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

end