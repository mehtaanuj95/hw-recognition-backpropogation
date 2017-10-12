%% Initialization
clear;
input_layer_size  = 400;  				% 20x20 Input Images of Digits
hidden_layer_size = 25;   				% 25 hidden units
num_labels = 10;          				% 10 labels, from 1 to 10   

%% Load and visualize data

load('data_train.mat'); 				% training data stored in arrays X, y
m = size(X, 1);

rand_indices = randperm(m); 			% Randomly select 100 data points to display
selected = X(rand_indices(1:100), :);
displayData(selected);

%% Loading parameters
load('weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];

%% Calculating cost of feed forward network
lambda = 1;
J = costfunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

%% sigmoid gradient function
%% The below statement only for debug purpose. value of g at 0 shoulb be 0.25.
g = sigmoidgradient(0);

%% Randomly initialize weights of layer with L_in incoming and L_out outgoing connections.
initial_Theta1 = randominitializeweight(input_layer_size, hidden_layer_size);
initial_Theta2 = randominitializeweight(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Main Backpropogation Algorithm (implemented in costfunction.m)

%% Training Neural network

options = optimset('MaxIter', 5);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costfunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Visualizing the neural network function (hidden layers)
displayData(Theta1(:, 2:end));
