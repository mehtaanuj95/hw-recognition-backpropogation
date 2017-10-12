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

