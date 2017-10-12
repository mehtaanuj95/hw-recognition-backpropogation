function g = sigmoidgradient(z)
g = sigmoid(z) .* (1 - sigmoid(z));
end
