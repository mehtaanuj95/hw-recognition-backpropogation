# hw-recognition-backpropogation

### Pre-requisite

The Neural network has 3 layers. One input, one hidden and one output layer.

Since the images are of size 20*20, this gives us 400 input layer units ( plus one extra bias unit which always output +1 ).
The hidden layer contains 25 units.
The output layer contains 10 layers. This corrosponds to 10 digit classes.

weights.mat contains a set of network parameters (theta1, theta2). These are already trained.
```
Theta1 has size 25 x 401
Theta2 has size 10 x 26
```

**Note1:** This statement ```nn_params = [Theta1(:) ; Theta2(:)];``` creates a vector with dimentions **10285x1**. This is because Theta1(:) gives 10025x1 and Theta2(:) gives 260x1 and this when combined gives 10285x1.

**Note2:** We are randomly initializing the weights before training neural network for symmetry breaking.