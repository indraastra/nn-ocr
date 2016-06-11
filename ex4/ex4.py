# Port of Coursera ML's ex4.m neural network runner.
import random

import numpy as np
import scipy.io as sio
import scipy.optimize as sopt

from display_data import display_data
from nn_cost_function import nn_cost_function
from sigmoid import sigmoid_gradient
from rand_initialize_weights import rand_initialize_weights
from utils import reshape_params, flatten_params
from predict import predict

## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('> Loading and Visualizing Data ...\n')

data = sio.loadmat('../data/fontdata_small.mat');
X = data['X']
# This correction is for going from labels in Octave to zero-indexed labels
# in python.
y = data['y'].flatten()
y[y == 10] = 0

m = X.shape[0]

# Randomly select 100 data points to display
sel = random.sample(range(m), 100)

display_data(X[sel, :]);

input('\nProgram paused. Press enter to continue.\n');

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('> Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weights = sio.loadmat('../data/ex4weights.mat');
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

# Unroll parameters 
nn_params = flatten_params(Theta1, Theta2)

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('> Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
regularization = 0

J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                        num_labels, X, y, regularization);

print('Cost at parameters (loaded from ex4weights): {} '
      '\n(this value should be about 0.287629)\n'.format(J));

input('Program paused. Press enter to continue.\n');

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('> Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
regularization = 1;

J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                        num_labels, X, y, regularization);

print('Cost at parameters (loaded from ex4weights): {} '
      '\n(this value should be about 0.383770)\n'.format(J));

input('Program paused. Press enter to continue.\n');

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('Evaluating sigmoid gradient...')

g = sigmoid_gradient(np.array([1,-0.5,0,0.5,1]));
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:');
print(g);
print();

input('Program paused. Press enter to continue.\n');


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('Initializing Neural Network Parameters ...\n')

initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size);
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels);

# Unroll parameters
initial_nn_params = flatten_params(initial_Theta1, initial_Theta2)


###
### NOTE: Skipping this and comparing with Octave results instead.
###
### =============== Part 7: Implement Backpropagation ===============
##  Once your cost matches up with ours, you should proceed to implement the
##  backpropagation algorithm for the neural network. You should add to the
##  code you've written in nnCostFunction.m to return the partial
##  derivatives of the parameters.
##
#print('Checking Backpropagation... \n');
#
##  Check gradients by running checkNNGradients
## checkNNGradients;
#
#print('\nProgram paused. Press enter to continue.\n');
#pause;
#

###
### NOTE: Adjusting a part of this.
###
# ## =============== Part 8: Implement Regularization ===============
# #  Once your backpropagation implementation is correct, you should now
# #  continue to implement the regularization with the cost and gradient.
# #
# 
print('\nChecking Backpropagation (w/ Regularization) ... \n')

# #  Check gradients by running checkNNGradients
regularization = 3;
# checkNNGradients(lambda);
# 
# Also output the costFunction debugging values
debug_J, debug_grad = nn_cost_function(nn_params, input_layer_size,
                                       hidden_layer_size, num_labels,
                                       X, y, regularization);

print('Cost and sum of gradients at (fixed) debugging parameters '
      '(w/ lambda = 10): {}, {} \n(these values should be about'
      'about 0.576051, -0.022227)\n'.format(debug_J, debug_grad.sum()));

input('Program paused. Press enter to continue.\n');


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('Training Neural Network...')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
opts = {
    'maxiter': 50,
    'disp': True
}
def callback(x):
    callback.iter += 1
    print("Iteration", callback.iter)
callback.iter = 0

#  You should also try different values of lambda
regularization = 1

# Create "short hand" for the cost function to be minimized
cost_function = lambda p: \
    nn_cost_function(p, input_layer_size, hidden_layer_size,
                     num_labels, X, y, regularization);

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
res = sopt.minimize(cost_function, initial_nn_params, callback=callback,
                    jac=True, options=opts, method='CG');
print(res.message)
nn_params = res.x

# Obtain Theta1 and Theta2 back from nn_params
Theta1, Theta2 = reshape_params(nn_params, input_layer_size,
                                hidden_layer_size, num_labels)

print('Program paused. Press enter to continue.\n');


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('Visualizing Neural Network...\n')

display_data(Theta1[:, 1:]);

input('Program paused. Press enter to continue.\n');

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);
accuracy = np.mean(pred == y) * 100

print('\nTraining Set Accuracy: {}\n'.format(accuracy));


## == Extra! ==
sio.savemat('../data/weights.mat', {'Theta1': Theta1, 'Theta2': Theta2})
