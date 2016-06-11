import numpy as np
from scipy import sparse

from .sigmoid import sigmoid, sigmoid_gradient
from .utils import reshape_params, flatten_params

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                     num_labels, X, y, regularization):
    # Reshape weights from flattened param vectors.
    Theta1, Theta2 = reshape_params(nn_params, input_layer_size,
                                    hidden_layer_size, num_labels)

    # Number of datapoints.
    m = X.shape[0]

    # Feedforward computation.
    bias = np.ones((m, 1))
    a1 = np.append(bias, X, axis=1)
    z2 = a1.dot(Theta1.T)
    a2 = np.append(bias, sigmoid(z2), axis=1)
    a3 = sigmoid(a2.dot(Theta2.T))

    # Explode the scalar labels into one-hot vector labels.
    # eg. label '3' => [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    I = np.arange(0, max(y.shape))
    J = y.ravel()
    y_exploded = np.zeros((m, num_labels))
    y_exploded[I, J] = 1

    # Backpropagation - compute deltas and gradients.
    d3 = a3 - y_exploded
    d2 = d3.dot(Theta2)[:, 1:] * sigmoid_gradient(z2)
    Theta2_grad = d3.T.dot(a2) / m
    Theta1_grad = d2.T.dot(a1) / m
    
    Theta1_reg = (regularization / m) * Theta1
    Theta1_reg[:, 0] = 0
    Theta2_reg = (regularization / m) * Theta2
    Theta2_reg[:, 0] = 0

    Theta1_grad += Theta1_reg
    Theta2_grad += Theta2_reg

    # Compute loss with regularization of parameters, excluding bias terms.
    J = (-y_exploded      * np.log(a3) -
         (1 - y_exploded) * np.log(1 - a3)).sum() / m
    J_reg = regularization / (2 * m) * \
            ( ( Theta1[:, 1:] ** 2 ).sum() + \
              ( Theta2[:, 1:] ** 2 ).sum() )
    J += J_reg

    grad = flatten_params(Theta1_grad, Theta2_grad)

    return (J, grad)

