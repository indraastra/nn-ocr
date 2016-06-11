import random

import click
import numpy as np
import scipy.io as sio
import scipy.optimize as sopt

from ex4.nn_cost_function import nn_cost_function
from ex4.rand_initialize_weights import rand_initialize_weights
from ex4.utils import flatten_params, load_data
from predict import make_classifier, classifier_accuracy, save_classifier


def train(X, y, num_classes,
          # These are the metaparemeters -
          hidden_layer_size, regularization, max_iterations,
          silent=False):

    input_layer_size  = X.shape[1]

    ## === Initialize weights of NN randomly. ===
    initial_Theta1 = rand_initialize_weights(input_layer_size,
                                             hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size,
                                             num_classes)

    # Unroll parameters
    initial_nn_params = flatten_params(initial_Theta1, initial_Theta2)

    ## === Train NN. ===
    opts = {
        'maxiter': max_iterations,
        'disp': not silent
    }

    # Create "short hand" for the cost function to be minimized
    cost_function = lambda p: \
        nn_cost_function(p, input_layer_size, hidden_layer_size,
                         num_classes, X, y, regularization)
    iteration = 1
    def callback(x):
        nonlocal iteration
        if iteration == max_iterations:
            print("\rIter:", iteration)
        else:
            print("\rIter:", iteration, end="")
        iteration += 1
    if silent:
        callback = lambda x: x

    # Now, costFunction is a function that takes in only one argument (the
    # neural network parameters)
    res = sopt.minimize(cost_function, initial_nn_params,
                        jac=True, options=opts, method='CG',
                        callback=callback)
    nn_params = res.x

    # Return the trained classifier.
    return make_classifier(nn_params, input_layer_size,
            hidden_layer_size, num_classes)


def make_trainer(X, y, num_classes, silent=False):
    return lambda hl, reg, mi: \
            train(X, y, num_classes, hl, reg, mi, silent)


@click.command()
@click.argument('input', type=click.Path())
@click.argument('output', type=click.Path())
@click.option('--image_pixels', default=20)
@click.option('--regularization', default=1)
@click.option('--hidden_layer_size', default=25)
@click.option('--num_classes', default=10)
@click.option('--max_iterations', default=100)
@click.option('--source', default='numpy',
              type=click.Choice(['numpy', 'matlab']))
def run_training(input, output, image_pixels, regularization,
          hidden_layer_size, num_classes, max_iterations, source):
    ## === Load training data. ===
    X, y = load_data(input, source)
    input_layer_size  = image_pixels * image_pixels  # NxN input images
    assert(input_layer_size == X.shape[1])

    # Train a classifier seeded with metaparams.
    trainer = make_trainer(X, y, num_classes)
    classifier = trainer(hidden_layer_size, regularization, max_iterations)

    ## === Predict labels for training data ===
    accuracy = classifier_accuracy(classifier, X, y)
    print('Training Set Accuracy: {}\n'.format(accuracy));

    ## == Save weights! ==
    print('Saving out Neural Network weights.\n')
    save_classifier(classifier, output)


if ( __name__ == '__main__' ):
    run_training()
