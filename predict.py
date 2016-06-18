import random

import click
import numpy as np
import scipy.io as sio
import scipy.optimize as sopt

from matlab_port.utils import reshape_params, flatten_params, load_data
from matlab_port.sigmoid import sigmoid


def predict_best(classifier, X):
    label, _ = predict_top_n(classifier, X, 1)
    return label.flatten()

# For backwards compatibility...
predict = lambda c, X, l=1: predict_best(c, X)


def predict_top_n(classifier, X, limit=10):
    '''Returns a list of sorted (label, score) tuples.'''
    Theta1 = classifier['Theta1']
    Theta2 = classifier['Theta2']

    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros((m, 1))

    bias = np.ones((m, 1))
    h1 = sigmoid(np.c_[bias, X].dot(Theta1.T))
    h2 = sigmoid(np.c_[bias, h1].dot(Theta2.T))

    # Return top n classes.
    top_n = (-h2).argsort(axis=1)[:, :limit]
    rows = np.tile(np.arange(len(h2)), (limit, 1)).T
    return top_n.flatten(), h2[rows, top_n].flatten()


def load_classifier(params_file):
    # Load the weights into variables Theta1 and Theta2
    params = sio.loadmat(params_file)
    weights = params['weights'].flatten()
    input_layer_size = params['input_size'][0][0]
    hidden_layer_size = params['hidden_size'][0][0]
    num_classes = params['output_size'][0][0]

    return make_classifier(weights, input_layer_size, hidden_layer_size, num_classes)


def save_classifier(classifier, params_file):
    sio.savemat(params_file, classifier)


def make_classifier(weights, input_size, hidden_size, num_classes):
    Theta1, Theta2 = reshape_params(weights,
            input_size, hidden_size, num_classes)
    return {
        'weights': weights,
        'Theta1': Theta1,
        'Theta2': Theta2,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': num_classes
    }


def classifier_accuracy(classifier, X, correct_labels):
    predicted = predict(classifier, X, 1)
    accuracy = np.mean(predicted == correct_labels) * 100
    return accuracy


@click.command()
@click.argument('test_data', type=click.Path())
@click.argument('weights', type=click.Path())
@click.option('--hidden_layer_size', default=25)
@click.option('--num_classes', default=10)
@click.option('--source', default='numpy',
              type=click.Choice(['numpy', 'matlab']))
def run_predicton(test_data, weights, hidden_layer_size, num_classes, source):
    ## === Load and visualize the test data. ===
    X, y = load_data(test_data, source)
    input_layer_size = X.shape[1]

    ## === Load NN weights. ===
    print('Loading saved Neural Network parameters ...')
    classifier = load_classifier(weights)

    ## === Predict labels for test data ===
    accuracy = classifier_accuracy(classifier, X, y)
    print('Test Set Accuracy:', accuracy)

    # Predict value for a random instance.
    instance = random.randint(0, X.shape[0])
    example = (X[instance, :].reshape(20, 20, order='F').round().astype(np.uint8)) 
    print(example)
    pred = predict(classifier, example.reshape(1, 400, order='F'), 1)
    print(pred, y[instance])


if ( __name__ == '__main__' ):
    run_predicton()
