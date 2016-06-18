import itertools

import click

from train import make_trainer
from predict import classifier_accuracy, save_classifier
from matlab_port.utils import partition_data, shuffle_data, load_data

HIDDEN_LAYER_OPTS   = [ 25, 50, 100 ]
REGULARIZATION_OPTS = [ 0.03, 0.1, 0.3, 1, 3, 10, 30 ]
MAX_ITERATION_OPTS  = [ 50, 100 ]

def optimize_hyperparams(X, y, num_classes,
        hidden_layer_sizes, regularization_terms, max_iterations):

    ## === Divide training data. ===
    X_train, y_train, X_val, y_val = partition_data(X, y, split=.9)

    trainer = make_trainer(X_train, y_train, num_classes, silent=True)

    all_combos = itertools.product(
            hidden_layer_sizes, regularization_terms, max_iterations)

    best_classifier = None
    best_opts = None
    best_accuracy = 0

    # Find hyperparameter combination that maximizes validation set accuracy.
    # TODO: Use multiprocessing to map across these combinations!
    # TODO: Rewrite this as some kind of argmax.
    for combo in all_combos:
        print('> Attempting:', combo)
        classifier = trainer(*combo)
        accuracy = classifier_accuracy(classifier, X_val, y_val)
        print('< Validation set accuracy:', accuracy)
        if accuracy > best_accuracy:
            best_opts = combo
            best_classifier = classifier
            best_accuracy = accuracy

    return best_classifier, best_opts, best_accuracy


@click.command()
@click.argument('input', type=click.Path())
@click.argument('output', type=click.Path())
@click.option('--num_classes', default=10)
@click.option('--source', default='numpy',
              type=click.Choice(['numpy', 'matlab']))
def run_optimized_training(input, output, num_classes, source):
    X, y = load_data(input, source)
    X, y = shuffle_data(X, y)

    ## === Divide training data. ===
    X, y, X_test, y_test = partition_data(X, y, split=.9)

    classifier, opts, val_accuracy = optimize_hyperparams(
            X, y, num_classes,
            HIDDEN_LAYER_OPTS,
            REGULARIZATION_OPTS,
            MAX_ITERATION_OPTS)

    test_accuracy = classifier_accuracy(classifier, X_test, y_test)

    print()
    print('===================')
    print('OPTIMAL PARAMETERS:')
    print('Hidden layer size: ', opts[0])
    print('Regularization value: ', opts[1])
    print('Max training iterations: ', opts[2])
    print('Accuracy on validation set:', val_accuracy)
    print('Accuracy on test set:', test_accuracy)
    print('===================')
    print()

    print('Saving out Neural Network weights.')
    save_classifier(classifier, output)


if ( __name__ == '__main__' ):
    run_optimized_training()
