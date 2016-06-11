#!/usr/bin/env python

import random

import click
import numpy as np

from ex4.display_data import display_data
from ex4.utils import load_data


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

@click.command()
@click.argument('input')
@click.option('--source', default='numpy',
              type=click.Choice(['numpy', 'matlab']))
@click.option('--label', type=int, default=None)
def visualize_data(input, source, label):
    # Load Training Data
    print('> Loading and Visualizing Data ...\n')

    X, y = load_data(input, source)
    if label:
        X = X[y == label, :]
        y = y[y == label]
    m = X.shape[0]

    # Randomly select 100 data points to display
    sel = random.sample(range(m), 100)

    display_data(X[sel, :], order='F');


if (__name__ == '__main__'):
    visualize_data()
