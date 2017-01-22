#!/usr/bin/env python

import math
import random

import click
import numpy as np
import scipy.io as sio

from matlab_port.display_data import display_data
from matlab_port.utils import load_data


FORMAT = {
    'numpy': 'C',
    'matlab': 'F'
}


@click.command()
@click.argument('input')
@click.argument('output')
@click.option('--source', default='matlab',
              type=click.Choice(['numpy', 'matlab']))
@click.option('--target', default='numpy',
              type=click.Choice(['numpy', 'matlab']))
def convert_data(input, output, source, target):
    X, y = load_data(input, source)
    m = X.shape[0]

    # Load Training Data
    print('Loading data!')

    # Randomly select 100 data points to display
    sel = random.sample(range(m), 100)
    display_data(X[sel, :], order=FORMAT[target])

    # Transpose all images.
    for i in range(m):
        pixels = int(math.sqrt(X.shape[1]))
        image = X[i, :].reshape(pixels, pixels, order=FORMAT[source])
        X[i, :] = image.reshape(1, X.shape[1],  order=FORMAT[target])

    display_data(X[sel, :], order=FORMAT[target])

    sio.savemat(output, {'X': X, 'y': y})

if (__name__ == '__main__'):
    convert_data()
