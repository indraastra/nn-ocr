import math
import os

import click
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave


DATASET = 'mnist'
LABELS = '0123456789'


def load():
    return mnist.load_data()


def save(X, y, output_dir):
    classdirs = []
    for label in LABELS:
        classdir = os.path.join(output_dir, label)
        os.makedirs(classdir, exist_ok=True)
        classdirs.append(classdir)

    for i in range(len(y)):
        class_idx = y[i]
        label = LABELS[class_idx]
        filename = os.path.join(classdirs[class_idx], str(i)) + '.png'
        imsave(filename, X[i])


def preview(X, cells, randomize):
    # Select as many images as can neatly fit in a square display.
    w, h = cells
    print("Displaying images in a {}x{} grid.".format(h, w))

    num_images = w * h
    images = X[:num_images]
    size = images[0].shape[0]

    combined_image = np.zeros((size * h, size * w))

    for i in range(h):
        for j in range(w):
            image = images[i * w + j]
            start_y = i * size;
            start_x = j * size;
            combined_image[start_y:(start_y + size),
                           start_x:(start_x + size)] = image.reshape((size, size), order='F')

    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(combined_image, interpolation='nearest')
    plt.show()


@click.group()
def cli():
    pass


@cli.command()
def load_data():
    click.echo('Loading MNIST data...')
    return load()


@cli.command()
@click.argument('output_dir')
def save_data(output_dir):
    (X_train, y_train), (X_test, y_test) = load()
    dataset_dir = os.path.join(output_dir, DATASET)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    save(X_train, y_train, train_dir)
    save(X_test, y_test, test_dir)


@cli.command()
@click.option('--cells_x', default=10)
@click.option('--cells_y', default=10)
@click.option('--randomize', default=False)
def preview_data(cells_x, cells_y, randomize):
    click.echo('Previewing MNIST data...')
    (X_train, y_train), (X_test, y_test) = load()
    preview(X_train, (cells_x, cells_y), randomize)


if __name__ == '__main__':
    cli()
