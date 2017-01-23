import os

import click
from keras.datasets import mnist

from dataset.dataset import save, preview


DATASET = 'mnist'
LABELS = '0123456789'


def load():
    return mnist.load_data()


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

    return save(X_train, y_train, LABELS, output_dir)
    return save(X_test, y_test, LABELS, output_dir)


@cli.command()
@click.option('--cells_x', default=10)
@click.option('--cells_y', default=10)
@click.option('--randomize', default=False)
def preview_data(cells_x, cells_y, randomize):
    click.echo('Previewing MNIST data...')
    (X_train, y_train), (X_test, y_test) = load()
    preview(X_train, y_train, LABELS, (cells_x, cells_y), randomize)


if __name__ == '__main__':
    cli()
