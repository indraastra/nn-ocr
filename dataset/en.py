import os
import string

import click
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np

from dataset.dataset import load, save, preview
from fonts import en


DATASET = 'en'
LABELS = string.ascii_letters + string.digits + string.punctuation


def load_data(image_size, depth_3=False, categorical=False, train_ratio=.8, font_limit=-1):
    fonts = en.get_fonts()
    return load(LABELS, fonts, image_size, depth_3, categorical, train_ratio, font_limit)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('output_dir')
@click.option('--image_size', default=72)
@click.option('--font_limit', default=-1)
def save_data(output_dir, image_size, font_limit):
    (X_train, y_train), (X_test, y_test) = load_data(image_size, font_limit=font_limit)
    dataset_dir = os.path.join(output_dir, DATASET)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    save(X_train, y_train, LABELS, output_dir)
    save(X_test, y_test, LABELS, output_dir)


@cli.command()
@click.option('--image_size', default=72)
@click.option('--cells_x', default=10)
@click.option('--cells_y', default=10)
@click.option('--font_limit', default=-1)
@click.option('--label', default=None)
@click.option('--randomize', default=False)
def preview_data(image_size, cells_x, cells_y, font_limit, label, randomize):
    (X_train, y_train), (X_test, y_test) = load_data(image_size, font_limit=font_limit)
    if label:
        click.echo('Previewing en data for label: {}'.format(label))
        label_idx = LABELS.index(label)
        X_preview = X_train[y_train == label_idx]
        y_preview = y_train[y_train == label_idx]
    else:
        click.echo('Previewing en data for all labels')
        X_preview = X_train
        y_preview = y_train
    preview(X_preview, y_preview, LABELS, (cells_x, cells_y), randomize)


if __name__ == '__main__':
    cli()
