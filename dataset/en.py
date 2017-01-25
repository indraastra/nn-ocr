import os
import random
import string

import click
from keras.datasets import mnist
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
import numpy as np

from dataset.dataset import save, preview
from fonts import en


DATASET = 'en'
LABELS = string.ascii_letters + string.digits + string.punctuation


def load(image_size, depth_3=False, categorical=False, train_ratio=.8, font_limit=-1):
    fonts = en.get_fonts()[:font_limit]
    data = []
    # For each font, generate an image for each label and append to the
    # dataset.
    print('Generating images from fonts...')
    with click.progressbar(fonts) as pb:
        for font in pb:
            font_size = image_size - image_size // 10  # 10% boundary.
            font = en.load_font(font, font_size)
            for label_idx, label in enumerate(LABELS):
                # TODO: Skip empty images.
                image = en.glyph_to_image(label, font, image_size)
                # This is a grayscale image of size (dim, dim, 1).
                np_image = img_to_array(image)
                # This is a 3D image of size (dim, dim, 3).
                if depth_3:
                    np_image = np.tile(np_image, (1, 1, 3))
                data.append((np_image, label_idx))

    print('{} images generated from {} fonts.'.format(len(data),
                                                      len(fonts)))

    # Shuffle data and split into training and test sets.
    random.shuffle(data)
    X = np.array([t[0] for t in data])
    y = np.array([t[1] for t in data])
    if categorical:
        y = to_categorical(y)

    split_idx = int(train_ratio * len(data))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return (X_train, y_train), (X_test, y_test)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('output_dir')
@click.option('--image_size', default=72)
@click.option('--font_limit', default=-1)
def save_data(output_dir, image_size, font_limit):
    (X_train, y_train), (X_test, y_test) = load(image_size, font_limit=font_limit)
    dataset_dir = os.path.join(output_dir, DATASET)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    return save(X_train, y_train, LABELS, output_dir)
    return save(X_test, y_test, LABELS, output_dir)


@cli.command()
@click.option('--image_size', default=72)
@click.option('--cells_x', default=10)
@click.option('--cells_y', default=10)
@click.option('--font_limit', default=-1)
@click.option('--randomize', default=False)
def preview_data(image_size, cells_x, cells_y, font_limit, randomize):
    (X_train, y_train), (X_test, y_test) = load(image_size, font_limit=font_limit)
    click.echo('Previewing en data...')
    preview(X_train, y_train, LABELS, (cells_x, cells_y), randomize)


if __name__ == '__main__':
    cli()
