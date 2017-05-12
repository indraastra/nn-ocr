import os
import random

import click
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave

from fonts import load_font, char_to_glyph

def load(labels, fonts, image_size, depth_3=False, categorical=False, train_ratio=.8, font_limit=-1):
    fonts = fonts[:font_limit]
    data = []
    skipped = 0
    # For each font, generate an image for each label and append to the
    # dataset.
    print('Generating images from fonts...')
    with click.progressbar(fonts) as pb:
        for font in pb:
            font_size = image_size - image_size // 10  # 10% boundary.
            font = load_font(font, font_size)
            for label_idx, label in enumerate(labels):
                image = char_to_glyph(label, font, image_size)
                if not image:
                    skipped += 1
                    continue
                # This is a grayscale image of size (dim, dim, 1).
                np_image = img_to_array(image)
                # This is a 3D image of size (dim, dim, 3).
                if depth_3:
                    np_image = np.tile(np_image, (1, 1, 3))
                data.append((np_image, label_idx))

    print('{} images generated from {} fonts.'.format(len(data),
                                                      len(fonts)))
    print('{} characters skipped.'.format(skipped))

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

def save(X, y, labels, output_dir):
    classdirs = []
    for i in range(len(labels)):
        classdir = os.path.join(output_dir, str(i))
        os.makedirs(classdir, exist_ok=True)
        classdirs.append(classdir)

    for i in range(len(y)):
        class_idx = y[i]
        filename = os.path.join(classdirs[class_idx], str(i)) + '.png'
        if len(X[i].shape) == 3 and X[i].shape[-1] == 1:
            # Drop the last dimension if grayscale.
            imsave(filename, X[i].reshape(X[i].shape[:2]))
        else:
            imsave(filename, X[i])


def preview(X, y, labels, cells, randomize):
    # Select as many images as can neatly fit in a square display.
    w, h = cells

    num_images = w * h
    images = X[:num_images]
    size = images[0].shape[0]

    print("Displaying {}x{} images in a {}x{} grid.".format(size, size,
        h, w))

    combined_image = np.zeros((size * h, size * w))
    image_labels = [[None] * w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            image = images[idx]
            start_y = i * size;
            start_x = j * size;
            combined_image[start_y:(start_y + size),
                           start_x:(start_x + size)] = image.reshape((size, size), order='F')
            image_labels[i][j] = labels[y[idx]]

    # Show image and display labels for debugging.
    print('Labels:')
    print('\n'.join(' '.join(ls) for ls in image_labels))

    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(combined_image, interpolation='nearest')
    plt.show()

