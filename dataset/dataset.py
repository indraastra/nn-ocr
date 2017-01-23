import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave


def save(X, y, labels, output_dir):
    classdirs = []
    for i in range(len(labels)):
        classdir = os.path.join(output_dir, str(i))
        os.makedirs(classdir, exist_ok=True)
        classdirs.append(classdir)

    for i in range(len(y)):
        class_idx = y[i]
        filename = os.path.join(classdirs[class_idx], str(i)) + '.png'
        imsave(filename, X[i])


def preview(X, y, labels, cells, randomize):
    # Select as many images as can neatly fit in a square display.
    w, h = cells
    print("Displaying images in a {}x{} grid.".format(h, w))

    num_images = w * h
    images = X[:num_images]
    size = images[0].shape[0]

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

