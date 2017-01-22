import math

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


LABELS = '0123456789'

def load_data():
    return mnist.load_data()

def preview_data(X, cells=(10, 10), randomize=False):
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
            start_x = i * size;
            start_y = j * size;
            combined_image[start_y:(start_y + size),
                           start_x:(start_x + size)] = image.reshape((size, size), order='F')

    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(combined_image, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    # Preview a random set of training data.
    (X_train, y_train), (X_test, y_test) = load_data()
    preview_data(X_train, randomize=True)
