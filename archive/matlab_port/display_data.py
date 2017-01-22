import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def display_data(images, order='F'):
    # Select as many images as can neatly fit in a square display.
    w = int(math.sqrt(len(images)))
    h = int(len(images) / w)
    images = images[:w*h]
    print("Displaying images in a {}x{} array.".format(h, w))

    size = int(math.sqrt(images[0].shape[0]))

    combined_image = np.zeros((size * h, size * w))

    for i in range(h):
        for j in range(w):
            image = images[i * w + j]
            start_x = i * size;
            start_y = j * size;
            combined_image[start_y:(start_y + size),
                           start_x:(start_x + size)] = image.reshape((size, size), order=order)

    # NOTE: IF this is failing to display anything, there's a known issue on Ubuntu.
    # See:
    # http://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/
    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(combined_image, interpolation='nearest')
    plt.show()
