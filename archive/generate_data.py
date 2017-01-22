#!/usr/bin/env python

import itertools

import click
import numpy as np
import PIL
import random
import scipy.io as sio

from en_utils import get_fonts, load_glyph_set, load_font, glyph_to_image, image_to_numpy


@click.command()
@click.option('--num_fonts', default=0, help='The number of fonts to generate data for')
@click.option('--char_set', default='numbers', help='The set of characters to output')
@click.option('--pixels', default=20, help='Both dimensions of the generated images in pixels')
@click.argument('output')
def generate_data(output, num_fonts, char_set, pixels):
    fonts = get_fonts()
    if num_fonts:
        fonts = random.sample(fonts, num_fonts)

    glyphs = load_glyph_set(char_set)
    if not glyphs:
        raise click.BadParameter("Couldn't find character set: {}".format(char_set))

    # Allocate space for training data.
    H = len(fonts) * len(glyphs)  # Number of training instances.
    W = pixels * pixels           # Pixels in image.
    X = np.zeros((H, W), dtype=np.float64)
    y = np.zeros((H, 1), dtype=np.uint8)

    # Generate all numbers in all fonts.
    print("Generating font-based training data...")
    with click.progressbar(fonts) as pb:
        for i, font_path in enumerate(pb):
            font = load_font(font_path, pixels * 8 - 4)
            for j, glyph in enumerate(glyphs):
                try:
                    im = glyph_to_image(glyph, font, pixels * 8)
                    im = im.resize((pixels, pixels), resample=PIL.Image.ANTIALIAS)
                    im_np = image_to_numpy(im)
                    idx = i * len(glyphs) + j
                    X[idx, :] = im_np
                    y[idx] = j
                except OSError:
                    print("Error with font:", font_path)
                    break

    sio.savemat(output, {'X': X, 'y': y})
    click.echo('Successfully created training data!')


if (__name__ == '__main__'):
    generate_data()
