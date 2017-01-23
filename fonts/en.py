import os
import random
import string

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps

FONT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'en')
FONT_BLACKLIST = set([
    'Corben-Bold.ttf',
    'FiraMono-Regular.ttf',
    'FiraMono-Medium.ttf',
    'FiraMono-Bold.ttf',
])


def get_fonts(shuffle=True):
    fonts = []
    for font in os.listdir(FONT_DIR):
        if (font not in FONT_BLACKLIST and
            (font.endswith('.ttf') or font.endswith('.otf') or font.endswith('.otc'))):
            fonts.append(os.path.join(FONT_DIR, font))
    if shuffle:
        random.shuffle(fonts)
    return fonts


def load_font(path, font_size=64):
    return ImageFont.truetype(path, font_size)


def glyph_to_image(glyph, font, image_size=64, mode='L'):
    im = Image.new(mode, (image_size, image_size), 'black')
    draw = ImageDraw.Draw(im)
    # Vertically and horizontally center glyph in canvas, accounting for offset.
    w, h = font.getsize(glyph)
    x, y = font.getoffset(glyph)
    draw.text(((image_size-w-x)/2, (image_size-h-y)/2), glyph, font=font, fill='white')
    return im


if (__name__ == '__main__'):
    fonts = get_fonts()
    print(len(fonts), 'available fonts to choose from.')
    font_path = fonts[0]
    print('Randomly selecting', font_path)
    font = load_font(font_path, 64)
    glyph = random.choice(string.digits + string.ascii_letters + string.punctuation)
    print('Showing random glyph in this font:', glyph)
    im = glyph_to_image(glyph, font, 72)
    im.show()
