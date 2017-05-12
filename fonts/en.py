import os
import random
import string

from fonts import load_font, char_to_glyph

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


if (__name__ == '__main__'):
    fonts = get_fonts()
    print(len(fonts), 'available fonts to choose from.')
    font_path = fonts[0]
    print('Randomly selecting', font_path)
    font = load_font(font_path, 64)
    char = random.choice(string.digits + string.ascii_letters + string.punctuation)
    print('Showing random glyph in this font for char:', char)
    im = glyph_to_image(glyph, font, 72)
    im.show()
