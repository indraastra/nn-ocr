import os
import random
import string
import sys

from fonts import load_font, char_to_glyph

FONT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cjk')
FONT_BLACKLIST = set([
])

CJK_START = 0x4E00
CJK_END = 0x9FD5

numbers = '零一二三四五六七八九'

def random_char():
    '''Returns a character in the unified CJK range.'''
    return chr(random.randrange(CJK_START, CJK_END))


def random_decimal_char():
    '''Returns a single-digit number.'''
    return random.choice(numbers)


def chars_by_range(start=CJK_START, end=CJK_END, limit=None):
    if limit:
        end = start + limit
    for i in range(start, end):
        yield chr(i)


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
    char = (len(sys.argv) > 1 and sys.argv[1]) or random_char()
    print('Showing random glyph in this font for char:', char)
    im = char_to_glyph(char, font, 72)
    im.show()
