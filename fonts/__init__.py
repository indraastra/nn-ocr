import freetype
from PIL import Image, ImageFont, ImageDraw

def load_font(path, font_size=64):
    pil_font = ImageFont.truetype(path, font_size)
    ft_font = freetype.Face(path)
    ft_font.set_char_size(font_size*font_size)
    return (pil_font, ft_font)


def char_to_glyph(char, font, image_size=64, mode='L'):
    pil_font, ft_font = font
    if ft_font.get_char_index(char) == 0:
        # Character is not part of this font face.
        return None
    im = Image.new(mode, (image_size, image_size), 'black')
    draw = ImageDraw.Draw(im)
    # Vertically and horizontally center glyph in canvas, accounting for offset.
    w, h = pil_font.getsize(char)
    x, y = pil_font.getoffset(char)
    draw.text(((image_size-w-x)/2, (image_size-h-y)/2), char, font=pil_font, fill='white')
    return im


