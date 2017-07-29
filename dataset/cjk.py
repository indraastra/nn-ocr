import os

from fonts import cjk


DATASET_NAME = 'cjk'
LABELS = list(cjk.chars_by_range())


def get_labels():
    return LABELS

get_fonts = cjk.get_fonts
