import string

from fonts import en


DATASET_NAME = 'en'
LABELS = string.ascii_letters + string.digits + string.punctuation


def get_labels():
    return LABELS


get_fonts = en.get_fonts
