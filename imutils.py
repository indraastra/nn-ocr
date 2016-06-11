def _sliding_windows(image_size, window_size, step_size):
    print(image_size, window_size, step_size)
    w, h = image_size
    w_x, w_y = window_size
    s_x, s_y = step_size
    for x in range(0, w - w_x + 1, s_x):
        for y in range(0, h - w_y + 1, s_y):
            yield (x, y, x + w_x, y + w_y)


def _reducing_windows(image_size, minimum_size,
                      reduction_factor=.75, step_size_factor=.25):
    window_size = image_size[:]
    while all(dim > minimum_size for dim in window_size):
        step_size = (int(step_size_factor * window_size[0]),
                     int(step_size_factor * window_size[1]))
        for box in _sliding_windows(image_size, window_size, step_size):
            yield box
        window_size = (int(reduction_factor * window_size[0]),
                       int(reduction_factor * window_size[1]))


def image_windows(image, target_size, minimum_size,
                  reduction_factor, step_size_factor):
    for box in _reducing_windows(image.size, minimum_size,
                                 reduction_factor,
                                 step_size_factor):
        # Extract window and resize to original image size for
        # classification.
        subimage = image.crop(box)
        subimage = subimage.resize(target_size)
        yield subimage


def square_bbox(image):
    bbox = image.getbbox()
    if not bbox: return

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w > h:
        diff = w - h
        bbox = (bbox[0], bbox[1] - diff//2, bbox[2], bbox[3] + diff//2)
    elif w < h:
        diff = h - w
        bbox = (bbox[0] - diff//2, bbox[1], bbox[2] + diff//2, bbox[3])

    return bbox
