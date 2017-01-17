from base64 import b64decode
from collections import defaultdict
from io import BytesIO

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from en_utils import image_to_numpy, load_glyph_set
from imutils import square_bbox
from keras_port.lenet import LeNet
from predict import load_classifier, predict_top_n


IMG_PREFIX = 'data:image/png;base64,'
IMG_SIZE = 28
IMG_WEIGHTS = 'weights/lenet_mnist_aug.h5'
GLYPH_SET = 'numbers'
GLYPHS = load_glyph_set(GLYPH_SET)


app = Flask(__name__)


def image_to_input(image_bytes):
    image = Image.open(BytesIO(b64decode(image_bytes)))
    image = ImageOps.grayscale(image)
    target_size = (IMG_SIZE, IMG_SIZE)

    # Crop image to bounding box and resize to target image.
    bbox = square_bbox(image)
    if not bbox:
      return
    b = 20
    bbox = (bbox[0] - b, bbox[1] - b, bbox[2] + b, bbox[3] + b)

    subimage = image.crop(bbox)
    subimage = subimage.resize(target_size)

    mat = np.array(subimage.getdata(), np.float64) / 255
    mat = mat.reshape(target_size, order='C')
    print(mat.astype('uint8'))
    mat = mat[np.newaxis, :, :, np.newaxis]
    return mat


@app.route('/')
def index():
    return render_template('draw.html')


@app.route('/classify', methods=['POST'])
def classify():
    results = {'results': [], 'status': None}

    if 'imageb64' not in request.form:
        results['status'] = 'missing image'
        return jsonify(results)

    data = request.form['imageb64']
    if not data.startswith(IMG_PREFIX):
        results['status'] = 'image not in base64'
        return jsonify(results)

    classifier_input = image_to_input(data[len(IMG_PREFIX):])
    if not input:
        results['status'] = 'no image to classify'
        return jsonify(results)

    #labels, scores = predict_top_n(classifier, subimage, limit=3)
    with graph.as_default():
      scores = classifier.predict(classifier_input).flatten()
      print(scores)

      res = []
      for i, score in enumerate(scores):
          res.append({
              'label': i,
              'score': float(score),
              'glyph': GLYPHS[i],
          })
          print(GLYPHS[i], score)
      res.sort(key=lambda d: d['score'], reverse=True)
      results['results'] = res

    return jsonify(results)


if __name__ == '__main__':
    classifier = LeNet.build(width=IMG_SIZE, height=IMG_SIZE, depth=1, num_classes=len(GLYPHS),
        weights_path=IMG_WEIGHTS)
    graph = tf.get_default_graph()

    app.run(debug=False)

