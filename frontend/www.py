from base64 import b64decode
from io import BytesIO

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from imutils import square_bbox
from models import trained_models

IMG_PREFIX = 'data:image/png;base64,'

app = Flask(__name__)
img_size, labels, model = trained_models.en_vgg16()


def image_to_classifier_input(image_bytes, depth_3=True):
    image = Image.open(BytesIO(b64decode(image_bytes)))
    image = ImageOps.grayscale(image)
    target_size = (img_size, img_size)
    image = image.resize(target_size)

    mat = np.array(image.getdata(), np.float64) / 255
    mat = mat.reshape(target_size, order='C')
    mat = mat[np.newaxis, :, :, np.newaxis]
    if depth_3:
        mat = np.tile(mat, (1, 1, 1, 3))
    return mat


@app.route('/classify', methods=['POST'])
@app.route('/classify/<int:top_n>', methods=['POST'])
def classify(top_n=5):
    results = {'results': [], 'status': None}

    if 'imageb64' not in request.form:
        results['status'] = 'missing image'
        return jsonify(results)

    data = request.form['imageb64']
    if not data.startswith(IMG_PREFIX):
        results['status'] = 'image not in base64'
        return jsonify(results)

    image_data = data[len(IMG_PREFIX):]
    classifier_input = image_to_classifier_input(image_data)
    if classifier_input is None:
        results['status'] = 'no image to classify'
        return jsonify(results)

    with graph.as_default():
        probs = model.predict(classifier_input).flatten()
    res = []
    for i, score in enumerate(probs):
        res.append({
            'score': float(score),
            'label': labels[i],
        })
    res.sort(key=lambda d: d['score'], reverse=True)
    results['results'] = res[:top_n]

    return jsonify(results)


@app.route('/')
def index():
    return render_template('draw.html')


if __name__ == '__main__':
    graph = tf.get_default_graph()
    app.run(debug=False, threaded=True)

