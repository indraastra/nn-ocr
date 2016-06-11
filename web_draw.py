from base64 import b64decode
from io import BytesIO

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image

from en_utils import image_to_numpy, load_glyph_set
from predict import load_classifier, predict_top_n


IMG_PREFIX = 'data:image/png;base64,'
IMG_SIZE = 20
IMG_WEIGHTS = 'weights/letters.mat'
GLYPH_SET = 'letters'
GLYPHS = load_glyph_set(GLYPH_SET)


app = Flask(__name__)
classifier = load_classifier(IMG_WEIGHTS)


@app.route('/')
def index():
    return render_template('draw.html')


@app.route('/classify', methods=['POST'])
def classify():
    results = {}
    if 'imageb64' not in request.form:
        return jsonify(results)

    data = request.form['imageb64']
    if not data.startswith(IMG_PREFIX):
        return jsonify(results)

    data = data[len(IMG_PREFIX):]
    image = Image.open(BytesIO(b64decode(data)))
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image_to_numpy(image)
    print(image.reshape(IMG_SIZE, IMG_SIZE, order='F').astype(np.uint8))
    labels, scores = predict_top_n(classifier, image, limit=3)

    results['labels'] = labels.tolist()[0]
    results['scores'] = scores.tolist()[0]
    results['glyphs'] = [GLYPHS[l] for l in results['labels']]

    print(results)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

