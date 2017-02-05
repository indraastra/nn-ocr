import json

from flask import Flask, jsonify, render_template, request
import numpy as np
import tensorflow as tf

from models import trained_models

app = Flask(__name__)
img_size, labels, model = trained_models.en_vgg16()
num_classes = len(labels)


@app.route('/')
def index():
    rand_image = np.random.random((1, img_size, img_size, 1))
    rand_image = np.tile(rand_image, (1, 1, 1, 3))
    with graph.as_default():
        probs = model.predict(rand_image)
    return json.dumps(probs[0].tolist())


if __name__ == '__main__':
    graph = tf.get_default_graph()
    app.run(debug=False, threaded=True)

