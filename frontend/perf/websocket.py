import json

import eventlet
import eventlet.wsgi
from flask import Flask, render_template
import numpy as np
import socketio
import tensorflow as tf

from models import trained_models

sio = socketio.Server()
app = Flask(__name__)

img_size, num_classes, model = trained_models.en_vgg16()


@app.route('/')
def index():
    rand_image = np.random.random((1, img_size, img_size, 1))
    rand_image = np.tile(rand_image, (1, 1, 1, 3))
    with graph.as_default():
        probs = model.predict(rand_image)
    return json.dumps(probs[0].tolist())


@sio.on('classify', namespace='/ocr')
def classify(message):
    rand_image = np.random.random((1, img_size, img_size, 1))
    rand_image = np.tile(rand_image, (1, 1, 1, 3))
    with graph.as_default():
        probs = model.predict(rand_image)
    emit('classify_response', {'data': probs[0].tolist()})

@sio.on('connect', namespace='/ocr')
def connect():
    emit('connect_response', {'data': 'Connected'})

@sio.on('disconnect', namespace='/ocr')
def disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    graph = tf.get_default_graph()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

    socketio.run(app, debug=False)
