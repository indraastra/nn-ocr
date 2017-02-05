import os
import h5py
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split

from dataset import dataset, en
from models.vgg16 import augmented_vgg16


# Path to the model weights file.
top_model_weights_path = 'models/weights/en_bottleneck_model.h5'

img_size = 64
num_classes = len(en.LABELS)

(X_train, y_train), (X_test, y_test) = en.load(img_size, depth_3=True,
        categorical=True, font_limit=100)
image_shape = X_train.shape[1:]

# Create a stacked model for the task at hand.
model = augmented_vgg16(image_shape, num_classes, weights_path=top_model_weights_path)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1)

test_generator = datagen.flow(
    X_test, y_test,
    batch_size=32,
    shuffle=False)

scores = model.evaluate_generator(test_generator, 1000)
print('Accuracy: {}'.format(scores[1]))

# randomly select a few testing images
rand_idx = np.random.choice(np.arange(0, len(y_test)), size=(25,))
rand_images = X_test[rand_idx] / 255.
probs = model.predict(rand_images)
predictions = probs.argmax(axis=1)
true_labels = y_test[rand_idx].argmax(axis=1)

print('predicted:', [en.LABELS[p] for p in predictions])
print('actual:', [en.LABELS[p] for p in true_labels])
print(sum(predictions == true_labels))
dataset.preview(rand_images[:, :, :, 1], predictions, en.LABELS, (5, 5), False)

K.clear_session()
