import os
import h5py
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split

from dataset import en
from models.vgg16 import augmented_vgg16


# Path to the model weights file.
top_model_weights_path = 'weights/en_bottleneck_model.h5'

img_size = 64
num_classes = len(en.LABELS)

nb_epoch = 500

(X_train, y_train), (X_test, y_test) = en.load_data(img_size, depth_3=True, categorical=True)
image_shape = X_train.shape[1:]


datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1)

train_generator = datagen.flow(
        X_train, y_train,
        batch_size=64,
        shuffle=False)

test_generator = datagen.flow(
        X_test, y_test,
        batch_size=64,
        shuffle=False)

# Create a stacked model for the task at hand.
en_model = augmented_vgg16(image_shape, num_classes)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1)

test_generator = datagen.flow(
        X_test, y_test,
        batch_size=64,
        shuffle=False)

en_model.fit_generator(train_generator, samples_per_epoch=2000,
        nb_epoch=nb_epoch, validation_data=test_generator, nb_val_samples=1000)
en_model.save_weights(top_model_weights_path)
