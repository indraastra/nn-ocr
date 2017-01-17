# Adapted from http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

from keras_port.lenet import LeNet
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--save_model', action='store_true',
    help='(optional) whether or not model should be saved to disk')
ap.add_argument('-l', '--load_model', action='store_true',
    help='(optional) whether or not pre-trained model should be loaded')
ap.add_argument('-w', '--weights_file', type=str,
    help='(optional) path to weights file')
args = vars(ap.parse_args())

print('[INFO] downloading MNIST...')
dataset = datasets.fetch_mldata('MNIST Original')
batch_size = 128
nb_classes = 10
nb_epoch = 75
nb_samples_per_epoch = 2048

# Reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits.
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
# Construct a 1xNxN one-channel image from an NxN image.
data = data[:, :, :, np.newaxis]
(train_data, test_data, train_labels, test_labels) = train_test_split(
    data / 255.0, dataset.target.astype('int'), test_size=0.33)

# Transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels.
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

# Create generators for both datasets.
train_datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

test_datagen = ImageDataGenerator()

# Initialize the optimizer and model.
print('[INFO] compiling model...')
model = LeNet.build(width=28, height=28, depth=1, num_classes=nb_classes,
    weights_path=args['weights_file'] if args['load_model'] > 0 else None)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
    metrics=['accuracy'])

# Only train and evaluate the model if we *are not* loading a
# pre-existing model.
if not args['load_model']:
  print('[INFO] training...')
  model.fit_generator(train_datagen.flow(train_data, train_labels, batch_size=batch_size),
        samples_per_epoch=nb_samples_per_epoch, nb_epoch=nb_epoch, verbose=1)

  # Show the accuracy on the testing set.
  print('[INFO] evaluating...')
  (loss, accuracy) = model.evaluate_generator(test_datagen.flow(test_data, test_labels),
          val_samples=nb_samples_per_epoch)
  print('[INFO] testing accuracy: {:.2f}%'.format(accuracy * 100))

# Check to see if the model should be saved to file.
if args['save_model'] > 0:
  print('[INFO] dumping weights to file...')
  model.save_weights(args['weights_file'], overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(test_labels)), size=(10,)):
  # classify the digit
  disp = np.round(test_data[i].reshape(28, 28, order='C')).astype('uint8')
  print(disp)
  probs = model.predict(test_data[np.newaxis, i])
  prediction = probs.argmax(axis=1)

  # resize the image from a 28 x 28 image to a 96 x 96 image so we
  # can better see it
  image = (test_data[i, :, :, 0] * 255).astype('uint8')
  image = cv2.merge([image] * 3)
  image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LINEAR)
  cv2.putText(image, str(prediction[0]), (5, 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

  # show the image and prediction
  print('[INFO] Predicted: {}, Actual: {}'.format(prediction[0],
      np.argmax(test_labels[i])))
  cv2.imshow('Digit', image)
  cv2.waitKey(0)

K.clear_session()
