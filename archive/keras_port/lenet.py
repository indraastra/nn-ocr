# Adapted from http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout

class LeNet:
  @staticmethod
  def build(width, height, depth, num_classes, weights_path=None, dropout=True):
    model = Sequential()

    # First set of CONV => RELU => POOL.
    model.add(Convolution2D(20, 5, 5, border_mode='same',
                            input_shape=(height, width, depth)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second set of CONV => RELU => POOL.
    model.add(Convolution2D(50, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Last set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(.5))

    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # If a weights path is supplied (inicating that the model was
    # pre-trained), then load the weights
    if weights_path is not None:
      model.load_weights(weights_path)

    # return the constructed network architecture
    return model

