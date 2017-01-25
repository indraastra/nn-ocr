from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


def augmented_vgg16(image_shape, num_classes, weights_path=None):
    vgg16_model = VGG16(include_top=False,
                        weights='imagenet' if not weights_path else None,
                        input_shape=image_shape)
    # Freeze VGG16 model.
    for layer in vgg16_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(vgg16_model)
    model.add(Flatten(input_shape=vgg16_model.layers[-1].output_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
