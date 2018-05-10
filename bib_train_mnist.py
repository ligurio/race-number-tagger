# https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
# https://stackoverflow.com/questions/45024328/resize-mnist-in-tensorflow
# https://www.tensorflow.org/api_docs/python/tf/image/resize_images
# https://github.com/yunjey/domain-transfer-network/blob/master/prepro.py

# source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from PIL import Image
import numpy as np
import settings
import cv2

def rotate(l):

    rotated = []
    for i in xrange(len(l), 0, -1):
        rotated.append(l[i - 1])
    return rotated


def show(array):

    cv2.namedWindow('Resized image', cv2.WINDOW_NORMAL)
    cv2.imshow('Resized image', array)
    print("Press any key")
    cv2.waitKey()


def resize_images(image_arrays, size=[32, 32]):

    image_arrays = (image_arrays * 255).astype('uint8')
    resized_image_arrays = np.zeros([image_arrays.shape[0]] + rotate(size))
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size, resample=Image.BILINEAR)
        resized_image_arrays[i] = np.asarray(resized_image)

    return np.expand_dims(resized_image_arrays, 3)


batch_size = 128
num_classes = 10
epochs = 15
# input image dimensions
img_rows, img_cols = settings.BOX_HEIGHT, settings.BOX_WIDTH
# image dimensions in dataset
size = [settings.BOX_WIDTH, settings.BOX_HEIGHT]

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("resize images (28 x 28) --> (%s)" % size)
x_train = resize_images(x_train, size=size)
x_test = resize_images(x_test, size=size)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.summary())

# save model
model_json = model.to_json()
with open("model-train.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(settings.MODEL_WEIGHTS_FILE)
model.save(settings.MODEL_FILE, overwrite=True)
