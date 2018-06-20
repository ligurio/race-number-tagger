# source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

from __future__ import print_function
import argparse
import numpy as np
import cv2
import h5py
import os
import matplotlib.pyplot as plt
import keras
import shutil
from keras import backend as K
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from bib_prepare_dataset import create_dir
from bib_prepare_dataset import find_files
from PIL import Image


LOG_DIR = "train_log"

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        embeddings_freq=1,
    )
]


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
    cv2.destroyAllWindows()


def resize_images(image_arrays, size=[32, 32]):

    image_arrays = (image_arrays * 255).astype('uint8')
    resized_image_arrays = np.zeros([image_arrays.shape[0]] + rotate(size))
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size, resample=Image.BILINEAR)
        resized_image_arrays[i] = np.asarray(resized_image)

    return np.expand_dims(resized_image_arrays, 3)


def plot_results(history, image_file):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(image_file)


def train_baseline(batch_size, num_classes, epochs, box_h, box_w):

    # input image dimensions
    img_rows, img_cols = box_h, box_w
    # image dimensions in dataset
    size = [box_w, box_h]

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

    return model


def train_bib_numbers(base_model, epochs, batch_size, num_classes, base_dir, box_h, box_w):

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_samples = len([ f for f in find_files(train_dir, '*.jpg') ])
    validation_samples = len([ f for f in find_files(validation_dir, '*.jpg') ])
    test_samples = len([ f for f in find_files(test_dir, '*.jpg') ])
    img_rows, img_cols = box_h, box_w

    if K.image_data_format() == 'channels_first':
            input_shape = (1, img_rows, img_cols)
    else:
            input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(base_model)
    model.add(Reshape((81, 31, 64), input_shape=input_shape))
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

    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adadelta(),
                              metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2)

    validation_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2)

    train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(box_h, box_w),
                    batch_size=batch_size,
                    color_mode='grayscale',
                    class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(box_h, box_w),
                    batch_size=batch_size,
                    color_mode='grayscale',
                    class_mode='categorical')

    history = model.fit_generator(
                    train_generator,
                    samples_per_epoch=train_samples,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_samples)

    print(train_samples)

    score = model.evaluate_generator(validation_generator, validation_samples/batch_size, workers=12)
    print("Total: ", len(validation_generator.filenames))
    print("Loss: ", score[0], "Accuracy: ", score[1])

    return history, model


def log_cleanup(logdir):

    print("Remove log directory %s" % logdir)
    shutil.rmtree(logdir)
    create_dir(logdir)


def main():

    batch_size = 128
    num_classes = 10
    epochs = 15

    bib_epochs = 15
    bib_batch_size = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("--box_w", type=int, dest="box_w",
                        help="box width")
    parser.add_argument("--box_h", type=int, dest="box_h",
                        help="box height")
    parser.add_argument("--processed_images_dir", type=str, dest="processed_images_dir",
                        help="path to a dir with processed images")
    args = parser.parse_args()

    #
    # Base model training
    #
    log_cleanup(LOG_DIR)
    base_model_filename = 'base_model_h%s_w%s.h5' % (args.box_h, args.box_w)
    if not os.path.isfile(base_model_filename):
        print("Base model %s is not exist. Starting to train." % base_model_filename)
        base_model = train_baseline(batch_size, num_classes, epochs, args.box_h, args.box_w)
        print("Save base model to %s." % base_model_filename)
        base_model.save(base_model_filename)

    #
    # Main model training
    #
    base_model = load_model(base_model_filename)
    print(base_model.summary())
    print("Removing layers in a base model")
    base_model = keras.Model(base_model.inputs, base_model.layers[-5].output)
    print(base_model.summary())
    print("Freezing layers in a base model")
    base_model.trainable = False
    # Freeze pretrained layers
    for layer in base_model.layers:
        layer.trainable = False
    log_cleanup(LOG_DIR)
    history, model = train_bib_numbers(base_model, bib_epochs, bib_batch_size, num_classes,
                                    args.processed_images_dir, args.box_h, args.box_w)
    model_filename = 'model_h%s_w%s.h5' % (args.box_h, args.box_w)
    plot_results(history)
    model.save(model_filename, overwrite=True)

if __name__ == "__main__":
    main()
