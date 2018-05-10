# https://github.com/tanmayb123/MNIST-CNN-in-Keras/blob/master/customtrain.py
# MNIST model example https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from bib_prepare_dataset import create_dir
from bib_prepare_dataset import find_files
import h5py
import settings
import os

base_dir = './data/race_numbers/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_samples = len([ f for f in find_files(train_dir, '*.jpg') ])
validation_samples = len([ f for f in find_files(validation_dir, '*.jpg') ])
test_samples = len([ f for f in find_files(test_dir, '*.jpg') ])
epochs = 30
batch_size = 16

# model = load_model(settings.MODEL_FILE_MNIST)

# ** Model Begins **
model = Sequential()
model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(settings.BOX_WIDTH, settings.BOX_HEIGHT, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))

model.add(Dense(10, activation='softmax'))
# ** Model Ends **

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
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
        target_size=(settings.BOX_WIDTH, settings.BOX_HEIGHT),
        batch_size=32,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(settings.BOX_WIDTH, settings.BOX_HEIGHT),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples)

score = model.evaluate_generator(validation_generator, validation_samples/batch_size, workers=12)
print("Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])
print model.summary()

model.save(settings.MODEL_FILE_REAL, overwrite=True)
model.save_weights(settings.MODEL_WEIGHTS_FILE_REAL, overwrite=True)
