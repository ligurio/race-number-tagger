import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
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
epochs = 15
batch_size = 16
#batch_size = 128

model = load_model(settings.MODEL_FILE_MNIST)

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
        target_size=(settings.BOX_HEIGHT, settings.BOX_WIDTH),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(settings.BOX_HEIGHT, settings.BOX_WIDTH),
        batch_size=batch_size,
        color_mode='grayscale',
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
