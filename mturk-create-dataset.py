# -*- coding: utf-8 -*-

"""
на вход принимает аннотацию в JSON формате, сортирует изображения
и раскладывает их в директориях такой структуры:

race_numbers
|____
|    \
|    train
|      |--- 0 
|      |--- 1
|      |--- 2
|      |--- 3
|      |--- ..
|____
|    \
|    validation
|____
     \
     test

"""

import json
import shutil
import sys
import os

annotation_file = sys.argv[1]
annotation = json.load(open(annotation_file))

original_dataset_dir = 'race_numbers_original_data'
base_dir = 'race_numbers'
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

for number in range(0, 9):
    train_num_dir = os.path.join(train_dir, str(number))
    os.mkdir(train_num_dir)

    validation_num_dir = os.path.join(validation_dir, str(number))
    os.mkdir(validation_num_dir)

    test_num_dir = os.path.join(test_dir, str(number))
    os.mkdir(test_num_dir)

# TODO: shuffle image structs in annotation
# TODO: copy predefined number of files to train, validation and test dirs

# нужно сразу делать JSON с отдельными записями на box, а не на image

TEST_NUM = 300
TRAIN_NUM = 200
VALIDATION_NUM = 300

for image in annotation:
    for box in image['boxes']:
        number = box['label']
        src = os.path.join(original_dataset_dir, image['filename'])
        train_num_dir = os.path.join(train_dir, str(number))
        dst = os.path.join(train_num_dir, image['filename'])
        shutil.copyfile(src, dst)

print('total training images:', len(os.listdir(train_num_dir) - 10))
print('total validation images:', len(os.listdir(validation_num_dir)) - 10)
print('total test images:', len(os.listdir(test_num_dir)) - 10)
