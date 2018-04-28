# -*- coding: utf-8 -*-

#
#

import csv
import json
import os
import sys
import json
import pprint
import random
import numpy as np
from PIL import Image
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import settings
import shutil


def id_generator():
    return ''.join([ random.choice(string.ascii_letters + string.digits) for n in xrange(32) ])

def box_dim_percentile(hits, percentile):

    width = []
    height = []
    for image in hits:
        for box in image['boxes']:
            width.append(box['width'])
            height.append(box['height'])

    return np.percentile(np.array(width), percentile),
            np.percentile(np.array(height), percentile)


def sanitize_boxes(boxes):

    updated_boxes = []
    for box in boxes:
        b = {'height': int(box['height']),
             'top': int(box['top']),
             'left': int(box['left']),
             'label': int(box['label']),
             'width': int(box['width'])}
        updated_boxes.append(b)

    return updated_boxes


def read_csv(csv_file):
    """
    Read annotation CSV file downloaded from Amazon MTurk
    """

    hits = []
    with open(csv_file, 'r') as fp:
        reader = csv.DictReader(fp)
        # skip header
        next(reader, None)
        for line in reader:
            if line['AssignmentStatus'] == "Rejected":
                continue
            boxes = json.loads(line['Answer.annotation_data'])
            hits.append({ "filename": line['Input.objects_to_find'], "boxes": boxes })

    return hits


def write_json(filename, hits):
    
    print "JSON written to %s" % filename
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hits)
    with open(filename, 'w') as outfile:
            outfile.write(json.dumps(hits, sort_keys=True, indent=4, separators=(',', ': ')))


def image_crop(orig_image_file, crop_image_file, box):

    os.path.exists(orig_image_file)

    im = Image.open(orig_image_file)
    piece = im.crop(box)
    img = Image.new('RGB', (box['height'], box['width']), 255)
    img.paste(piece)
    img.save(crop_image_file)


#def draw_box(filename, boxes, image_path, save = False):
#
#    """
#    Draw annotated box on an image, show and save an image
#    """
#
#    fullpath = os.path.join(image_path, filename)
#    im = np.array(Image.open(fullpath), dtype=np.uint8)
#    fig, ax = plt.subplots(1)
#    ax.imshow(im)
#
#    for box in boxes:
#        rect = patches.Rectangle((box["left"], box["top"]), box["width"], box["height"],
#                                linewidth=1, edgecolor='r', facecolor='none')
#        ax.add_patch(rect)
#        ax.annotate(box['label'], (box["left"] + 25, box["top"] - 25), color='red', weight='bold',
#                    fontsize=15, ha='center', va='center')
#    #plt.show()
#
#    if save:
#        name = filename.split('.')[0]
#        annotated_image = name + '-annotate.png'
#        print "Saved to %s" % annotated_image
#        fig.savefig(annotated_image)
#    plt.close(fig)
#    plt.close('all')


def split_box_per_number(box):

    assert type(box) is dict
    
    label = box['label']
    numbers = len(str(label))
    start_x = box['left']
    start_y = box['top']
    splitted_width = box['width'] / numbers
    for l in str(label):
        yield { 'label': l,
                'left': start_x,
                'top': start_y,
                'height': box['height'],
                'width': splitted_width }
        start_x = start_x + splitted_width


def is_intersected(label_box, box):
    """
    decide is given box intersected with another box
    returns overlapArea
    """
    # https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    # returns None if rectangles don't intersect
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    
    xmin = label_box['left']
    ymin = label_box['top'] - label_box['height']
    xmax = label_box['left'] + label_box['width']
    ymax = label_box['top']
    ra = Rectangle(xmin, ymin, xmax, ymax)
    xmin = box['left']
    ymin = box['top'] - box['height']
    xmax = box['left'] + box['width']
    ymax = box['top']
    rb = Rectangle(xmin, ymin, xmax, ymax)

    dx = min(ra.xmax, rb.xmax) - max(ra.xmin, rb.xmin)
    dy = min(ra.ymax, rb.ymax) - max(ra.ymin, rb.ymin)
    if (dx>=0) and (dy>=0):
	return dx*dy
        

def label_with_max_intersection(boxes, box):

    overlapArea = 0
    for b in boxes:
        overlap = is_intersected(b, box)
        if overlap > overlapArea:
            overlapArea = b['label']

    return overlapArea


def image_crop(path_to_image, box_width, box_height):

    # https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    # разделить изображение на одинаковые участки и вернуть список боксов для таких участков
    # в формате {"height": 174, "width": 59, "top": 956, "left": 1692, "label": "5" }

    im = Image.open(path_to_image)
    img_width, img_height = im.size
    for y_pos in range(int(round(img_width // box_width))):
        for x_pos in range(int(round(img_height // box_height))):
            # { 'height': 174, 'width': 59, 'top': 956, 'left': 1692, 'label': '' }
            box = { 'height': box_height, 'width': box_width,
                    'top': y_pos*box_height, 'left': x_pos*box_width, 'label': '' }
            yield box


def prepare_dir_struct(base_dir, train_dir, validation_dir, test_dir):

    """
    Create directories:
    race_numbers/{train, validation,test}/{0,1,2,3,4,5,6,7,8,9}

    """

    os.mkdir(base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    for number in range(0, 9):
        train_num_dir = os.path.join(train_dir, str(number))
        os.mkdir(train_num_dir)

        validation_num_dir = os.path.join(validation_dir, str(number))
        os.mkdir(validation_num_dir)

        test_num_dir = os.path.join(test_dir, str(number))
        os.mkdir(test_num_dir)


def main():

    base_dir = './data/race_numbers'
    print "Reading CSV results %s" % settings.CSV_RESULTS
    hits = read_csv(settings.CSV_RESULTS)

    # split images on single number in annotation
    annotation = []
    for image in hits[10]:
        for box in image['boxes']:
            boxes = [ b for b in split_box_per_number(box) ]
            annotation.extend('filename': image['filename'], 'boxes': boxes)

    box_w, box_h = box_dim_percentile(annotation, 75)

    # For learning we should have images with same fixed size.
    # Split images on images with box_w and box_h size. 
    # update labels according to intersections with boxes from annotation

    dataset_annotation = []
    number = 1
    for image in len(annotation):
        path_to_image = os.path.join(settings.ORIGINAL_IMAGES, image['filename'])
        for box in image_crop(path_to_image, box_w, box_h):
            box['label'] = label_with_max_intersection(image['boxes'], box)
            if box['label']:
                dataset_annotation.append({ 'filename': image['filename'], 'boxes': [box] })
        number += 1
        print "[%s/%s] Processing %s" % (number, len(annotation), path_to_image)

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    prepare_dir_struct('data/race_numbers', train_dir, validation_dir, test_dir)
    #assert (settings.TRAIN_NUM + settings.TEST_NUM + settings.VALIDATE_NUM) == len(dataset_annotation)
    random.shuffle(dataset_annotation)
    for image in dataset_annotation[settings.TRAIN_NUM]:
        filename = image['filename'].split('.')[0] + id_generator() + '.jpg'
        image['filename'] = filename
        number = image['boxes'][0]['label']

        src_image = os.path.join(settings.ORIGINAL_IMAGES, image['filename'])
        dst_image = os.path.join(train_dir, str(number), filename)
        image_crop(src_image, dst_image, box)

    print('total training images:', len(os.listdir(train_num_dir)))
    print('total validation images:', len(os.listdir(validation_num_dir)))
    print('total test images:', len(os.listdir(test_num_dir)))

    write_json('annotation.json', dataset_annotation)
    write_json('hits.json', hits)
    print "Update width %s and heigth %s in settings.py" % (box_w, box_h)

if __name__ == '__main__':
    main()
