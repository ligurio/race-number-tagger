# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import os
import sys
import string
import json
import pprint
import random
import numpy as np
from PIL import Image
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fnmatch


def create_dir(path):

    if not os.path.exists(path):
        print "create dir", path
        os.makedirs(path)


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def draw_box(boxes, src_image, dst_image=None):
    """
    Draw boxes on an image
    :param boxes - list with box dicts
    :param src_image - image where boxes will be drawn
    :param dst_image - save image with drawn boxes on this path
    :returns none
    """

    im = np.array(Image.open(src_image), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        rect = patches.Rectangle((box["left"], box["top"]), box["width"], box["height"],
                                linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.annotate(box['label'], (box["left"] + 25, box["top"] - 25), color='red', weight='bold',
                    fontsize=15, ha='center', va='center')
    fig.savefig(dst_image)
    plt.close(fig)
    plt.close('all')


def is_intersected(box1, box2):
    """
    Decide is a box intersected by another one or not.
    :param box1 - box dict
    :param box2 - box dict
    :returns True of False
    """

    h_overlaps = (box1['left'] <= box2['left'] + box2['width']
                  ) and (box1['left'] + box1['width'] >= box2['left'])
    v_overlaps = (box1['top'] - box1['height'] <= box2['top']
                  ) and (box1['top'] >= box2['top'] - box2['height'])

    return h_overlaps and v_overlaps


def id_generator():
    """
    Generate random string
    :returns string
    """
    return ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(32)])


def box_dim_percentile(hits, percentile):
    """
    Calculate specified percentile of width and heights
    for a list of boxes
    :param hits - list of hits
    :param percentile - percentile value
    :returns pair of width and height
    """

    width = []
    height = []
    for image in hits:
        for box in image['boxes']:
            width.append(box['width'])
            height.append(box['height'])

    return np.percentile(np.array(width), percentile), np.percentile(np.array(height), percentile)


def read_csv(csv_file):
    """
    Read annotation CSV file downloaded from Amazon MTurk
    :param csv_file - path to the file in csv format
    :returns dict with hit
    """

    with open(csv_file, 'r') as fp:
        reader = csv.DictReader(fp)
        # skip header
        next(reader, None)
        for line in reader:
            if line['AssignmentStatus'] == "Rejected":
                continue
            boxes = json.loads(line['Answer.annotation_data'])
            yield {"filename": line['Input.objects_to_find'], "boxes": boxes}


def write_json(filename, hits):

    print "JSON written to %s" % filename
    with open(filename, 'w') as outfile:
            outfile.write(json.dumps(hits, sort_keys=True,
                          indent=4, separators=(',', ': ')))


def image_crop(box, orig_image_file, crop_image_file=None):

    assert os.path.exists(orig_image_file)

    w, h = img_size(orig_image_file)
    area = (box['left'], box['top'] - box['height'],
            box['left'] + box['width'], box['top'])
    size = (int(box['width']), int(box['height']))
    im = Image.open(orig_image_file)
    piece = im.crop(area)
    img = Image.new('RGB', size, (255, 255, 255, 0))
    img.paste(piece)
    if crop_image_file:
        img.save(crop_image_file)

    return img


def split_box_per_number(box):
    """
    Split annotated box with full number to a set of boxes with single number.
    :param box - dict of original box with full number
    :returns generator with box dicts
    """

    assert type(box) is dict

    label = box['label']
    start_x = box['left']
    splitted_width = box['width'] / len(str(label))
    for number in str(label):
        yield {'label': number,
                'left': start_x,
                'top': box['top'],
                'height': box['height'],
                'width': splitted_width}
        start_x = start_x + splitted_width


def overlap_area(box1, box2):
    """
    Decide is given box intersected with another box
    :param box1 - dict
    :param box2 - dict
    :returns overlapArea or 0 if rectangles don't intersect
    """

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    ra = Rectangle(box1['left'], box1['top'] - box1['height'],
                   box1['left'] + box1['width'], box1['top'])
    rb = Rectangle(box2['left'], box2['top'] - box2['height'],
                   box2['left'] + box2['width'], box2['top'])
    dx = min(ra.xmax, rb.xmax) - max(ra.xmin, rb.xmin)
    dy = min(ra.ymax, rb.ymax) - max(ra.ymin, rb.ymin)
    if (dx >= 0) and (dy >= 0):
       return dx * dy
    else:
        return 0


def max_overlap_box(boxes, box, threshold=0):
    """
    Find a box with max overlap with box from a list
    :param boxes - list of boxes
    :param box - box
    :returns box dict and overlap area
    """

    overlapBox = {}
    maxArea = threshold
    for b in boxes:
        area = overlap_area(b, box)
        if area > maxArea:
            maxArea = area
            overlapBox = b

    return overlapBox, maxArea


def image_split(img_width, img_height, box_width, box_height):
    """
    Split image for a smaller parts.
    :param img_width - width of the original image
    :param img_height - height of the original image
    :param box_width - width of an image to split
    :param box_height - height of an image to split
    :returns generator with box dicts
    """

    # https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    # разделить изображение на одинаковые участки и вернуть список боксов для таких участков
    # в формате {"height": 174, "width": 59, "top": 956, "left": 1692, "label": "5" }
    # Участки пересекаются на box_width/2 по ширине и на box_height/2 по высоте

    for y_pos in range(int(round(img_height // box_height)) * 2):
        for x_pos in range(int(round(img_width // box_width)) * 2):
            box = {'height': box_height, 'width': box_width,
                    'top': y_pos * box_height / 2, 'left': x_pos * box_width / 2, 'label': ''}
            yield box


def build_dataset(annotation, base_dir, orig_images_dir,
            train_ratio, validate_ratio, test_ratio):
    """

    """
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    # create subdirs:
    # race_numbers/{train, validation,test}/{0,1,2,3,4,5,6,7,8,9}
    for number in range(0, 10):
        create_dir(os.path.join(train_dir, str(number)))
        create_dir(os.path.join(validation_dir, str(number)))
        create_dir(os.path.join(test_dir, str(number)))

    num_train_images = len(annotation) * train_ratio / 100
    num_validate_images = len(annotation) * validate_ratio / 100
    num_test_images = len(annotation) * test_ratio / 100
    num = 1
    for image in annotation:
        label = str(image['boxes'][0]['label'])
        if not label.isdigit():
            continue
        if num < num_train_images:
            subdir = train_dir
        elif num >= num_train_images and num <= num_train_images + num_validate_images:
            subdir = validation_dir
        else:
            subdir = test_dir
        filename = image['filename'].split(
            '.')[0] + '-' + id_generator() + '.jpg'
        src_image = os.path.join(orig_images_dir, image['filename'])
        dst_image = os.path.join(subdir, label, filename)
        print "[%s/%s] Put %s to the data set" % (num, len(annotation), dst_image)
        image_crop(image['boxes'][0], src_image, dst_image)
        num += 1

    return len([f for f in find_files(train_dir, '*.jpg')]), \
           len([f for f in find_files(validation_dir, '*.jpg')]), \
           len([f for f in find_files(test_dir, '*.jpg')])


def img_size(path_to_image):
    """
    :param path_to_image - path to the image
    :returns pair of width and height
    """

    im = Image.open(path_to_image)
    img_width, img_height = im.size
    im.close

    return img_width, img_height


def main():

    annotation_path = 'data/annotation.json'

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, dest="csv",
                        help="csv results")
    parser.add_argument("--percentile", type=int, dest="percentile",
                        help="percentile")
    parser.add_argument("--orig_images_dir", type=str, dest="orig_images_dir",
	                help="directory with original images")
    parser.add_argument("--processed_images_dir", type=str, dest="processed_images_dir",
                        help="directory with processed images")
    parser.add_argument("--train_ratio", type=int, dest="train_ratio",
                        help="train ratio")
    parser.add_argument("--validate_ratio", type=int, dest="validate_ratio",
                        help="validate ratio")
    parser.add_argument("--test_ratio", type=int, dest="test_ratio",
                        help="test ratio")
    parser.add_argument('--calc_size', dest='calc_size', action='store_true',
                        help="calculate piece size only")
    args = parser.parse_args()

    print "Reading CSV results %s" % args.csv
    annotation = []
    for image in read_csv(args.csv):
        boxes = []
        for box in image['boxes']:
            boxes.extend([ b for b in split_box_per_number(box) ])
        annotation.append({ 'filename': image['filename'], 'boxes': boxes })
    box_w, box_h = box_dim_percentile(annotation, args.percentile)
    print "BOX_WIDTH - %s, BOX_HEIGHT - %s" % (box_w, box_h)
    if args.calc_size:
        return

    # For learning we should have images with same fixed size.
    # Split original images on small images with fixed size (box_w x box_h)
    # and update labels according to intersections with boxes from annotation
    number = 1
    dataset_annotation = []
    # overlapThreshold = box_w * box_h * 0.4
    for image in annotation:
        path_to_image = os.path.join(args.orig_images_dir, image['filename'])
        print "[%s/%s] Processing image %s" % (number, len(annotation), path_to_image)
        img_width, img_height = img_size(path_to_image)
        boxes = [ box for box in image_split(img_width, img_height, box_w, box_h) ]
        boxes = []
        boxes.extend(image['boxes'])
        for box in image_split(img_width, img_height, box_w, box_h):
            overlapBox, _ = max_overlap_box(image['boxes'], box)
            # overlapBox, _ = max_overlap_box(image['boxes'], box, overlapThreshold)
            if overlapBox:
                # boxes.append(box)
                box['label'] = overlapBox['label']
                dataset_annotation.append({ 'filename': image['filename'], 'boxes': [box] })
        number += 1

        filename = image['filename'].split('.')[0] + '-' + id_generator() + '.jpg'
        dst_image = os.path.join(args.processed_images_dir, 'check',  filename)
        create_dir(os.path.join(args.processed_images_dir, 'check'))
        draw_box(boxes, os.path.join(args.orig_images_dir, image['filename']), dst_image)

    print "BOX_WIDTH - %s, BOX_HEIGHT - %s" % (box_w, box_h)
    random.shuffle(dataset_annotation)

    num_train, num_validation, num_test = build_dataset(dataset_annotation, processed_images_dir,
							args.orig_images_dir, args.train_ratio,
							args.validate_ratio, args.test_ratio)
    print 'total training images:', num_train
    print 'total validation images:', num_validation
    print 'total test images:', num_test

    # show information about processing annotation
    write_json(annotation_path, annotation)
    print "total number of images in annotation with box per number", len(annotation)
    print "total number of images in dataset annotation", len(dataset_annotation)
    print "BOX_WIDTH - %s, BOX_HEIGHT - %s" % (box_w, box_h)


if __name__ == '__main__':
    main()
