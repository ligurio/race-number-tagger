# -*- coding: utf-8 -*-

#
#

import settings

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


def draw_box(boxes, src_image, dst_image = None):
    """
    Draw boxes on an image
    """

    im = np.array(Image.open(src_image), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        rect = patches.Rectangle((box["left"],box["top"]),box["width"],box["height"],
                                linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.annotate(box['label'], (box["left"] + 25, box["top"] - 25), color='red', weight='bold',
                    fontsize=15, ha='center', va='center')
    #plt.show()
    fig.savefig(dst_image)
    plt.close(fig)
    plt.close('all')


def id_generator():
    return ''.join([ random.choice(string.ascii_letters + string.digits) for n in xrange(32) ])


def box_dim_percentile(hits, percentile):

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
    with open(filename, 'w') as outfile:
            outfile.write(json.dumps(hits, sort_keys=True, indent=4, separators=(',', ': ')))


def image_crop(orig_image_file, crop_image_file, box):

    assert os.path.exists(orig_image_file)

    w, h = img_size(orig_image_file)
    area = (box['left'], box['top'] - box['height'], box['left'] + box['width'], box['top'])
    size = (int(box['width']), int(box['height']))
    piece = im.crop(area)
    img = Image.new('RGB', size, (255,255,255,0))
    img.paste(piece)
    img.save(crop_image_file)


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


def overlap_area(box1, box2):
    """
    Decide is given box intersected with another box
    :returns overlapArea or None if rectangles don't intersect
    """
    
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    ra = Rectangle(box1['left'], box1['top'] - box1['height'], box1['left'] + box1['width'], box1['top'])
    rb = Rectangle(box2['left'], box2['top'] - box2['height'], box2['left'] + box2['width'], box2['top'])
    dx = min(ra.xmax, rb.xmax) - max(ra.xmin, rb.xmin)
    dy = min(ra.ymax, rb.ymax) - max(ra.ymin, rb.ymin)
    if (dx >= 0) and (dy >= 0):
	return dx * dy
        

def max_overlap_label(boxes, box, threshold):
    """
    take a label from box which has max overlap area with our box
    :returns string with label
    """
    maxArea = threshold
    label = ''
    for b in boxes:
        area = overlap_area(b, box)
        if area > maxArea:
            maxArea = area
            label = b['label']

    return maxArea, label


def image_split(img_width, img_height, box_width, box_height):

    # https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    # разделить изображение на одинаковые участки и вернуть список боксов для таких участков
    # в формате {"height": 174, "width": 59, "top": 956, "left": 1692, "label": "5" }

    for y_pos in range(int(round(img_height // box_height))):
        for x_pos in range(int(round(img_width // box_width))):
            box = { 'height': box_height, 'width': box_width,
                    'top': y_pos*box_height, 'left': x_pos*box_width, 'label': '' }
            yield box


def build_dateset(annotation, base_dir):
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

    num_train_images = len(annotation) * settings.IMAGE_SUBSETS['train']/100
    num_validate_images = len(annotation) * settings.IMAGE_SUBSETS['validate']/100
    num_test_images = len(annotation) * settings.IMAGE_SUBSETS['test']/100
    random.shuffle(annotation)
    num = 1
    for image in annotation:
        if num < num_train_images:
            subdir = train_dir
        elif num >= num_train_images and num <= num_train_images + num_validate_images:
            subdir = validation_dir
        else:
            subdir = test_dir
        filename = image['filename'].split('.')[0] + '-' + id_generator() + '.jpg'
        src_image = os.path.join(settings.ORIGINAL_IMAGES, image['filename'])
        dst_image = os.path.join(subdir, str(image['boxes'][0]['label']), filename)
        print "[%s/%s] Put %s to the data set" % (num, len(annotation), dst_image)
        #draw_box(image['boxes'], src_image, dst_image)
        image_crop(src_image, dst_image, image['boxes'][0])
        num += 1

    return len([ f for f in find_files(train_dir, '*.jpg') ]), \
           len([ f for f in find_files(validation_dir, '*.jpg') ]), \
           len([ f for f in find_files(test_dir, '*.jpg') ])


def img_size(path_to_image):
    im = Image.open(path_to_image)
    img_width, img_height = im.size
    im.close
    return img_width, img_height


def images_to_pieces(annotation, box_w, box_h, base_dir):

    pieces_annotation = []
    number = 1
    for image in annotation:
        path_to_image = os.path.join(settings.ORIGINAL_IMAGES, image['filename'])
        print "[%s/%s] Splitting image %s" % (number, len(annotation), path_to_image)
        img_width, img_height = img_size(path_to_image)
        permitted_labels = [ b['label'] for b in image['boxes']]
        print "Permitted labels", permitted_labels

        threshold = box_w * box_h * 0.5
        for box in image_split(img_width, img_height, box_w, box_h):
            area, label = max_overlap_label(image['boxes'], box, threshold)
            if label and label.isdigit():
                assert(label in permitted_labels)
                box['label'] = label
                print "Found label %s, overlap area %s" % (box, area)
                pieces_annotation.append({ 'filename': image['filename'], 'boxes': [box] })
                # keep images with piece overlapped with a number box
                filename = image['filename'].split('.')[0] + '-' + id_generator() + '.jpg'
                dst_image = os.path.join(base_dir, 'check',  filename)
                create_dir(os.path.join(base_dir, 'check'))
                box_list = []
                box_list.extend(image['boxes'])
                box_list.append(box)
                draw_box(box_list, path_to_image, dst_image)
        number += 1

    return pieces_annotation


def main():

    base_dir = './data/race_numbers/'
    print "Reading CSV results %s" % settings.CSV_RESULTS
    hits = read_csv(settings.CSV_RESULTS)

    annotation = []
    for image in hits:
        for box in image['boxes']:
            boxes = [ b for b in split_box_per_number(box) ]
            annotation.append({ 'filename': image['filename'], 'boxes': boxes })
    box_w, box_h = box_dim_percentile(annotation, 75)

    # For learning we should have images with same fixed size.
    # Split original images on small images with fixed size (box_w x box_h)
    # and update labels according to intersections with boxes from annotation
    dataset_annotation = images_to_pieces(annotation, box_w, box_h, base_dir)
    assert (len(dataset_annotation) != 0)

    # put images in directories
    num_train, num_validation, num_test = build_dateset(dataset_annotation, base_dir)
    print 'total training images:', num_train
    print 'total validation images:', num_validation
    print 'total test images:', num_test

    # show information about processing annotation
    print "total number of hits in original annotation", len(hits)
    write_json('hits.json', hits)
    write_json('hits-splitted.json', annotation)
    print "total number of image in updated annotation", len(dataset_annotation)
    write_json('hits-dataset.json', dataset_annotation)
    print "update width %s and height %s in settings.py" % (box_w, box_h)


if __name__ == '__main__':
    main()
