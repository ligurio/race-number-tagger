# -*- coding: utf-8 -*-

from PIL import Image
import json
import os
import sys
import numpy as np
from collections import namedtuple

annotation_file = sys.argv[1]
annotation = json.load(open(annotation_file))
path_to_images = 'mturk_1_100'
mean_width = 53     # calculated by json-calc-average-size.py
mean_height = 118   # calculated by json-calc-average-size.py


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
        

def label_from_max_intersection(boxes, box):
    overlapArea = 0
    for b in boxes:
        overlap = is_intersected(b, box)
        if overlap > overlapArea:
            overlapArea = b['label']

    return overlapArea


def crop(path_to_image, box_width, box_height):

    # https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    # разделить изображение на боксы и вернуть список этих боксов
    # в формате {"height": 174, "width": 59, "top": 956, "left": 1692, "label": "5" }

    im = Image.open(path_to_image)
    img_width, img_height = im.size
    print "Processing", path_to_image, img_width, "x", img_height
    for y_pos in range(img_width // box_width):
        for x_pos in range(img_height // box_height):
            # { 'height': 174, 'width': 59, 'top': 956, 'left': 1692, 'label': '' }
            box = { 'height': box_height, 'width': box_width,
                    'top': y_pos*box_height, 'left': x_pos*box_width, 'label': '' }
            yield box


def main():

    new_annotation = []
    for image in annotation:
        path_to_image = os.path.join(path_to_images, image['filename'])
        img = { 'filename': image['filename'], 'boxes': [] }
        for k, box in enumerate(crop(path_to_image, mean_height, mean_width)):
            box['label'] = label_with_max_intersection(image['boxes'], box)
            img['boxes'].append(box)
        print img['filename'], len(img['boxes'])
        new_annotation.append(img)

        #im = Image.open(infile)
        #piece = im.crop(box)
        #img = Image.new('RGB', (mean_height, mean_width), 255)
        #img.paste(piece)
        #path = os.path.join('/tmp',"IMG-%s.png" % k)
        #img.save(path)

    # TODO: write boxes with labels only
    with open('single-box-new-dim.json', 'w') as outfile:
        outfile.write(json.dumps(new_annotation, ensure_ascii=False))
    print "Written to single-box-new-dim.json"


if __name__ == '__main__':
    main()
