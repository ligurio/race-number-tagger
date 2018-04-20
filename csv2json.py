# -*- coding: utf-8 -*-

#
#

import csv
import json
import os
import sys
import json
import pprint
import numpy as np
from PIL import Image
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import settings


def mean_box_dimension(hits):
    """
    на вход принимает аннотацию в JSON формате
    и вычисляет размер боксов по 75 перцентилю
    """
    width = []
    height = []
    for image in hits:
        for box in image['boxes']:
            width.append(box['width'])
            height.append(box['height'])

    w = np.array(width)
    #print 'Width: 75th percentile %s, mean average %s' % (np.percentile(w, 75), np.mean(width))
    h = np.array(height)
    #print 'Heigth: 75th percentile %s, mean average %s' % (np.percentile(h, 75), np.mean(height))
    return np.percentile(w, 75), np.percentile(h, 75)


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


def read_mturk_csv(csv_file):

    hits = []
    with open(csv_file, 'r') as fp:
        reader = csv.DictReader(fp)
        header = next(reader, None)
        for line in reader:
            status = line['AssignmentStatus']
            if status == "Rejected":
                continue
            filename = line['Input.objects_to_find']
            # FIXME: почему-то пропускает некоторые изображения
            try:
                boxes = sanitize_boxes(json.loads(line['Answer.annotation_data']))
            except ValueError:
                continue
            hits.append({"filename": filename, "boxes": boxes})

    return hits


def write_json(filename, hits):
    
    print "Written to %s" % filename
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hits)
    with open(filename, 'w') as outfile:
            #outfile.write(json.dumps(hits, ensure_ascii=False))
            outfile.write(json.dumps(updated_annotation, sort_keys=True, indent=4, separators=(',', ': ')))


def crop_box(image_file, path_to_box_image, box):

    im = Image.open(image_file)
    piece = im.crop(box)
    img = Image.new('RGB', (box['height'], box['width']), 255)
    img.paste(piece)
    img.save(box_image)


def draw_box(filename, boxes, image_path, save = False):

    """
    Draw annotated box on an image and show an image
    """

    fullpath = os.path.join(image_path, filename)
    im = np.array(Image.open(fullpath), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        rect = patches.Rectangle((box["left"], box["top"]), box["width"], box["height"],
                                linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.annotate(box['label'], (box["left"] + 25, box["top"] - 25), color='red', weight='bold',
                    fontsize=15, ha='center', va='center')
    #plt.show()

    if save:
        name = filename.split('.')[0]
        annotated_image = name + '-annotate.png'
        print "Saved to %s" % annotated_image
        fig.savefig(annotated_image)
    plt.close(fig)
    plt.close('all')


def split_box_per_number(box):
    """
    на вход принимает словарь для бокса,
    на выходе список словарей с боксами
    :param box - dict with label, top, left, width and height keys

    """
    
    label = box['label']
    numbers = len(str(label))
    start_x = box['left']
    start_y = box['top']
    splitted_width = box['width'] / numbers
    boxes = []
    for l in str(label):
        box = { 'label': l, 'left': start_x, 'top': start_y, 'height': box['height'], 'width': splitted_width }
        boxes.append(box)
        start_x = start_x + splitted_width

    return boxes


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


def crop(path_to_image, box_width, box_height):

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


def main():

    results = settings.CSV_RESULTS
    hits = read_mturk_csv(results)
    
    """
    на вход принимает аннотацию в JSON формате и записывает в другой файл
    аннотацию с отдельным боксом для каждой цифры в номере
    """
    updated_annotation = []
    for image in hits:
        boxes = []
        for box in image['boxes']:
            boxes.extend(split_box_per_number(box)) 
        updated_annotation.append({ 'filename': image['filename'], 'boxes': boxes })
        #draw_box(image['filename'], boxes, './mturk_1_100/')

    # calculate mean box size with single number (mtuk-json-calc-average-size.py)
    box_w, box_h = mean_box_dimension(updated_annotation)

    # split images on rectangles with fixed size and
    # update labels according to intersections with boxes from annotation
    # (mturk-box-intersection.py)

    new_annotation = []
    total_number = len(updated_annotation)
    number = 1
    for image in updated_annotation:
        path_to_image = os.path.join(settings.ORIGINAL_IMAGES, image['filename'])
        img = { 'filename': image['filename'], 'boxes': [] }
        for k, box in enumerate(crop(path_to_image, box_h, box_w)):
            box['label'] = label_with_max_intersection(image['boxes'], box)
            img['boxes'].append(box)
        print "[%s/%s] Processing %s" % (number, total_number, path_to_image)
        new_annotation.append(img)
        number+=1
        #crop_box(path_to_image, path_to_box_image, box):

    # TODO: write boxes with labels only
    write_json('annotation.json', new_annotation)


if __name__ == '__main__':
    main()
