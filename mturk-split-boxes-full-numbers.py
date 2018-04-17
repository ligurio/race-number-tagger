# -*- coding: utf-8 -*-

"""
на вход принимает аннотацию в JSON формате
и записывает в другой файл аннотацию с отдельным боксом для каждой цифры в номере

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import urllib
import os
import json
import sys

annotation_file = sys.argv[1]
annotation = json.load(open(annotation_file))


def draw_box(filename, boxes, image_path, save = False):

    """
    Draw annotated box on an image
    """

    fullpath = os.path.join(image_path, filename)
    print fullpath
    im = np.array(Image.open(fullpath), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    print "BOX:", boxes

    for box in boxes:
        print "\t", box
        print box["left"], box["top"]
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


def split_box(box):
    """
    :param box - dict with label, top, left, width and height keys

    """
    
    label = box['label']
    numbers = len(label)
    #print label
    start_x = box['left']
    start_y = box['top']
    splitted_width = box['width'] / numbers
    boxes = []
    for l in label:
        box = { 'label': l, 'left': start_x, 'top': start_y, 'height': box['height'], 'width': splitted_width }
        boxes.append(box)
        start_x = start_x + splitted_width

    return boxes


updated_annotation = []
for image in annotation:
    boxes = []
    for box in image['boxes']:
        boxes.extend(split_box(box)) 
    updated_annotation.append({ 'filename': image['filename'], 'boxes': boxes })
    #print 'draw_box', boxes
    draw_box(image['filename'], boxes, './mturk_1_100/')
    #try:
    #    draw_box(image['filename'], boxes, './')
    #except AttributeError:
    #    continue

print updated_annotation
#print json.dumps(updated_annotation, sort_keys=True, indent=4, separators=(',', ': '))
with open('single-box.json', 'w') as outfile:
        outfile.write(json.dumps(updated_annotation, ensure_ascii=False))
