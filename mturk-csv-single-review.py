# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import csv
import json
import urllib
import sys
import os

"""
для запуска по одной фотке из консоли
сделал чтобы обойти ограничение на размер памяти процесса в openbsd

"""

RESULTS = '../Batch_3183602_batch_results-2-1000.csv'

def draw_box(filename, boxes):

    # {"left":1141,"top":1131,"width":258,"height":160,"label":"1912"}
    im = np.array(Image.open(filename), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        print "\t", box
        if not box['label'].isdigit():
            print "Wrong label %s" % box['label']
            invalid_annotations.append(filename)
            continue
        rect = patches.Rectangle((box["left"],box["top"]),box["width"],box["height"],
                                linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.annotate(box['label'], (box["left"] + 25, box["top"] - 25), color='red', weight='bold',
                    fontsize=15, ha='center', va='center')
    #plt.show()
    name = filename.split('.')[0]
    annotated_image = name + '-annotate.png'
    print "Saved to %s" % annotated_image
    fig.savefig(annotated_image)
    plt.close(fig)
    #plt.close('all')


name = sys.argv[1]
print name
with open(RESULTS, 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    next(reader, None)

    for line in reader:
        boxes = json.loads(line[-2]);
        filename = line[-4]
        url = line[-3]
        if filename == name:
            print filename, url
            draw_box(filename, boxes)
