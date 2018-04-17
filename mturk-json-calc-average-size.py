# -*- coding: utf-8 -*-

"""
на вход принимает аннотацию в JSON формате и вычисляет размер боксов по 75 перцентилю

"""

import json
import numpy as np
import sys

annotation_file = sys.argv[1]
annotation = json.load(open(annotation_file))
width = []
height = []
for image in annotation:
    for box in image['boxes']:
        width.append(box['width'])
        height.append(box['height'])

w = np.array(width)
print 'Width: 75th percentile %s, mean average %s' % (np.percentile(w, 75), np.mean(width))

h = np.array(height)
print 'Heigth: 75th percentile %s, mean average %s' % (np.percentile(h, 75), np.mean(height))
