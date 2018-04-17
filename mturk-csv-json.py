# -*- coding: utf-8 -*-

"""
конвертирует csv в json для обучения
"""

import csv
import json
import os
import sys
import json
import pprint

results = sys.argv[1]

def write_json(filename, hits):

    with open(filename, 'w') as outfile:
        #pprint.pprint(hits, outfile)
        json.dumps(hits, outfile)


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


pp = pprint.PrettyPrinter(indent=4)
hits = []
with open(results, 'r') as fp:
    reader = csv.DictReader(fp)
    header = next(reader, None)
    for line in reader:
        status = line['AssignmentStatus']
        #if status == "Rejected":
        #    continue
        filename = line['Input.objects_to_find']
        #if filename == "0001_1682997":
        #    print "sdfsdfsdfsdfsdf"
        #boxes = sanitize_boxes(json.loads(line['Answer.annotation_data']))
        boxes = json.loads(line['Answer.annotation_data'])
        hits.append({"filename": filename, "boxes": boxes})

    json_file = os.path.basename(results).split('.')[0] + '-learning.json'
    #write_json(json_file, hits)
    #pp.pprint(hits)
    print json.dumps(hits)
    #json.dumps(hits, outfile)
    #print hits[2]

    #print "Written to ", json_file
    #json.load(open(json_file))
