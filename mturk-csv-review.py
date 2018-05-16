# -*- coding: utf-8 -*-

"""
Принимает два параметра - файл с результатами в csv и путь к изображениям.
На каждой фотографии рисует аннотированные боксы и сохраняет на диск.
"""

import csv
import json
import sys
import os
from bib_prepare_dataset import draw_box
from bib_prepare_dataset import create_dir

RESULTS = sys.argv[1]
IMAGE_PATH = sys.argv[2]

def sanitize(box):

    if not box['label'].isdigit():
        return False
    else:
        return True


def print_usage():
    print """
Tool to review MTurk annotation file.
Enter two parameters: path to a CSV file and path to an images
"""

def main():

    if len(sys.argv) == 1:
        print_usage()
    
    basename = os.path.basename(os.path.abspath(IMAGE_PATH))
    dst_dir = os.path.join(os.path.dirname(os.path.abspath(IMAGE_PATH)), basename + "-review")
    create_dir(dst_dir)
    print "Dir with boxes", dst_dir
    wrong_labels = []
    with open(RESULTS, 'r') as fp:
        reader = csv.DictReader(fp)
        header = next(reader, None)
        for row in reader:
            boxes = json.loads(row['Answer.annotation_data'])
            for box in boxes:
                if not sanitize(box):
                    print 'Wrong label "%s", assignment id %s, worker id %s' \
                        % (box['label'], row['AssignmentId'], row['WorkerId'])
                    wrong_labels.append(row['AssignmentId'])

            filename = row['Input.objects_to_find']
            if row['AssignmentStatus'] == 'Rejected':
                continue 
            src_image = os.path.join(IMAGE_PATH, filename)
            if not os.path.exists(src_image):
                print "Image not found %s" % src_image
            
            dst_image = os.path.join(dst_dir, filename)
            print "Draw on %s" % src_image
            draw_box(boxes, src_image, dst_image)

    print wrong_labels

if __name__ == '__main__':
    main()
