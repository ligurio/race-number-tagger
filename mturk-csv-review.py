# -*- coding: utf-8 -*-

"""
Принимает два параметра - файл с результатами в csv и путь к изображениям.
На каждой фотографии рисует аннотированные боксы и сохраняет на диск.
"""

import csv
import json
import sys
import os
import logging
from bib_prepare_dataset import draw_box
from bib_prepare_dataset import create_dir


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

    if len(sys.argv) < 2:
        print_usage()

    RESULTS = sys.argv[1]       # CSV file with annotations
    IMAGE_PATH = sys.argv[2]    # path to annotated images

    basename = os.path.basename(os.path.abspath(IMAGE_PATH))
    dst_dir = os.path.join(os.path.dirname(
        os.path.abspath(IMAGE_PATH)), basename + "-review")
    create_dir(dst_dir)
    logging.info("Dir with boxes", dst_dir)
    wrong_labels = []
    with open(RESULTS, 'r') as fp:
        reader = csv.DictReader(fp)
        header = next(reader, None)
        for row in reader:
            boxes = json.loads(row['Answer.annotation_data'])
            if row['AssignmentStatus'] == 'Rejected':
                continue
            for box in boxes:
                if not sanitize(box):
                    logging.warning('Image %s, wrong label "%s", assignment id %s, worker id %s' %
                                    (row['Input.objects_to_find'], box['label'],
                                     row['AssignmentId'], row['WorkerId']))
                    wrong_labels.append(row['Input.objects_to_find'])
                    break
            filename = row['Input.objects_to_find']
            src_image = os.path.join(IMAGE_PATH, filename)
            if not os.path.exists(src_image):
                logging.warning("Image not found %s" % src_image)

            dst_image = os.path.join(dst_dir, filename)
            logging.info("Draw on %s" % src_image)
            draw_box(boxes, src_image, dst_image)

    print wrong_labels


if __name__ == '__main__':
    main()
